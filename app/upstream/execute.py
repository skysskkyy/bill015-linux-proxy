from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterable

import httpx
from fastapi import HTTPException

from ..bridge.parse import parse_emit_value
from ..bridge.payload import build_upstream_payload
from ..config import Settings, settings
from ..context.compact import compact_history
from ..context.pack import pack_turn_items
from ..ingest.web_intent import local_web_preflight
from ..ops.audit import redact
from ..protocol.ids import make_item_id
from ..protocol.models import BridgeToolCall, Turn, TurnResult, WrapperCall, local_response_id
from ..protocol.sse import SSEEvent, parse_async_sse_lines
from ..tools.loop import apply_proxy_tools, split_proxy_calls
from .client import NATIVE_USER_AGENT, http_timeout, upstream_auth_headers
from .errors import (
    RATE_LIMIT_RETRIES,
    RATE_LIMIT_RETRY_DELAY_SECONDS,
    is_context_length_exceeded,
    key_rotation_error_reason,
    retryable_upstream_error_reason,
    sanitize_upstream_error_detail,
)
from .keys import upstream_key_pool


def finalize_visible_answer(result: TurnResult, *, did_local_work: bool = False) -> None:
    if result.tool_calls:
        return
    if (result.answer or "").strip():
        return
    note = (result.commentary or "").strip()
    if note:
        result.answer = note
    elif did_local_work:
        result.answer = "The model ended this turn without a user-facing answer after local web research."


def merge_reasoning_item(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if not isinstance(item, dict) or item.get("type") != "reasoning":
        return
    rid = str(item.get("id") or "") or make_item_id("reasoning")
    summary = item.get("summary") if isinstance(item.get("summary"), list) else []
    content = item.get("content") if isinstance(item.get("content"), list) else []
    blob: dict[str, Any] = {
        "id": rid,
        "type": "reasoning",
        "status": item.get("status") or "completed",
        "summary": summary,
        "content": content,
    }
    encrypted = item.get("encrypted_content")
    if isinstance(encrypted, str) and encrypted:
        blob["encrypted_content"] = encrypted
    for existing in items:
        if existing.get("id") == rid:
            if blob.get("encrypted_content"):
                existing["encrypted_content"] = blob["encrypted_content"]
            if summary:
                existing["summary"] = summary
            if content:
                existing["content"] = content
            if item.get("status"):
                existing["status"] = item["status"]
            return
    items.append(blob)


def merge_reasoning_summary(
    items: list[dict[str, Any]],
    item_id: str,
    *,
    delta: str = "",
    text: str | None = None,
    summary_index: int = 0,
) -> None:
    rid = item_id or make_item_id("reasoning")
    target = next((item for item in items if item.get("id") == rid), None)
    if target is None:
        target = {"id": rid, "type": "reasoning", "status": "in_progress", "summary": [], "content": []}
        items.append(target)
    summary = target.get("summary")
    if not isinstance(summary, list):
        summary = []
        target["summary"] = summary
    while len(summary) <= summary_index:
        summary.append({"type": "summary_text", "text": ""})
    part = summary[summary_index]
    if not isinstance(part, dict):
        part = {"type": "summary_text", "text": str(part)}
        summary[summary_index] = part
    if text is not None:
        part["text"] = text
    elif delta:
        part["text"] = (part.get("text") or "") + delta


def push_live(turn: Turn, event: dict[str, Any]) -> None:
    sink = turn.event_sink
    if sink is None:
        return
    try:
        sink.put_nowait(event)
    except Exception:
        return


def merge_wrappers(wrappers: list[WrapperCall]) -> tuple[str, str, list[BridgeToolCall], bool]:
    commentary: list[str] = []
    answers: list[str] = []
    tools: list[BridgeToolCall] = []
    malformed = False
    for wrapper in wrappers:
        malformed = malformed or wrapper.malformed
        if wrapper.mode == "tool_call" and wrapper.tool_calls:
            tools.extend(wrapper.tool_calls)
            if wrapper.answer.strip():
                commentary.append(wrapper.answer.strip())
        elif wrapper.answer.strip():
            answers.append(wrapper.answer.strip())
    if tools:
        return "\n\n".join(answers), "\n\n".join(commentary), tools, malformed
    if answers:
        return "\n\n".join(answers), "\n\n".join(commentary), [], malformed
    return "", "\n\n".join(commentary), [], malformed


def collect_from_events(events: Iterable[SSEEvent], turn: Turn, cfg: Settings = settings) -> TurnResult:
    result = TurnResult(local_request_id=local_response_id())
    buffers: dict[str, list[str]] = {}
    open_calls: set[str] = set()
    wrappers: list[WrapperCall] = []
    text_buf: list[str] = []
    for ev in events:
        obj = ev.json
        typ = (obj or {}).get("type") if obj else ev.event
        if typ:
            result.event_sequence.append(str(typ))
        if not obj:
            continue
        if typ == "response.created":
            response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
            result.upstream_response_id = response.get("id") or obj.get("id")
        elif typ == "response.output_item.added":
            item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
            if item.get("type") == "reasoning":
                merge_reasoning_item(result.reasoning_items, item)
            elif item.get("name") == cfg.function_name or item.get("type") == "function_call":
                key = str(item.get("call_id") or item.get("id") or f"open-{len(open_calls)}")
                open_calls.add(key)
        elif typ == "response.output_item.done":
            item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
            if item.get("type") == "reasoning":
                merge_reasoning_item(result.reasoning_items, item)
        elif typ == "response.reasoning_summary_text.delta":
            delta = obj.get("delta") if isinstance(obj.get("delta"), str) else ""
            merge_reasoning_summary(
                result.reasoning_items,
                str(obj.get("item_id") or ""),
                delta=delta,
                summary_index=int(obj.get("summary_index") or 0),
            )
        elif typ == "response.reasoning_summary_text.done":
            text = obj.get("text") if isinstance(obj.get("text"), str) else None
            merge_reasoning_summary(
                result.reasoning_items,
                str(obj.get("item_id") or ""),
                text=text,
                summary_index=int(obj.get("summary_index") or 0),
            )
        elif typ == "response.function_call_arguments.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                key = str(obj.get("call_id") or obj.get("item_id") or "active")
                buffers.setdefault(key, []).append(delta)
        elif typ == "response.output_text.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                text_buf.append(delta)
        elif typ == "response.function_call_arguments.done":
            key = str(obj.get("call_id") or obj.get("item_id") or "active")
            raw = obj.get("arguments") if isinstance(obj.get("arguments"), str) else "".join(buffers.get(key) or [])
            wrappers.append(parse_emit_value(raw, turn.catalog, cfg))
            open_calls.discard(key)
            result.args_done_seen = True
            result.wrapper_count += 1
            if result.wrapper_count >= cfg.max_emit_value_calls:
                result.aborted = True
                break
        elif typ == "response.completed":
            result.upstream_completed_seen = True
            response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
            for item in response.get("output") or []:
                if isinstance(item, dict):
                    merge_reasoning_item(result.reasoning_items, item)
            if not wrappers and text_buf:
                result.answer = "".join(text_buf)
            break
        elif typ in {"response.failed", "error"}:
            result.error = json.dumps(obj, ensure_ascii=False)[:2000]
            break
    answer, commentary, tools, malformed = merge_wrappers(wrappers)
    result.answer = result.answer or answer
    result.commentary = commentary
    result.tool_calls = tools
    result.malformed = malformed
    result.raw_arguments = "\n".join(w.raw_arguments for w in wrappers)
    if result.args_done_seen:
        result.aborted = True
    return result


def _apply_wrappers(result: TurnResult, wrappers: list[WrapperCall]) -> None:
    answer, commentary, tools, malformed = merge_wrappers(wrappers)
    result.answer = answer
    result.commentary = commentary
    result.tool_calls = tools
    result.malformed = malformed
    result.raw_arguments = "\n".join(w.raw_arguments for w in wrappers)
    result.wrapper_count = len(wrappers)
    result.args_done_seen = bool(wrappers)


async def execute_turn(turn: Turn, mode: str = "exploit", cfg: Settings = settings, local_request_id: str | None = None) -> TurnResult:
    request_id = local_request_id or local_response_id()
    result = TurnResult(local_request_id=request_id)
    _ = mode
    start = time.perf_counter()
    has_proxy_web = any(spec.proxy for spec in turn.catalog.specs.values())
    preflight = None
    if cfg.tool_bridge_local_web_research_preflight and not has_proxy_web:
        preflight = local_web_preflight(turn.current_user, turn.catalog, turn.items)
    if preflight and not turn.is_compaction:
        result.tool_calls = [preflight]
        result.commentary = ""
        result.retry_reasons.append("local_web_research_preflight")
        result.duration_ms = int((time.perf_counter() - start) * 1000)
        return result

    selection = upstream_key_pool(cfg).current()
    if selection is None:
        raise HTTPException(status_code=500, detail=f"Missing upstream API key env {cfg.upstream_api_key_env}")
    tried = {selection.key}
    result.upstream_key_index = selection.index + 1
    result.upstream_key_count = selection.count

    max_retries = max(0, cfg.upstream_retries)
    transient = 0
    context_attempt = 0
    policy_rotations = 0
    proxy_rounds = 0
    rate_limit_attempts = 0
    rate_limit_cap = max(0, RATE_LIMIT_RETRIES)
    try:
        async with httpx.AsyncClient(timeout=http_timeout(cfg), headers={"User-Agent": NATIVE_USER_AGENT}) as client:
            headers = upstream_auth_headers(api_key=selection.key, cfg=cfg, client_headers=turn.request_headers)
            if turn.is_compaction:
                compacted = await compact_history(client, headers, turn, [item for item in turn.items if isinstance(item, dict)], cfg)
                if compacted is None:
                    raise HTTPException(status_code=502, detail="upstream compact did not return output")
                result.compacted_output = compacted
                result.args_done_seen = True
                result.retry_reasons.append("responses_compact")
                return result

            pack = pack_turn_items(turn, cfg)
            if pack.needs_compact:
                compacted = await compact_history(client, headers, turn, pack.items, cfg)
                if compacted:
                    turn.items = compacted
                    pack = pack_turn_items(turn, cfg)
                    result.compaction_count += 1
                    result.retry_reasons.append("responses_compact")
            turn.estimated_input_tokens = pack.tokens
            result.packed_item_count = len(pack.items)
            result.clipped_tool_output_count = pack.clipped_outputs
            result.loss_notices = pack.notices
            payload = build_upstream_payload(turn, cfg, packed_items=pack.items)

            while True:
                headers = upstream_auth_headers(api_key=selection.key, cfg=cfg, client_headers=turn.request_headers)
                buffers: dict[str, list[str]] = {}
                open_calls: dict[str, float] = {}
                reasoning_open: set[str] = set()
                reasoning_items: list[dict[str, Any]] = []
                wrappers: list[WrapperCall] = []
                text_buf: list[str] = []
                rate_limited = False
                deadline = time.perf_counter() + max(1.0, cfg.args_done_timeout_ms / 1000)
                last_event = time.perf_counter()
                attempt_events = len(result.event_sequence)
                try:
                    async with client.stream("POST", cfg.upstream_base_url + "/v1/responses", headers=headers, json=payload) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            detail = sanitize_upstream_error_detail(resp.status_code, body, content_type=resp.headers.get("content-type", ""))
                            retry_reason = retryable_upstream_error_reason(body)
                            if retry_reason and rate_limit_attempts < rate_limit_cap:
                                rate_limit_attempts += 1
                                result.retry_count += 1
                                result.retry_reasons.append(retry_reason)
                                await asyncio.sleep(RATE_LIMIT_RETRY_DELAY_SECONDS)
                                continue
                            if is_context_length_exceeded(body) and context_attempt < cfg.context_recovery_retries:
                                compacted = await compact_history(client, headers, turn, [item for item in turn.items if isinstance(item, dict)], cfg)
                                if compacted:
                                    turn.items = compacted
                                    result.retry_reasons.append("responses_compact")
                                pack = pack_turn_items(turn, cfg, aggressive=True)
                                turn.estimated_input_tokens = pack.tokens
                                payload = build_upstream_payload(turn, cfg, packed_items=pack.items)
                                context_attempt += 1
                                result.compaction_count += 1
                                result.retry_count += 1
                                result.retry_reasons.append("context_length_exceeded")
                                continue
                            reason = key_rotation_error_reason(body)
                            if reason and cfg.cyber_policy_rotate and policy_rotations < cfg.max_policy_rotations_per_request:
                                nxt = upstream_key_pool(cfg).rotate_after_failure(
                                    selection.key, tried, cooldown_seconds=cfg.failed_key_cooldown_seconds
                                )
                                if nxt is not None:
                                    if cfg.cyber_policy_retry_delay_seconds > 0:
                                        await asyncio.sleep(cfg.cyber_policy_retry_delay_seconds)
                                    tried.add(nxt.key)
                                    selection = nxt
                                    policy_rotations += 1
                                    result.key_switch_count += 1
                                    result.upstream_key_index = nxt.index + 1
                                    result.retry_reasons.append(reason)
                                    continue
                            if transient < max_retries and resp.status_code >= 500:
                                result.retry_count += 1
                                result.retry_reasons.append(f"http_{resp.status_code}")
                                await asyncio.sleep((cfg.upstream_retry_backoff_ms / 1000) * (transient + 1))
                                transient += 1
                                continue
                            raise HTTPException(status_code=502, detail=detail)

                        async for ev in parse_async_sse_lines(resp.aiter_lines()):
                            now = time.perf_counter()
                            if now > deadline:
                                await resp.aclose()
                                result.retry_reasons.append("args_done_timeout")
                                break
                            obj = ev.json
                            typ = (obj or {}).get("type") if obj else ev.event
                            if typ:
                                result.event_sequence.append(str(typ))
                                last_event = now
                            if not obj:
                                continue
                            if typ == "response.created":
                                response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
                                result.upstream_response_id = response.get("id") or obj.get("id")
                            elif typ == "response.output_item.added":
                                item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
                                if item.get("type") == "reasoning":
                                    key = str(item.get("id") or "")
                                    if key:
                                        reasoning_open.add(key)
                                    merge_reasoning_item(reasoning_items, item)
                                    current = next((row for row in reasoning_items if row.get("id") == (key or row.get("id"))), item)
                                    push_live(turn, {"kind": "reasoning_added", "item": dict(current)})
                                    result.reasoning_live = True
                                elif item.get("name") == cfg.function_name or item.get("type") == "function_call":
                                    key = str(item.get("call_id") or item.get("id") or f"open-{len(open_calls)}")
                                    open_calls[key] = now
                            elif typ == "response.output_item.done":
                                item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
                                if item.get("type") == "reasoning":
                                    merge_reasoning_item(reasoning_items, item)
                                    reasoning_open.discard(str(item.get("id") or ""))
                                    current = next((row for row in reasoning_items if row.get("id") == item.get("id")), item)
                                    push_live(turn, {"kind": "reasoning_done", "item": dict(current)})
                                    result.reasoning_live = True
                            elif typ == "response.reasoning_summary_text.delta":
                                delta = obj.get("delta") if isinstance(obj.get("delta"), str) else ""
                                item_id = str(obj.get("item_id") or "")
                                index = int(obj.get("summary_index") or 0)
                                merge_reasoning_summary(reasoning_items, item_id, delta=delta, summary_index=index)
                                push_live(turn, {"kind": "reasoning_summary_delta", "item_id": item_id, "delta": delta, "summary_index": index})
                                result.reasoning_live = True
                            elif typ == "response.reasoning_summary_text.done":
                                item_id = str(obj.get("item_id") or "")
                                index = int(obj.get("summary_index") or 0)
                                text = obj.get("text") if isinstance(obj.get("text"), str) else None
                                merge_reasoning_summary(reasoning_items, item_id, text=text, summary_index=index)
                                push_live(turn, {"kind": "reasoning_summary_done", "item_id": item_id, "text": text or "", "summary_index": index})
                                result.reasoning_live = True
                            elif typ == "response.function_call_arguments.delta":
                                delta = obj.get("delta")
                                if isinstance(delta, str):
                                    key = str(obj.get("call_id") or obj.get("item_id") or next(iter(open_calls), "active"))
                                    buffers.setdefault(key, []).append(delta)
                            elif typ == "response.output_text.delta":
                                delta = obj.get("delta")
                                if isinstance(delta, str):
                                    text_buf.append(delta)
                            elif typ == "response.function_call_arguments.done":
                                key = str(obj.get("call_id") or obj.get("item_id") or next(iter(open_calls), "active"))
                                raw = obj.get("arguments") if isinstance(obj.get("arguments"), str) else "".join(buffers.get(key) or [])
                                wrappers.append(parse_emit_value(raw, turn.catalog, cfg))
                                open_calls.pop(key, None)
                                result.args_done_seen = True
                                if len(wrappers) >= cfg.max_emit_value_calls and not open_calls and not reasoning_open:
                                    await resp.aclose()
                                    result.aborted = True
                                    break
                            elif typ in {"response.completed", "response.incomplete"}:
                                result.upstream_completed_seen = typ == "response.completed"
                                response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
                                for item in response.get("output") or []:
                                    if isinstance(item, dict):
                                        merge_reasoning_item(reasoning_items, item)
                                await resp.aclose()
                                result.aborted = bool(wrappers)
                                break
                            elif typ in {"response.failed", "error"}:
                                retry_reason = retryable_upstream_error_reason(obj)
                                if retry_reason:
                                    await resp.aclose()
                                    if rate_limit_attempts < rate_limit_cap:
                                        rate_limit_attempts += 1
                                        result.retry_count += 1
                                        result.retry_reasons.append(retry_reason)
                                        rate_limited = True
                                        wrappers = []
                                        break
                                    raise HTTPException(
                                        status_code=502,
                                        detail=sanitize_upstream_error_detail(503, obj),
                                    )
                                reason = key_rotation_error_reason(obj)
                                if reason and cfg.cyber_policy_rotate:
                                    await resp.aclose()
                                    nxt = upstream_key_pool(cfg).rotate_after_failure(
                                        selection.key, tried, cooldown_seconds=cfg.failed_key_cooldown_seconds
                                    )
                                    if nxt is not None and policy_rotations < cfg.max_policy_rotations_per_request:
                                        tried.add(nxt.key)
                                        selection = nxt
                                        policy_rotations += 1
                                        result.key_switch_count += 1
                                        result.retry_reasons.append(reason)
                                        wrappers = []
                                        break
                                raise RuntimeError(json.dumps(obj, ensure_ascii=False)[:2000])

                            quiet = cfg.emit_value_quiet_ms / 1000
                            if wrappers and not open_calls and not reasoning_open and (now - last_event) >= quiet:
                                await resp.aclose()
                                result.aborted = True
                                break

                        if rate_limited:
                            result.args_done_seen = False
                            result.tool_calls = []
                            result.answer = ""
                            result.commentary = ""
                            await asyncio.sleep(RATE_LIMIT_RETRY_DELAY_SECONDS)
                            continue
                        result.reasoning_items = reasoning_items
                        _apply_wrappers(result, wrappers)
                        if result.args_done_seen:
                            result.aborted = True
                            proxy_calls, client_calls = split_proxy_calls(result.tool_calls)
                            if proxy_calls and proxy_rounds < cfg.web_max_rounds_per_request:
                                await apply_proxy_tools(turn, proxy_calls, cfg)
                                proxy_rounds += 1
                                result.retry_reasons.append(f"proxy_web:{','.join(c.name for c in proxy_calls)}")
                                pack = pack_turn_items(turn, cfg)
                                turn.estimated_input_tokens = pack.tokens
                                payload = build_upstream_payload(turn, cfg, packed_items=pack.items)
                                result.tool_calls = []
                                result.answer = ""
                                result.commentary = ""
                                result.args_done_seen = False
                                result.reasoning_items = []
                                wrappers = []
                                continue
                            result.tool_calls = client_calls
                            finalize_visible_answer(result, did_local_work=bool(proxy_rounds))
                            break
                        if result.retry_reasons[-1:] == ["args_done_timeout"] and result.retry_count < cfg.stream_recovery_retries:
                            result.retry_count += 1
                            pack = pack_turn_items(turn, cfg, aggressive=True)
                            turn.estimated_input_tokens = pack.tokens
                            payload = build_upstream_payload(turn, cfg, packed_items=pack.items)
                            continue
                        if not result.answer and text_buf:
                            result.answer = "".join(text_buf)
                            break
                        if not result.answer and not result.tool_calls:
                            if transient < max(0, cfg.empty_stream_retries):
                                transient += 1
                                result.retry_count += 1
                                result.retry_reasons.append("empty_stream")
                                continue
                            result.answer = "The upstream stream closed before a complete emit_value batch arrived."
                            result.args_done_seen = True
                        break
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.TimeoutException) as e:
                    reason = type(e).__name__
                    if transient < max_retries and len(result.event_sequence) == attempt_events:
                        result.retry_count += 1
                        result.retry_reasons.append(reason)
                        await asyncio.sleep((cfg.upstream_retry_backoff_ms / 1000) * (transient + 1))
                        transient += 1
                        continue
                    if isinstance(e, httpx.ReadTimeout) and result.retry_count < cfg.stream_recovery_retries:
                        result.retry_count += 1
                        result.retry_reasons.append(reason)
                        continue
                    raise
        return result
    except HTTPException as e:
        e.bill015_result = result  # type: ignore[attr-defined]
        raise
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=502, detail=result.error) from e
    finally:
        result.duration_ms = int((time.perf_counter() - start) * 1000)


def dry_run_response(turn: Turn, mode: str = "dry-run") -> dict[str, Any]:
    pack = pack_turn_items(turn)
    turn.estimated_input_tokens = pack.tokens
    payload = build_upstream_payload(turn, packed_items=pack.items)
    safe = redact(payload)
    if not settings.store_prompts:
        for item in safe.get("input", []):
            if isinstance(item, dict) and item.get("role") == "user":
                item["content"] = f"<redacted prompt chars={len(turn.current_user)}>"
    return {
        "id": local_response_id(),
        "object": "response",
        "status": "completed",
        "mode": mode,
        "would_post": settings.upstream_base_url + "/v1/responses",
        "upstream_configured": settings.upstream_configured,
        "usage_estimate": {"input_tokens": pack.tokens, "output_tokens": 0},
        "payload": safe,
        "parallel_tool_calls": True,
        "emit_value_batch": True,
    }


def audit_from_result(result: TurnResult, turn: Turn, mode: str) -> dict[str, Any]:
    return {
        "local_request_id": result.local_request_id,
        "upstream_response_id": result.upstream_response_id,
        "mode": mode,
        "model": turn.model,
        "client_api": turn.client_api,
        "aborted": result.aborted,
        "args_done_seen": result.args_done_seen,
        "wrapper_count": result.wrapper_count,
        "tool_calls": [{"name": c.name, "call_type": c.call_type, "namespace": c.namespace} for c in result.tool_calls],
        "duration_ms": result.duration_ms,
        "retry_count": result.retry_count,
        "retry_reasons": result.retry_reasons,
        "upstream_key_index": result.upstream_key_index,
        "upstream_key_count": result.upstream_key_count,
        "key_switch_count": result.key_switch_count,
        "packed_item_count": result.packed_item_count,
        "clipped_tool_output_count": result.clipped_tool_output_count,
        "prompt_chars": len(turn.current_user),
        "answer_chars": len(result.answer or ""),
        "commentary_chars": len(result.commentary or ""),
        "reasoning_item_count": len(result.reasoning_items),
        "compaction_count": result.compaction_count,
        "error": result.error,
    }
