from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterable

import httpx
from fastapi import HTTPException

from ..bridge.parse import parse_emit_value
from ..bridge.payload import build_upstream_payload
from ..bridge.progress import should_continue_progress
from ..config import Settings, settings
from ..context.pack import pack_turn_items
from ..ingest.web_intent import local_web_preflight
from ..ops.audit import redact
from ..protocol.models import BridgeToolCall, Turn, TurnResult, WrapperCall, local_response_id, new_call_id
from ..protocol.sse import SSEEvent, parse_async_sse_lines
from ..tools.loop import apply_proxy_tools, split_proxy_calls
from .client import NATIVE_USER_AGENT, http_timeout, upstream_auth_headers
from .errors import (
    is_context_length_exceeded,
    key_rotation_error_reason,
    sanitize_upstream_error_detail,
)
from .keys import upstream_key_pool


def _progress_nudge(note: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "developer",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "[LOCAL TURN CONTINUATION] Previous emit_value was a progress note, not a final answer:\n"
                    f"{note}\n"
                    "Use mode=tool_call with local tools now. mode=answer only after the user request is fully handled."
                ),
            }
        ],
    }


def _progress_fallback_call(turn: Turn) -> BridgeToolCall | None:
    spec = turn.catalog.get("tool_search")
    if spec is None or spec.call_type != "tool_search":
        return None
    query = (turn.current_user or "continue the current task").strip()[:200] or "continue the current task"
    return BridgeToolCall(
        id=new_call_id(),
        name="tool_search",
        arguments="",
        call_type="tool_search",
        execution="client",
        search_arguments={"query": query, "limit": 12},
    )


def merge_wrappers(wrappers: list[WrapperCall]) -> tuple[str, str, list[BridgeToolCall], bool]:
    commentary: list[str] = []
    answers: list[str] = []
    tools: list[BridgeToolCall] = []
    malformed = False
    for wrapper in wrappers:
        malformed = malformed or wrapper.malformed
        if wrapper.mode == "tool_call":
            tools.extend(wrapper.tool_calls)
            if wrapper.answer.strip():
                commentary.append(wrapper.answer.strip())
        elif wrapper.mode == "progress":
            if wrapper.answer.strip():
                commentary.append(wrapper.answer.strip())
        else:
            if wrapper.answer.strip():
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
            if item.get("name") == cfg.function_name or item.get("type") == "function_call":
                key = str(item.get("call_id") or item.get("id") or f"open-{len(open_calls)}")
                open_calls.add(key)
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
        preflight = local_web_research_preflight(turn.current_user, turn.catalog, turn.items)
    if preflight and not turn.is_compaction:
        result.tool_calls = [preflight]
        result.commentary = ""
        result.retry_reasons.append("local_web_research_preflight")
        result.duration_ms = int((time.perf_counter() - start) * 1000)
        return result

    pack = pack_turn_items(turn, cfg)
    result.packed_item_count = len(pack.items)
    result.clipped_tool_output_count = pack.clipped_outputs
    result.loss_notices = pack.notices
    payload = build_upstream_payload(turn, cfg, packed_items=pack.items)

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
    progress_rounds = 0
    progress_notes: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=http_timeout(cfg), headers={"User-Agent": NATIVE_USER_AGENT}) as client:
            while True:
                headers = upstream_auth_headers(api_key=selection.key, cfg=cfg, client_headers=turn.request_headers)
                buffers: dict[str, list[str]] = {}
                open_calls: dict[str, float] = {}
                wrappers: list[WrapperCall] = []
                text_buf: list[str] = []
                deadline = time.perf_counter() + max(1.0, cfg.args_done_timeout_ms / 1000)
                last_event = time.perf_counter()
                attempt_events = len(result.event_sequence)
                try:
                    async with client.stream("POST", cfg.upstream_base_url + "/v1/responses", headers=headers, json=payload) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            detail = sanitize_upstream_error_detail(resp.status_code, body, content_type=resp.headers.get("content-type", ""))
                            if is_context_length_exceeded(body) and context_attempt < cfg.context_recovery_retries:
                                pack = pack_turn_items(turn, cfg, aggressive=True)
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
                                if item.get("name") == cfg.function_name or item.get("type") == "function_call":
                                    key = str(item.get("call_id") or item.get("id") or f"open-{len(open_calls)}")
                                    open_calls[key] = now
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
                                if len(wrappers) >= cfg.max_emit_value_calls and not open_calls:
                                    await resp.aclose()
                                    result.aborted = True
                                    break
                            elif typ in {"response.completed", "response.incomplete"}:
                                result.upstream_completed_seen = typ == "response.completed"
                                await resp.aclose()
                                result.aborted = bool(wrappers)
                                break
                            elif typ in {"response.failed", "error"}:
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
                            if wrappers and not open_calls and (now - last_event) >= quiet:
                                await resp.aclose()
                                result.aborted = True
                                break

                        _apply_wrappers(result, wrappers)
                        if result.args_done_seen:
                            result.aborted = True
                            proxy_calls, client_calls = split_proxy_calls(result.tool_calls)
                            if proxy_calls and proxy_rounds < cfg.web_max_rounds_per_request:
                                await apply_proxy_tools(turn, proxy_calls, cfg)
                                proxy_rounds += 1
                                result.retry_reasons.append(f"proxy_web:{','.join(c.name for c in proxy_calls)}")
                                pack = pack_turn_items(turn, cfg)
                                payload = build_upstream_payload(turn, cfg, packed_items=pack.items)
                                result.tool_calls = []
                                result.answer = ""
                                result.commentary = ""
                                result.args_done_seen = False
                                wrappers = []
                                continue
                            result.tool_calls = client_calls
                            if should_continue_progress(result) and progress_rounds < cfg.progress_continuation_rounds:
                                note = (result.commentary or "").strip()
                                if note:
                                    progress_notes.append(note)
                                    turn.items.append(_progress_nudge(note))
                                progress_rounds += 1
                                result.retry_reasons.append("progress_commentary")
                                pack = pack_turn_items(turn, cfg)
                                payload = build_upstream_payload(turn, cfg, packed_items=pack.items)
                                result.packed_item_count = len(pack.items)
                                result.tool_calls = []
                                result.answer = ""
                                result.commentary = ""
                                result.args_done_seen = False
                                wrappers = []
                                continue
                            if progress_notes:
                                extra = (result.commentary or "").strip()
                                result.commentary = "\n\n".join([item for item in [*progress_notes, extra] if item])
                            if should_continue_progress(result):
                                fallback = _progress_fallback_call(turn)
                                if fallback is not None:
                                    result.tool_calls = [fallback]
                                    result.answer = ""
                                    result.retry_reasons.append("progress_fallback_tool")
                            break
                        if result.retry_reasons[-1:] == ["args_done_timeout"] and result.retry_count < cfg.stream_recovery_retries:
                            result.retry_count += 1
                            pack = pack_turn_items(turn, cfg, aggressive=True)
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
        "error": result.error,
    }
