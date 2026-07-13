from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterable

import httpx
from fastapi import HTTPException

from .audit import redact
from .config import Settings, settings
from .models import Bill015Result, BridgeToolCall, NormalizedRequest, local_response_id
from .payloads import build_bill015_payload
from .sse import SSEEvent, parse_async_sse_lines
from .tool_bridge import parse_function_arguments
from .tool_history import is_repeated_successful_call
from .upstream_client import http_timeout, upstream_auth_headers
from .upstream_errors import sanitize_upstream_error_detail
from .usage_estimator import usage_estimate_dict


def apply_tool_loop_guard(result: Bill015Result, n: NormalizedRequest) -> None:
    """Apply native-like, non-terminal loop hygiene.

    Native Codex does not surface a synthetic "[loop guard]" final message just
    because the model repeated a tool call. Tool call/output items stay in
    history and the next turn can continue from them.  The proxy should
    therefore avoid user-visible hard stops: drop only redundant already
    successful calls from a mixed batch, and let any remaining different action
    proceed. If the whole batch is redundant, keep the most recent call so the
    client/tool loop continues instead of ending the task mid-flight.
    """
    if result.bridge_mode != "tool_call" or not result.tool_calls or not n.tool_history:
        return

    repeated_success = [
        c for c in result.tool_calls
        if is_repeated_successful_call(c.name, c.arguments, n.tool_history)
    ]
    if repeated_success:
        repeated_ids = {id(c) for c in repeated_success}
        remaining = [c for c in result.tool_calls if id(c) not in repeated_ids]
        names = ", ".join(_tool_label(c) for c in repeated_success[:5])
        result.retry_reasons.append(f"filtered_repeated_success:{names}")
        if remaining:
            result.tool_calls = remaining
            return
        result.retry_reasons.append(f"allowed_repeated_success_to_avoid_abort:{names}")
        return

    if n.latest_tool_failed and _repeats_latest_failed_tool(result.tool_calls, n.tool_history):
        names = ", ".join(_tool_label(c) for c in result.tool_calls[:5])
        result.retry_reasons.append(f"repeated_failed_tool_allowed:{names}")


def _tool_label(call: Any) -> str:
    ns = getattr(call, "namespace", None)
    name = getattr(call, "name", "")
    return f"{ns}.{name}" if ns else str(name)


def _repeats_latest_failed_tool(calls: list[Any], history: Any) -> bool:
    latest_failed = [out for out in getattr(history, "latest_outputs", []) if getattr(out, "success", None) is False]
    if not latest_failed:
        return False
    failed_call_ids = {str(getattr(out, "call_id", "") or "") for out in latest_failed}
    failed_keys = set()
    for previous_call in getattr(history, "calls", []) or []:
        if str(getattr(previous_call, "call_id", "") or "") in failed_call_ids:
            failed_keys.add((
                str(getattr(previous_call, "name", "") or "").lower(),
                _normalize_tool_arguments(str(getattr(previous_call, "arguments", "") or "")),
            ))
    if not failed_keys:
        return False
    for call in calls:
        key = (
            str(getattr(call, "name", "") or "").lower(),
            _normalize_tool_arguments(str(getattr(call, "arguments", "") or "")),
        )
        if key in failed_keys:
            return True
    return False


def _normalize_tool_arguments(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return " ".join(text.split())


def _event_item_key(obj: dict[str, Any], item: dict[str, Any] | None = None) -> str:
    item = item or {}
    return str(
        obj.get("item_id")
        or obj.get("call_id")
        or item.get("id")
        or item.get("call_id")
        or "__active__"
    )


def _remember_stream_tool(active_tools: dict[str, dict[str, Any]], obj: dict[str, Any]) -> dict[str, Any]:
    item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
    key = _event_item_key(obj, item)
    current = dict(active_tools.get(key) or active_tools.get("__active__") or {})
    tool_type = item.get("type") or current.get("type")
    current.update({
        "item_id": item.get("id") or obj.get("item_id") or current.get("item_id"),
        "call_id": item.get("call_id") or obj.get("call_id") or current.get("call_id"),
        "name": item.get("name") or obj.get("name") or current.get("name") or _name_from_tool_type(str(tool_type or "")),
        "namespace": item.get("namespace") or obj.get("namespace") or current.get("namespace"),
        "type": tool_type,
    })
    active_tools[key] = current
    active_tools["__active__"] = current
    return current


def _active_stream_tool(active_tools: dict[str, dict[str, Any]], obj: dict[str, Any]) -> dict[str, Any]:
    key = _event_item_key(obj)
    current = dict(active_tools.get(key) or active_tools.get("__active__") or {})
    if obj.get("call_id") and not current.get("call_id"):
        current["call_id"] = obj.get("call_id")
    if obj.get("name") and not current.get("name"):
        current["name"] = obj.get("name")
    if obj.get("namespace") and not current.get("namespace"):
        current["namespace"] = obj.get("namespace")
    if current:
        active_tools[key] = current
        active_tools["__active__"] = current
    return current


def _json_object_from_text(text: str) -> tuple[dict[str, Any], bool]:
    try:
        obj = json.loads(text or "{}")
        return (obj if isinstance(obj, dict) else {}, False)
    except Exception:
        return {}, True


def _looks_like_emit_value_arguments(text: str, cfg: Settings) -> bool:
    obj, malformed = _json_object_from_text(text)
    if malformed:
        return False
    return "mode" in obj and cfg.answer_field in obj and "tool_calls" in obj


def _stringify_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False)


SAFE_EMPTY_UPSTREAM_ANSWER = (
    "[local proxy] Upstream ended without tool-call arguments or assistant text; "
    "the proxy safely completed this stream instead of disconnecting. Please retry the last request."
)


def _extract_text_parts(value: Any) -> list[str]:
    parts: list[str] = []
    if isinstance(value, str):
        if value:
            parts.append(value)
        return parts
    if isinstance(value, list):
        for item in value:
            parts.extend(_extract_text_parts(item))
        return parts
    if isinstance(value, dict):
        # Responses message content normally uses {"type":"output_text","text":"..."}.
        for key in ("text", "output_text", "content", "summary"):
            v = value.get(key)
            if isinstance(v, str) and v:
                parts.append(v)
            elif isinstance(v, (list, dict)):
                parts.extend(_extract_text_parts(v))
                break
        return parts
    return parts


def _extract_message_text_from_item(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    item_type = str(item.get("type") or "")
    if item_type and item_type not in {"message", "output_text", "text"}:
        return ""
    return "\n".join(part for part in _extract_text_parts(item.get("content", item.get("text", ""))) if part)


def _extract_response_text(obj: dict[str, Any]) -> str:
    response = obj.get("response") if isinstance(obj.get("response"), dict) else obj
    if not isinstance(response, dict):
        return ""
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if isinstance(item, dict):
            text = _extract_message_text_from_item(item)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _finalize_result_without_args_done(result: Bill015Result, answer_buffer: list[str]) -> None:
    """Avoid surfacing a hard stream-disconnect error for empty upstream closes."""
    if result.args_done_seen or result.error:
        return
    if not result.answer and answer_buffer:
        result.answer = "".join(answer_buffer)
    result.bridge_mode = "answer"
    if result.answer:
        result.retry_reasons.append("upstream_completed_without_args_done_used_text")
        return
    result.answer = SAFE_EMPTY_UPSTREAM_ANSWER
    result.upstream_completed_seen = True
    result.retry_reasons.append("upstream_completed_without_args_done_synthesized_safe_answer")


def _name_from_tool_type(tool_type: str) -> str:
    if tool_type == "tool_search_call":
        return "tool_search"
    if tool_type == "web_search_call":
        return "web_search"
    if tool_type == "computer_call":
        return "computer"
    return ""


def _direct_call_type(tool: dict[str, Any], name: str) -> str:
    typ = str(tool.get("type") or "")
    if typ == "custom_tool_call":
        return "custom"
    if typ == "tool_search_call" or name == "tool_search":
        return "tool_search"
    if typ == "web_search_call" or name == "web_search":
        return "web_search"
    return "function"


def _apply_tool_arguments_result(
    result: Bill015Result,
    final_args: str,
    tool: dict[str, Any],
    n: NormalizedRequest,
    cfg: Settings,
) -> None:
    name = str(tool.get("name") or _name_from_tool_type(str(tool.get("type") or "")) or "")
    namespace = tool.get("namespace")
    call_id = str(tool.get("call_id") or tool.get("item_id") or "") or ("call_" + local_response_id().removeprefix("resp_local_")[:18])
    result.raw_arguments = final_args
    result.args_done_seen = True
    if name == cfg.function_name or (not name and _looks_like_emit_value_arguments(final_args, cfg)):
        result.answer, result.malformed_function_args, result.repaired_args, result.bridge_mode, result.tool_calls = parse_function_arguments(final_args, cfg, n.tool_registry)
        return
    if name == cfg.final_answer_tool_name:
        obj, malformed = _json_object_from_text(final_args)
        result.malformed_function_args = malformed
        result.answer = str(obj.get(cfg.answer_field) or obj.get("answer") or final_args or "")
        result.bridge_mode = "answer"
        result.tool_calls = []
        return
    result.bridge_mode = "tool_call"
    result.tool_calls = [
        BridgeToolCall(
            id=call_id,
            name=name,
            namespace=str(namespace) if namespace else None,
            arguments=final_args if final_args else "{}",
            call_type=_direct_call_type(tool, name),
            requested_name=name,
        )
    ]


def _args_done_deadline(cfg: Settings = settings) -> float | None:
    return time.perf_counter() + cfg.args_done_timeout_ms / 1000 if cfg.args_done_timeout_ms > 0 else None


def _raise_if_args_done_timed_out(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise HTTPException(status_code=504, detail="timed out waiting for response.function_call_arguments.done")


def _retry_delay_seconds(attempt: int, cfg: Settings = settings) -> float:
    base_ms = max(0, cfg.upstream_retry_backoff_ms)
    if base_ms <= 0:
        return 0
    # attempt is zero-based for the failed try. Cap so a flaky upstream does not
    # freeze the local Codex stream for too long.
    return min(5.0, (base_ms / 1000.0) * (2 ** attempt))


def _upstream_detail_text(detail: Any) -> str:
    try:
        return json.dumps(detail, ensure_ascii=False)
    except Exception:
        return str(detail)


def _is_retryable_pre_stream_failure(status_code: int | None, detail: Any = None) -> bool:
    if status_code is not None and 500 <= status_code <= 599:
        return True
    text = _upstream_detail_text(detail).lower()
    return "do_request_failed" in text or "upstream error" in text


def collect_bill015_result_from_events(events: Iterable[SSEEvent], n: NormalizedRequest, cfg: Settings = settings) -> Bill015Result:
    """Offline helper used by regression tests to validate event-state behavior."""
    result = Bill015Result(local_request_id=local_response_id())
    active_tools: dict[str, dict[str, Any]] = {}
    args_buffers: dict[str, list[str]] = {}
    custom_buffers: dict[str, list[str]] = {}
    answer_buffer: list[str] = []
    for ev in events:
        obj = ev.json
        typ = (obj or {}).get("type") or ev.event
        if typ:
            result.event_sequence.append(str(typ))
        if not obj:
            if ev.data == "[DONE]":
                result.upstream_completed_seen = True
            continue
        if typ == "response.created":
            response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
            result.upstream_response_id = response.get("id") or obj.get("response_id") or obj.get("id")
        elif typ == "response.output_item.added":
            item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
            _remember_stream_tool(active_tools, obj)
            result.function_call_seen = item.get("type") in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "computer_call", "tool_call"} or item.get("name") == cfg.function_name or cfg.function_name in json.dumps(obj, ensure_ascii=False)
        elif typ == "response.function_call_arguments.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                key = _event_item_key(obj)
                args_buffers.setdefault(key, []).append(delta)
                if key != "__active__":
                    args_buffers.setdefault("__active__", []).append(delta)
        elif typ == "response.custom_tool_call_input.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                key = _event_item_key(obj)
                custom_buffers.setdefault(key, []).append(delta)
                if key != "__active__":
                    custom_buffers.setdefault("__active__", []).append(delta)
        elif typ == "response.output_text.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                answer_buffer.append(delta)
        elif typ == "response.output_text.done":
            text = obj.get("text")
            if isinstance(text, str):
                result.answer = text
        elif typ == "response.function_call_arguments.done":
            tool = _active_stream_tool(active_tools, obj)
            key = _event_item_key(obj)
            final_args = obj.get("arguments") if isinstance(obj.get("arguments"), str) else "".join(args_buffers.get(key) or args_buffers.get("__active__") or [])
            _apply_tool_arguments_result(result, final_args, tool, n, cfg)
            apply_tool_loop_guard(result, n)
            result.aborted = True
            break
        elif typ == "response.custom_tool_call_input.done":
            tool = _active_stream_tool(active_tools, obj)
            key = _event_item_key(obj)
            final_input = obj.get("input") if isinstance(obj.get("input"), str) else "".join(custom_buffers.get(key) or custom_buffers.get("__active__") or [])
            tool["type"] = tool.get("type") or "custom_tool_call"
            _apply_tool_arguments_result(result, final_input, tool, n, cfg)
            apply_tool_loop_guard(result, n)
            result.aborted = True
            break
        elif typ == "response.output_item.done":
            item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
            item_type = str(item.get("type") or "")
            if item_type in {"tool_search_call", "web_search_call", "computer_call"}:
                tool = _remember_stream_tool(active_tools, obj)
                final_args = _stringify_arguments(item.get("arguments") or item.get("action") or {})
                _apply_tool_arguments_result(result, final_args, tool, n, cfg)
                apply_tool_loop_guard(result, n)
                result.aborted = True
                break
            if item_type == "message":
                text = _extract_message_text_from_item(item)
                if text:
                    result.answer = text
        elif typ == "response.completed":
            result.upstream_completed_seen = True
            if not result.answer and answer_buffer:
                result.answer = "".join(answer_buffer)
            if not result.answer:
                result.answer = _extract_response_text(obj)
            break
        elif typ == "response.incomplete":
            result.error = "upstream response incomplete"
            if not result.answer and answer_buffer:
                result.answer = "".join(answer_buffer)
            if not result.answer:
                result.answer = _extract_response_text(obj)
            break
        elif typ in {"response.failed", "error"}:
            result.error = json.dumps(obj, ensure_ascii=False)[:2000]
            break
    _finalize_result_without_args_done(result, answer_buffer)
    return result

def audit_from_result(result: Bill015Result, n: NormalizedRequest, mode: str, fallback_used: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "local_request_id": result.local_request_id,
        "upstream_response_id": result.upstream_response_id,
        "mode": mode,
        "client_api": n.client_api,
        "model": n.model,
        "original_model": n.original_model,
        "strict_zero": settings.strict_zero,
        "function_call_seen": result.function_call_seen,
        "args_done_seen": result.args_done_seen,
        "upstream_completed_seen": result.upstream_completed_seen,
        "aborted_at": "tool_arguments_boundary" if result.aborted else None,
        "answer_chars": len(result.answer),
        "bridge_mode": result.bridge_mode,
        "tool_calls": [{"id": c.id, "name": c.name, "namespace": c.namespace, "requested_name": c.requested_name, "type": c.call_type, "arguments_chars": len(c.arguments)} for c in result.tool_calls],
        "duration_ms": result.duration_ms,
        "retry_count": result.retry_count,
        "retry_reasons": result.retry_reasons,
        "fallback_used": fallback_used,
        "error": result.error,
        "event_sequence": result.event_sequence[-50:],
        "malformed_function_args": result.malformed_function_args,
        "repaired_args": result.repaired_args,
        "latest_tool_outputs": len(n.tool_history.latest_outputs) if n.tool_history else 0,
        "latest_tool_failed": n.latest_tool_failed,
        "pending_tool_calls": n.pending_tool_call_count,
        "tool_loop_decision": result.bridge_mode,
        "repeated_tool_call_detected": any(
            is_repeated_successful_call(c.name, c.arguments, n.tool_history)
            for c in result.tool_calls
        ),
        "event_fidelity": {
            "output_item_count": len(result.tool_calls) if result.bridge_mode == "tool_call" else (1 if result.answer else 0),
            "event_types": result.event_sequence[-50:],
            "completed_status": "completed" if not result.error else "failed",
            "has_reasoning_summary": False,
            "has_annotations": False,
            "sequence_count": len(result.event_sequence),
        },
    }
    if settings.usage_audit_breakdown:
        rec["usage_estimate"] = usage_estimate_dict(n)
    if settings.store_prompts:
        rec["prompt"] = n.user_input
        rec["instructions"] = n.instructions
    if settings.store_answers:
        rec["answer"] = result.answer
    if result.verify_delta is not None:
        rec["verify"] = {"pre": result.verify_pre, "post": result.verify_post, "delta": result.verify_delta}
    return rec

async def fetch_user_self(cfg: Settings = settings) -> dict[str, Any] | None:
    if not cfg.packy_cookie:
        return None
    try:
        async with httpx.AsyncClient(timeout=http_timeout(cfg)) as client:
            r = await client.get(
                cfg.upstream_base_url + "/api/user/self",
                headers={"Cookie": cfg.packy_cookie, "new-api-user": cfg.packy_user_id, "User-Agent": "bill015-local-proxy/1.0"},
            )
        obj = r.json()
        if isinstance(obj, dict) and obj.get("success") and isinstance(obj.get("data"), dict):
            d = obj["data"]
            return {"id": d.get("id"), "quota": d.get("quota"), "used_quota": d.get("used_quota"), "request_count": d.get("request_count")}
        return {"status": r.status_code, "body": obj}
    except Exception as e:
        return {"error": type(e).__name__}

def compute_delta(pre: dict[str, Any] | None, post: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pre or not post:
        return None
    out: dict[str, Any] = {}
    for k in ("quota", "used_quota", "request_count"):
        if isinstance(pre.get(k), (int, float)) and isinstance(post.get(k), (int, float)):
            out[k] = post[k] - pre[k]
    return out or None

async def execute_bill015(n: NormalizedRequest, mode: str, cfg: Settings = settings, local_request_id: str | None = None) -> Bill015Result:
    request_id = local_request_id or local_response_id()
    result = Bill015Result(local_request_id=request_id)
    start = time.perf_counter()
    if mode == "verify":
        result.verify_pre = await fetch_user_self(cfg)
    if not cfg.upstream_api_key:
        raise HTTPException(status_code=500, detail=f"Missing upstream API key env {cfg.upstream_api_key_env}")

    payload = build_bill015_payload(n, cfg)
    active_tools: dict[str, dict[str, Any]] = {}
    args_buffers: dict[str, list[str]] = {}
    custom_buffers: dict[str, list[str]] = {}
    answer_buffer: list[str] = []
    args_done_deadline = _args_done_deadline(cfg)
    headers = upstream_auth_headers(stream=True, cfg=cfg)
    try:
        max_retries = 0 if cfg.strict_zero else max(0, int(getattr(cfg, "upstream_retries", 0)))
        async with httpx.AsyncClient(timeout=http_timeout(cfg)) as client:
            for attempt in range(max_retries + 1):
                active_tools = {}
                args_buffers = {}
                custom_buffers = {}
                answer_buffer = []
                args_done_deadline = _args_done_deadline(cfg)
                attempt_event_len = len(result.event_sequence)
                try:
                    async with client.stream("POST", cfg.upstream_base_url + "/v1/responses", headers=headers, json=payload) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            detail = sanitize_upstream_error_detail(
                                resp.status_code,
                                body,
                                content_type=resp.headers.get("content-type", ""),
                            )
                            if attempt < max_retries and _is_retryable_pre_stream_failure(resp.status_code, detail):
                                result.retry_count += 1
                                reason = f"http_{resp.status_code}"
                                result.retry_reasons.append(reason)
                                result.event_sequence.append(f"retry:{reason}")
                                await asyncio.sleep(_retry_delay_seconds(attempt, cfg))
                                continue
                            raise HTTPException(status_code=502, detail=detail)
                        async for ev in parse_async_sse_lines(resp.aiter_lines()):
                            _raise_if_args_done_timed_out(args_done_deadline)
                            obj = ev.json
                            typ = (obj or {}).get("type") or ev.event
                            if typ:
                                result.event_sequence.append(str(typ))
                            if not obj:
                                if ev.data == "[DONE]":
                                    result.upstream_completed_seen = True
                                continue
                            if typ == "response.created":
                                response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
                                result.upstream_response_id = response.get("id") or obj.get("response_id") or obj.get("id")
                            elif typ == "response.output_item.added":
                                item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
                                _remember_stream_tool(active_tools, obj)
                                if item.get("type") in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "computer_call", "tool_call"} or item.get("name") == cfg.function_name:
                                    result.function_call_seen = True
                                else:
                                    result.function_call_seen = result.function_call_seen or (cfg.function_name in json.dumps(obj, ensure_ascii=False))
                            elif typ == "response.function_call_arguments.delta":
                                delta = obj.get("delta")
                                if isinstance(delta, str):
                                    key = _event_item_key(obj)
                                    args_buffers.setdefault(key, []).append(delta)
                                    if key != "__active__":
                                        args_buffers.setdefault("__active__", []).append(delta)
                            elif typ == "response.custom_tool_call_input.delta":
                                delta = obj.get("delta")
                                if isinstance(delta, str):
                                    key = _event_item_key(obj)
                                    custom_buffers.setdefault(key, []).append(delta)
                                    if key != "__active__":
                                        custom_buffers.setdefault("__active__", []).append(delta)
                            elif typ == "response.output_text.delta":
                                delta = obj.get("delta")
                                if isinstance(delta, str):
                                    answer_buffer.append(delta)
                            elif typ == "response.output_text.done":
                                text = obj.get("text")
                                if isinstance(text, str):
                                    result.answer = text
                            elif typ == "response.function_call_arguments.done":
                                tool = _active_stream_tool(active_tools, obj)
                                key = _event_item_key(obj)
                                final_args = obj.get("arguments") if isinstance(obj.get("arguments"), str) else "".join(args_buffers.get(key) or args_buffers.get("__active__") or [])
                                _apply_tool_arguments_result(result, final_args, tool, n, cfg)
                                apply_tool_loop_guard(result, n)
                                await resp.aclose()
                                result.aborted = True
                                break
                            elif typ == "response.custom_tool_call_input.done":
                                tool = _active_stream_tool(active_tools, obj)
                                key = _event_item_key(obj)
                                final_input = obj.get("input") if isinstance(obj.get("input"), str) else "".join(custom_buffers.get(key) or custom_buffers.get("__active__") or [])
                                tool["type"] = tool.get("type") or "custom_tool_call"
                                _apply_tool_arguments_result(result, final_input, tool, n, cfg)
                                apply_tool_loop_guard(result, n)
                                await resp.aclose()
                                result.aborted = True
                                break
                            elif typ == "response.output_item.done":
                                item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
                                item_type = str(item.get("type") or "")
                                if item_type in {"tool_search_call", "web_search_call", "computer_call"}:
                                    tool = _remember_stream_tool(active_tools, obj)
                                    final_args = _stringify_arguments(item.get("arguments") or item.get("action") or {})
                                    _apply_tool_arguments_result(result, final_args, tool, n, cfg)
                                    apply_tool_loop_guard(result, n)
                                    await resp.aclose()
                                    result.aborted = True
                                    break
                                if item_type == "message":
                                    text = _extract_message_text_from_item(item)
                                    if text:
                                        result.answer = text
                            elif typ == "response.completed":
                                result.upstream_completed_seen = True
                                if not result.answer and answer_buffer:
                                    result.answer = "".join(answer_buffer)
                                if not result.answer:
                                    result.answer = _extract_response_text(obj)
                                break
                            elif typ == "response.incomplete":
                                result.error = "upstream response incomplete"
                                if not result.answer and answer_buffer:
                                    result.answer = "".join(answer_buffer)
                                if not result.answer:
                                    result.answer = _extract_response_text(obj)
                                break
                            elif typ in {"response.failed", "error"}:
                                raise RuntimeError(json.dumps(obj, ensure_ascii=False)[:2000])
                    break
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.TimeoutException) as e:
                    # Retry only if this attempt failed before yielding any
                    # upstream SSE event. Once the model has started streaming,
                    # retrying could duplicate work or affect billing.
                    if attempt < max_retries and len(result.event_sequence) == attempt_event_len:
                        result.retry_count += 1
                        reason = type(e).__name__
                        result.retry_reasons.append(reason)
                        result.event_sequence.append(f"retry:{reason}")
                        await asyncio.sleep(_retry_delay_seconds(attempt, cfg))
                        continue
                    raise
        if mode == "verify":
            await asyncio.sleep(0.5)
            result.verify_post = await fetch_user_self(cfg)
            result.verify_delta = compute_delta(result.verify_pre, result.verify_post)
        _finalize_result_without_args_done(result, answer_buffer)
        return result
    except HTTPException:
        raise
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=502, detail=result.error) from e
    finally:
        result.duration_ms = int((time.perf_counter() - start) * 1000)

def dry_run_response(n: NormalizedRequest, mode: str = "dry-run") -> dict[str, Any]:
    payload = build_bill015_payload(n)
    safe_payload = redact(payload)
    # Avoid returning full prompt unless explicitly enabled.
    if not settings.store_prompts:
        for item in safe_payload.get("input", []):
            if isinstance(item, dict) and item.get("role") == "user":
                item["content"] = f"<redacted prompt chars={len(n.user_input)}>"
    return {
        "id": local_response_id(),
        "object": "response",
        "status": "completed",
        "mode": mode,
        "would_post": settings.upstream_base_url + "/v1/responses",
        "upstream_configured": settings.upstream_configured,
        "usage_estimate": usage_estimate_dict(n),
        "payload": safe_payload,
    }
