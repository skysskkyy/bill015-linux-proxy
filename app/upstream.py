from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterable

import httpx
from fastapi import HTTPException

from .audit import redact
from .config import Settings, settings
from .context_compaction import compact_payload_history, context_threshold_tokens, payload_input_tokens, reinforce_final_action
from .key_pool import ApiKeySelection, upstream_key_pool
from .local_web_research import local_web_research_preflight
from .models import Bill015Result, BridgeToolCall, NormalizedRequest, local_response_id
from .payloads import build_bill015_payload, use_responses_lite_upstream
from .sse import SSEEvent, parse_async_sse_lines
from .tool_bridge import parse_function_arguments
from .tool_history import is_repeated_successful_call
from .upstream_client import http_timeout, upstream_auth_headers
from .upstream_errors import is_context_length_exceeded, key_rotation_error_reason, sanitize_upstream_error_detail
from .usage_estimator import usage_estimate_dict

KEY_ROTATION_DELAY_SECONDS = 10 * 60


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


def _looks_like_emit_value_arguments_loose(text: str, cfg: Settings) -> bool:
    if _looks_like_emit_value_arguments(text, cfg):
        return True
    # If the stream reached function_call_arguments.done but the JSON is
    # truncated, the strict parser above cannot inspect keys.  Codex still has
    # enough signal to classify this as the bridge finalizer rather than a
    # direct native tool call when the top-level emit_value keys are present in
    # the partial argument text.
    value = str(text or "")
    return '"mode"' in value and '"tool_calls"' in value


def _stringify_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False)


EMPTY_UPSTREAM_ERROR = "upstream ended without a final answer or tool call"


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
    result.error = EMPTY_UPSTREAM_ERROR
    result.upstream_completed_seen = True
    result.retry_reasons.append("upstream_completed_without_args_done")


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
    if name == cfg.function_name or (not name and _looks_like_emit_value_arguments_loose(final_args, cfg)):
        # Native tool_search_call is executed by Codex Core on both Desktop and
        # CLI. The TUI only rejects app-server DynamicToolCall requests, which
        # this Responses bridge never emits.
        result.answer, result.malformed_function_args, result.repaired_args, result.bridge_mode, result.tool_calls = parse_function_arguments(final_args, cfg, n.tool_registry, allow_dynamic_tools=True)
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
        "responses_lite_client": n.responses_lite,
        "responses_lite_upstream": use_responses_lite_upstream(n, settings),
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
        "upstream_key_index": result.upstream_key_index,
        "upstream_key_count": result.upstream_key_count,
        "key_switch_count": result.key_switch_count,
        "compaction_count": result.compaction_count,
        "compacted_item_count": result.compacted_item_count,
        "clipped_tool_output_count": result.clipped_tool_output_count,
        "empty_stream_retry_count": result.empty_stream_retry_count,
        "stream_timeout_retry_count": result.stream_timeout_retry_count,
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

def _record_key_selection(result: Bill015Result, selection: ApiKeySelection) -> None:
    result.upstream_key_index = selection.index + 1
    result.upstream_key_count = selection.count


def _rotate_after_key_error(
    result: Bill015Result,
    selection: ApiKeySelection,
    tried_keys: set[str],
    reason: str,
    cfg: Settings,
) -> ApiKeySelection | None:
    next_selection = upstream_key_pool(cfg).rotate_after_failure(selection.key, tried_keys)
    if next_selection is None:
        return None
    tried_keys.add(next_selection.key)
    result.retry_count += 1
    result.key_switch_count += 1
    audit_reason = f"{reason}:key_switch:{selection.index + 1}->{next_selection.index + 1}"
    result.retry_reasons.append(audit_reason)
    result.event_sequence.append(f"retry:{audit_reason}")
    _record_key_selection(result, next_selection)
    return next_selection


async def _rotate_after_key_error_with_delay(
    result: Bill015Result,
    selection: ApiKeySelection,
    tried_keys: set[str],
    reason: str,
    cfg: Settings,
) -> ApiKeySelection | None:
    await asyncio.sleep(KEY_ROTATION_DELAY_SECONDS)
    return _rotate_after_key_error(result, selection, tried_keys, reason, cfg)

def _reset_result_for_key_retry(result: Bill015Result) -> None:
    result.upstream_response_id = None
    result.answer = ""
    result.bridge_mode = "answer"
    result.tool_calls = []
    result.raw_arguments = ""
    result.function_call_seen = False
    result.args_done_seen = False
    result.upstream_completed_seen = False
    result.aborted = False
    result.malformed_function_args = False
    result.repaired_args = False
    result.error = None


def _record_compaction(result: Bill015Result, reason: str, before_tokens: int, after_tokens: int, removed: int, clipped: int, *, count_retry: bool = True) -> None:
    result.compaction_count += 1
    result.compacted_item_count += removed
    result.clipped_tool_output_count += clipped
    if count_retry:
        result.retry_count += 1
    audit_reason = f"{reason}:{before_tokens}->{after_tokens}:removed={removed}:clipped={clipped}"
    result.retry_reasons.append(audit_reason)
    result.event_sequence.append(f"retry:{audit_reason}")


def _needs_exact_preemptive_token_count(n: NormalizedRequest, precompact_limit: int) -> bool:
    """Return whether hot-path precompaction needs a full payload token pass.

    Native-context preservation means every request builds a Responses payload,
    but most turns are far below the configured compaction threshold.  Exact
    payload tokenization is only needed near that threshold; skipping it for
    clearly small/medium turns keeps behavior identical while avoiding local
    CPU work before the upstream stream starts.
    """
    estimate = int(getattr(n, "estimated_input_tokens", 0) or 0)
    if estimate <= 0:
        return True
    # Keep a large safety margin for bridge instructions/tool catalog overhead.
    return estimate >= int(max(8_000, precompact_limit * 0.65))


def _compact_for_context_recovery(payload: dict[str, Any], result: Bill015Result, cfg: Settings, reason: str, attempt: int, *, aggressive: bool) -> tuple[dict[str, Any], bool]:
    before = payload_input_tokens(payload)
    target_percent = max(35, cfg.compact_target_percent - attempt * 10)
    target = min(context_threshold_tokens(cfg.context_window_tokens, target_percent), max(8_000, int(before * 0.80)))
    compacted, removed, clipped = compact_payload_history(payload, target, aggressive=aggressive, latest_tool_output_max_chars=cfg.latest_tool_output_max_chars)
    after = payload_input_tokens(compacted)
    if after >= before:
        return payload, False
    _record_compaction(result, reason, before, after, removed, clipped)
    return compacted, True


def _empty_upstream_detail(attempts: int) -> dict[str, Any]:
    return {
        "message": f"upstream ended without a final answer or tool call after {attempts} attempts",
        "code": "upstream_empty_completion",
    }


def _empty_stream_exhausted_answer(attempts: int) -> str:
    return (
        f"上游连续 {attempts} 次没有返回最终答案或工具调用，代理已停止继续重试并避免断流。\n"
        "这通常是上游 SSE 空结束、长上下文/大工具输出导致模型未到达 final-action 边界，"
        "或上游临时静默。请直接重试当前步骤；如果反复出现，建议开启新会话/压缩历史或减少最近工具输出。"
    )


def _stream_recovery_detail(reason: str, attempts: int) -> dict[str, Any]:
    return {
        "message": f"upstream stream did not reach a final answer or tool call after {attempts} recovery attempts: {reason}",
        "code": "upstream_stream_recovery_exhausted",
        "reason": reason,
    }


async def _retry_stream_recovery(
    payload: dict[str, Any],
    result: Bill015Result,
    cfg: Settings,
    reason: str,
    attempt: int,
    precompact_limit: int,
) -> tuple[dict[str, Any], bool]:
    """Retry recoverable mid-stream stalls without surfacing Codex disconnects.

    The native Codex client is strict about receiving either final text or
    `response.function_call_arguments.done`.  A slow/stalled upstream can close
    or time out after partial reasoning/events, which used to leak as
    `stream disconnected before completion`.  Treat that as a recoverable
    upstream attempt: reinforce the final-action instruction, compact if the
    payload is near the context ceiling, and restart the upstream stream.
    """
    if attempt >= max(0, cfg.stream_recovery_retries):
        return payload, False
    result.stream_timeout_retry_count += 1
    result.retry_count += 1
    audit_reason = f"{reason}_retry:{attempt + 1}"
    result.retry_reasons.append(audit_reason)
    result.event_sequence.append(f"retry:{audit_reason}")
    recovered = reinforce_final_action(payload)
    if payload_input_tokens(recovered) >= precompact_limit:
        recovered, _ = _compact_for_context_recovery(
            recovered,
            result,
            cfg,
            f"{reason}_compaction",
            attempt,
            aggressive=True,
        )
    _reset_result_for_key_retry(result)
    await asyncio.sleep(_retry_delay_seconds(attempt, cfg))
    return recovered, True


async def execute_bill015(n: NormalizedRequest, mode: str, cfg: Settings = settings, local_request_id: str | None = None) -> Bill015Result:
    request_id = local_request_id or local_response_id()
    result = Bill015Result(local_request_id=request_id)
    start = time.perf_counter()
    preflight_call = local_web_research_preflight(n, cfg)
    if preflight_call:
        result.bridge_mode = "tool_call"
        result.tool_calls = [preflight_call]
        result.retry_reasons.append("local_web_research_preflight")
        result.duration_ms = int((time.perf_counter() - start) * 1000)
        return result

    if mode == "verify":
        result.verify_pre = await fetch_user_self(cfg)

    selection = upstream_key_pool(cfg).current()
    if selection is None:
        raise HTTPException(status_code=500, detail=f"Missing upstream API key env {cfg.upstream_api_key_env}")
    tried_keys = {selection.key}
    _record_key_selection(result, selection)

    payload = build_bill015_payload(n, cfg)
    precompact_limit = context_threshold_tokens(cfg.context_window_tokens, cfg.auto_compact_percent)
    initial_tokens = (
        payload_input_tokens(payload)
        if _needs_exact_preemptive_token_count(n, precompact_limit)
        else int(getattr(n, "estimated_input_tokens", 0) or 0)
    )
    if initial_tokens >= precompact_limit:
        target = context_threshold_tokens(cfg.context_window_tokens, cfg.compact_target_percent)
        compacted, removed, clipped = compact_payload_history(
            payload,
            target,
            aggressive=False,
            latest_tool_output_max_chars=cfg.latest_tool_output_max_chars,
        )
        after_tokens = payload_input_tokens(compacted)
        if after_tokens < initial_tokens or removed or clipped:
            payload = compacted
            _record_compaction(result, "preemptive_compaction", initial_tokens, after_tokens, removed, clipped, count_retry=False)
    answer_buffer: list[str] = []
    try:
        max_retries = 0 if cfg.strict_zero else max(0, int(getattr(cfg, "upstream_retries", 0)))
        transient_attempt = 0
        context_retry_attempt = 0
        empty_stream_attempt = 0
        stream_recovery_attempt = 0
        async with httpx.AsyncClient(timeout=http_timeout(cfg)) as client:
            while True:
                active_tools: dict[str, dict[str, Any]] = {}
                args_buffers: dict[str, list[str]] = {}
                custom_buffers: dict[str, list[str]] = {}
                answer_buffer = []
                args_done_deadline = _args_done_deadline(cfg)
                attempt_event_len = len(result.event_sequence)
                rotated_selection: ApiKeySelection | None = None
                context_retry_requested = False
                stream_recovery_reason: str | None = None
                headers = upstream_auth_headers(
                    stream=True,
                    cfg=cfg,
                    api_key=selection.key,
                    responses_lite=use_responses_lite_upstream(n, cfg),
                    client_headers=n.request_headers,
                )

                try:
                    async with client.stream("POST", cfg.upstream_base_url + "/v1/responses", headers=headers, json=payload) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            detail = sanitize_upstream_error_detail(
                                resp.status_code, body, content_type=resp.headers.get("content-type", "")
                            )
                            if is_context_length_exceeded(body) and context_retry_attempt < max(0, cfg.context_recovery_retries):
                                recovered, changed = _compact_for_context_recovery(
                                    payload, result, cfg, "context_length_exceeded", context_retry_attempt, aggressive=True
                                )
                                if changed:
                                    payload = recovered
                                    context_retry_attempt += 1
                                    transient_attempt = 0
                                    _reset_result_for_key_retry(result)
                                    continue
                            rotation_reason = key_rotation_error_reason(body)
                            if rotation_reason:
                                rotated_selection = await _rotate_after_key_error_with_delay(result, selection, tried_keys, rotation_reason, cfg)
                                if rotated_selection is not None:
                                    selection = rotated_selection
                                    transient_attempt = 0
                                    _reset_result_for_key_retry(result)
                                    continue
                            if transient_attempt < max_retries and _is_retryable_pre_stream_failure(resp.status_code, detail):
                                result.retry_count += 1
                                reason = f"http_{resp.status_code}"
                                result.retry_reasons.append(reason)
                                result.event_sequence.append(f"retry:{reason}")
                                await asyncio.sleep(_retry_delay_seconds(transient_attempt, cfg))
                                transient_attempt += 1
                                continue
                            raise HTTPException(status_code=502, detail=detail)

                        async for ev in parse_async_sse_lines(resp.aiter_lines()):
                            if args_done_deadline is not None and time.perf_counter() > args_done_deadline:
                                stream_recovery_reason = "args_done_timeout"
                                await resp.aclose()
                                break
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
                                if is_context_length_exceeded(obj) and context_retry_attempt < max(0, cfg.context_recovery_retries):
                                    recovered, changed = _compact_for_context_recovery(
                                        payload, result, cfg, "context_length_exceeded", context_retry_attempt, aggressive=True
                                    )
                                    if changed:
                                        payload = recovered
                                        context_retry_attempt += 1
                                        context_retry_requested = True
                                        await resp.aclose()
                                        break
                                rotation_reason = key_rotation_error_reason(obj)
                                if rotation_reason:
                                    rotated_selection = await _rotate_after_key_error_with_delay(result, selection, tried_keys, rotation_reason, cfg)
                                    if rotated_selection is not None:
                                        await resp.aclose()
                                        break
                                raise RuntimeError(json.dumps(obj, ensure_ascii=False)[:2000])

                    if context_retry_requested:
                        transient_attempt = 0
                        _reset_result_for_key_retry(result)
                        continue
                    if stream_recovery_reason is not None:
                        payload, retrying = await _retry_stream_recovery(
                            payload,
                            result,
                            cfg,
                            stream_recovery_reason,
                            stream_recovery_attempt,
                            precompact_limit,
                        )
                        if retrying:
                            stream_recovery_attempt += 1
                            transient_attempt = 0
                            continue
                        raise HTTPException(
                            status_code=504,
                            detail=_stream_recovery_detail(stream_recovery_reason, stream_recovery_attempt),
                        )
                    if rotated_selection is not None:
                        selection = rotated_selection
                        transient_attempt = 0
                        _reset_result_for_key_retry(result)
                        continue
                    if not result.args_done_seen and not result.answer and not result.error:
                        if empty_stream_attempt < max(0, cfg.empty_stream_retries):
                            empty_stream_attempt += 1
                            result.empty_stream_retry_count += 1
                            result.retry_count += 1
                            reason = f"empty_stream_retry:{empty_stream_attempt}"
                            result.retry_reasons.append(reason)
                            result.event_sequence.append(f"retry:{reason}")
                            payload = reinforce_final_action(payload)
                            if payload_input_tokens(payload) >= precompact_limit:
                                payload, _ = _compact_for_context_recovery(
                                    payload, result, cfg, "empty_stream_compaction", empty_stream_attempt - 1, aggressive=True
                                )
                            _reset_result_for_key_retry(result)
                            continue
                        attempts = empty_stream_attempt + 1
                        result.answer = _empty_stream_exhausted_answer(attempts)
                        result.bridge_mode = "answer"
                        result.args_done_seen = True
                        result.upstream_completed_seen = True
                        result.retry_reasons.append(f"empty_stream_exhausted:{attempts}")
                        result.event_sequence.append(f"empty_stream_exhausted:{attempts}")
                        break
                    if result.error:
                        raise HTTPException(status_code=502, detail={"message": result.error, "code": "upstream_incomplete"})
                    break
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.TimeoutException) as e:
                    reason = type(e).__name__
                    if isinstance(e, httpx.ReadTimeout):
                        payload, retrying = await _retry_stream_recovery(
                            payload,
                            result,
                            cfg,
                            reason,
                            stream_recovery_attempt,
                            precompact_limit,
                        )
                        if retrying:
                            stream_recovery_attempt += 1
                            transient_attempt = 0
                            continue
                    if transient_attempt < max_retries and len(result.event_sequence) == attempt_event_len:
                        result.retry_count += 1
                        result.retry_reasons.append(reason)
                        result.event_sequence.append(f"retry:{reason}")
                        await asyncio.sleep(_retry_delay_seconds(transient_attempt, cfg))
                        transient_attempt += 1
                        continue
                    raise

        if mode == "verify":
            await asyncio.sleep(0.5)
            result.verify_post = await fetch_user_self(cfg)
            result.verify_delta = compute_delta(result.verify_pre, result.verify_post)
        _finalize_result_without_args_done(result, answer_buffer)
        if result.error:
            raise HTTPException(status_code=502, detail={"message": result.error, "code": "upstream_empty_completion"})
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
