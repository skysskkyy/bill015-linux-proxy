from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import time
from typing import Any, AsyncIterator, Iterable

import httpx
from fastapi import HTTPException

from .audit import redact
from .config import Settings, settings
from .models import Bill015Result, NormalizedRequest, local_response_id
from .payloads import build_bill015_payload
from .sse import SSEEvent, encode_sse, parse_async_sse_lines
from .tool_bridge import parse_function_arguments
from .tool_history import is_repeated_successful_call
from .usage_estimator import usage_estimate_dict


def _http_timeout(cfg: Settings = settings) -> httpx.Timeout | None:
    """Build the effective upstream timeout from config.

    A value <= 0 keeps the previous "no overall timeout" behavior. If only an
    idle/read timeout is configured, keep connect bounded so dead TCP handshakes
    do not hang forever.
    """
    total = cfg.upstream_timeout_seconds if cfg.upstream_timeout_seconds > 0 else None
    args_done = cfg.args_done_timeout_ms / 1000 if cfg.args_done_timeout_ms > 0 else None
    read = cfg.upstream_idle_timeout_ms / 1000 if cfg.upstream_idle_timeout_ms > 0 else (total or args_done)
    connect = total if total is not None else 10.0
    if total is None and read is None:
        return None
    return httpx.Timeout(timeout=total, connect=connect, read=read, write=total, pool=total)


def _args_done_deadline(cfg: Settings = settings) -> float | None:
    return time.perf_counter() + cfg.args_done_timeout_ms / 1000 if cfg.args_done_timeout_ms > 0 else None


def _raise_if_args_done_timed_out(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise HTTPException(status_code=504, detail="timed out waiting for response.function_call_arguments.done")


def collect_bill015_result_from_events(events: Iterable[SSEEvent], n: NormalizedRequest, cfg: Settings = settings) -> Bill015Result:
    """Offline helper used by regression tests to validate event-state behavior."""
    result = Bill015Result(local_request_id=local_response_id())
    args_buffer: list[str] = []
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
            result.function_call_seen = item.get("name") == cfg.function_name or cfg.function_name in json.dumps(obj, ensure_ascii=False)
        elif typ == "response.function_call_arguments.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                args_buffer.append(delta)
        elif typ == "response.output_text.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                answer_buffer.append(delta)
        elif typ == "response.output_text.done":
            text = obj.get("text")
            if isinstance(text, str):
                result.answer = text
        elif typ == "response.function_call_arguments.done":
            final_args = obj.get("arguments") if isinstance(obj.get("arguments"), str) else "".join(args_buffer)
            result.raw_arguments = final_args
            result.args_done_seen = True
            result.answer, result.malformed_function_args, result.repaired_args, result.bridge_mode, result.tool_calls = parse_function_arguments(final_args, cfg, n.tool_registry)
            result.aborted = True
            break
        elif typ == "response.completed":
            result.upstream_completed_seen = True
            if not result.answer and answer_buffer:
                result.answer = "".join(answer_buffer)
            break
        elif typ == "response.incomplete":
            result.error = "upstream response incomplete"
            if not result.answer and answer_buffer:
                result.answer = "".join(answer_buffer)
            break
        elif typ in {"response.failed", "error"}:
            result.error = json.dumps(obj, ensure_ascii=False)[:2000]
            break
    return result

def audit_from_result(result: Bill015Result, n: NormalizedRequest, mode: str, fallback_used: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "local_request_id": result.local_request_id,
        "upstream_response_id": result.upstream_response_id,
        "mode": mode,
        "client_api": n.client_api,
        "model": n.model,
        "function_call_seen": result.function_call_seen,
        "args_done_seen": result.args_done_seen,
        "upstream_completed_seen": result.upstream_completed_seen,
        "aborted_at": "response.function_call_arguments.done" if result.aborted else None,
        "answer_chars": len(result.answer),
        "bridge_mode": result.bridge_mode,
        "tool_calls": [{"id": c.id, "name": c.name, "namespace": c.namespace, "requested_name": c.requested_name, "type": c.call_type, "arguments_chars": len(c.arguments)} for c in result.tool_calls],
        "duration_ms": result.duration_ms,
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
        async with httpx.AsyncClient(timeout=_http_timeout(cfg)) as client:
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
    args_buffer: list[str] = []
    answer_buffer: list[str] = []
    args_done_deadline = _args_done_deadline(cfg)
    headers = {
        "Authorization": "Bearer " + cfg.upstream_api_key,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "bill015-local-proxy/1.0",
    }
    try:
        async with httpx.AsyncClient(timeout=_http_timeout(cfg)) as client:
            async with client.stream("POST", cfg.upstream_base_url + "/v1/responses", headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise HTTPException(status_code=502, detail={"upstream_status": resp.status_code, "body": body[:2000].decode("utf-8", errors="replace")})
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
                        if item.get("type") in {"function_call", "tool_call"} or item.get("name") == cfg.function_name:
                            result.function_call_seen = True
                        else:
                            result.function_call_seen = result.function_call_seen or (cfg.function_name in json.dumps(obj, ensure_ascii=False))
                    elif typ == "response.function_call_arguments.delta":
                        delta = obj.get("delta")
                        if isinstance(delta, str):
                            args_buffer.append(delta)
                    elif typ == "response.output_text.delta":
                        delta = obj.get("delta")
                        if isinstance(delta, str):
                            answer_buffer.append(delta)
                    elif typ == "response.output_text.done":
                        text = obj.get("text")
                        if isinstance(text, str):
                            result.answer = text
                    elif typ == "response.function_call_arguments.done":
                        final_args = obj.get("arguments") if isinstance(obj.get("arguments"), str) else "".join(args_buffer)
                        result.raw_arguments = final_args
                        result.args_done_seen = True
                        result.answer, result.malformed_function_args, result.repaired_args, result.bridge_mode, result.tool_calls = parse_function_arguments(final_args, cfg, n.tool_registry)
                        await resp.aclose()
                        result.aborted = True
                        break
                    elif typ == "response.completed":
                        result.upstream_completed_seen = True
                        if not result.answer and answer_buffer:
                            result.answer = "".join(answer_buffer)
                        break
                    elif typ == "response.incomplete":
                        result.error = "upstream response incomplete"
                        if not result.answer and answer_buffer:
                            result.answer = "".join(answer_buffer)
                        break
                    elif typ in {"response.failed", "error"}:
                        raise RuntimeError(json.dumps(obj, ensure_ascii=False)[:2000])
        if mode == "verify":
            await asyncio.sleep(0.5)
            result.verify_post = await fetch_user_self(cfg)
            result.verify_delta = compute_delta(result.verify_pre, result.verify_post)
        if not result.args_done_seen and result.answer:
            result.bridge_mode = "answer"
        if not result.args_done_seen and not result.answer:
            raise RuntimeError("upstream stream ended before response.function_call_arguments.done")
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

def _prepend_instruction(existing: Any, extra: str) -> str:
    if isinstance(existing, str) and existing.strip():
        if extra in existing:
            return existing
        return extra + "\n\n" + existing
    return extra

def _compress_data_url(value: str, max_bytes: int = 900_000) -> str:
    if not value.startswith("data:image/") or ";base64," not in value or len(value) <= max_bytes:
        return value
    header, b64 = value.split(",", 1)
    try:
        raw = base64.b64decode(b64, validate=False)
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return "[image omitted by local proxy: data URL too large and Pillow unavailable]"
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((1280, 1280))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=72, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        return "[image omitted by local proxy: failed to compress large data URL]"

def _shrink_media(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _shrink_media(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shrink_media(v) for v in obj]
    if isinstance(obj, str):
        return _compress_data_url(obj)
    return obj

def prepare_passthrough_payload(body: dict[str, Any], cfg: Settings = settings) -> dict[str, Any]:
    """Native Codex Responses passthrough.

    In normal mode the proxy should behave like a thin translator, not like an
    agent protocol adapter. Preserve Codex's original Responses request shape
    exactly and only map the model name so CC Switch/Codex can select gpt-5.4
    or gpt-5.5 through this local provider.
    """
    payload = copy.deepcopy(body)
    payload["model"] = cfg.map_model(payload.get("model"))
    return payload

async def normal_forward_stream(body: dict[str, Any], cfg: Settings = settings) -> AsyncIterator[bytes]:
    if not cfg.upstream_api_key:
        yield encode_sse({"type": "error", "error": {"message": f"Missing upstream API key env {cfg.upstream_api_key_env}"}}, "error")
        yield b"data: [DONE]\n\n"
        return
    headers = {"Authorization": "Bearer " + cfg.upstream_api_key, "Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": "bill015-local-proxy/1.0"}
    try:
        async with httpx.AsyncClient(timeout=_http_timeout(cfg)) as client:
            async with client.stream("POST", cfg.upstream_base_url + "/v1/responses", headers=headers, json=prepare_passthrough_payload(body, cfg)) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    yield encode_sse({"type": "error", "error": {"message": body_bytes[:2000].decode("utf-8", errors="replace"), "upstream_status": resp.status_code, "type": "upstream_error"}}, "error")
                    yield b"data: [DONE]\n\n"
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
    except Exception as e:
        yield encode_sse({"type": "error", "error": {"message": f"{type(e).__name__}: {e}", "type": "local_proxy_error"}}, "error")
        yield b"data: [DONE]\n\n"

async def normal_forward_json(body: dict[str, Any], cfg: Settings = settings) -> dict[str, Any]:
    if not cfg.upstream_api_key:
        raise HTTPException(status_code=500, detail=f"Missing upstream API key env {cfg.upstream_api_key_env}")
    headers = {"Authorization": "Bearer " + cfg.upstream_api_key, "Content-Type": "application/json", "User-Agent": "bill015-local-proxy/1.0"}
    async with httpx.AsyncClient(timeout=_http_timeout(cfg)) as client:
        r = await client.post(cfg.upstream_base_url + "/v1/responses", headers=headers, json=prepare_passthrough_payload(body, cfg))
    try:
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail={"upstream_status": r.status_code, "body": r.text[:2000]}) from e
