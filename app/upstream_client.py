from __future__ import annotations

import copy
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from .config import Settings, settings
from .models import local_response_id
from .sse import encode_sse
from .upstream_errors import error_message_from_detail, sanitize_upstream_error_detail


def http_timeout(cfg: Settings = settings) -> httpx.Timeout | None:
    """Build the effective upstream timeout from config."""
    total = cfg.upstream_timeout_seconds if cfg.upstream_timeout_seconds > 0 else None
    args_done = cfg.args_done_timeout_ms / 1000 if cfg.args_done_timeout_ms > 0 else None
    configured_read = cfg.upstream_idle_timeout_ms / 1000 if cfg.upstream_idle_timeout_ms > 0 else None
    read_candidates = [value for value in (configured_read, args_done, total) if value is not None]
    read = max(read_candidates) if read_candidates else None
    connect = total if total is not None else 10.0
    if total is None and read is None:
        return None
    return httpx.Timeout(timeout=total, connect=connect, read=read, write=total, pool=total)


def upstream_auth_headers(*, stream: bool = False, cfg: Settings = settings) -> dict[str, str]:
    headers = {
        "Authorization": "Bearer " + cfg.upstream_api_key,
        "Content-Type": "application/json",
        "User-Agent": "bill015-local-proxy/1.0",
    }
    if stream:
        headers["Accept"] = "text/event-stream"
    return headers


def prepare_passthrough_payload(body: dict[str, Any], cfg: Settings = settings) -> dict[str, Any]:
    """Native Codex Responses passthrough.

    Preserve the original Responses request shape and only map the model name so
    CC Switch/Codex can select configured upstream models through this provider.
    """
    payload = copy.deepcopy(body)
    payload["model"] = cfg.map_model(payload.get("model"))
    return payload


def _passthrough_failed_events(status_code: int, detail: dict[str, Any], body: dict[str, Any], cfg: Settings) -> list[bytes]:
    message = error_message_from_detail(detail)
    error = {
        "code": "upstream_error",
        "message": message,
        "type": "server_error" if status_code >= 500 else "invalid_request_error",
        "upstream_status": detail.get("upstream_status", status_code),
    }
    rid = local_response_id()
    now = int(time.time())
    response = {
        "id": rid,
        "object": "response",
        "created_at": now,
        "status": "failed",
        "background": False,
        "completed_at": now,
        "error": error,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": body.get("max_output_tokens"),
        "max_tool_calls": None,
        "model": str(body.get("model") or cfg.default_model),
        "output": [],
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
        "previous_response_id": body.get("previous_response_id") if isinstance(body.get("previous_response_id"), str) else None,
        "prompt_cache_key": body.get("prompt_cache_key") if isinstance(body.get("prompt_cache_key"), str) else None,
        "prompt_cache_retention": None,
        "reasoning": body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {},
        "store": bool(body.get("store", False)),
        "temperature": body.get("temperature"),
        "text": body.get("text") if isinstance(body.get("text"), dict) else {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") if isinstance(body.get("tools"), list) else [],
        "tool_usage": None,
        "top_p": body.get("top_p"),
        "truncation": body.get("truncation", "auto"),
        "usage": None,
        "user": None,
        "metadata": body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
    }
    return [
        encode_sse({"type": "response.failed", "response": response, "sequence_number": 0}, "response.failed"),
        encode_sse({"type": "error", "error": error, "sequence_number": 1}, "error"),
        b"data: [DONE]\n\n",
    ]


async def normal_forward_stream(body: dict[str, Any], cfg: Settings = settings) -> AsyncIterator[bytes]:
    if not cfg.upstream_api_key:
        yield encode_sse({"type": "error", "error": {"message": f"Missing upstream API key env {cfg.upstream_api_key_env}"}}, "error")
        yield b"data: [DONE]\n\n"
        return
    try:
        async with httpx.AsyncClient(timeout=http_timeout(cfg)) as client:
            async with client.stream(
                "POST",
                cfg.upstream_base_url + "/v1/responses",
                headers=upstream_auth_headers(stream=True, cfg=cfg),
                json=prepare_passthrough_payload(body, cfg),
            ) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    detail = sanitize_upstream_error_detail(
                        resp.status_code,
                        body_bytes,
                        content_type=resp.headers.get("content-type", ""),
                    )
                    for event in _passthrough_failed_events(resp.status_code, detail, body, cfg):
                        yield event
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
    except Exception as e:
        yield encode_sse({"type": "error", "error": {"message": f"{type(e).__name__}: {e}", "type": "local_proxy_error"}}, "error")
        yield b"data: [DONE]\n\n"


async def normal_forward_json(body: dict[str, Any], cfg: Settings = settings) -> dict[str, Any]:
    if not cfg.upstream_api_key:
        raise HTTPException(status_code=500, detail=f"Missing upstream API key env {cfg.upstream_api_key_env}")
    async with httpx.AsyncClient(timeout=http_timeout(cfg)) as client:
        r = await client.post(
            cfg.upstream_base_url + "/v1/responses",
            headers=upstream_auth_headers(cfg=cfg),
            json=prepare_passthrough_payload(body, cfg),
        )
    try:
        obj = r.json()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=sanitize_upstream_error_detail(
                r.status_code,
                r.text,
                content_type=r.headers.get("content-type", ""),
            ),
        ) from e
    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(
            status_code=r.status_code,
            detail=sanitize_upstream_error_detail(r.status_code, obj, preserve_json_body=True),
        )
    return obj
