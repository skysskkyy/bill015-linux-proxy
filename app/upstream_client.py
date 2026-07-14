from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from .config import Settings, settings
from .key_pool import ApiKeySelection, upstream_key_pool
from .models import local_response_id
from .sse import encode_sse
from .upstream_errors import error_message_from_detail, key_rotation_error_reason, sanitize_upstream_error_detail

KEY_ROTATION_DELAY_SECONDS = 10 * 60


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


RESPONSES_LITE_HEADER = "x-openai-internal-codex-responses-lite"


def request_uses_responses_lite(body: dict[str, Any]) -> bool:
    """Mirror Codex's Responses Lite request marker for passthrough calls.

    Codex core sets an internal header whenever model metadata enables
    Responses Lite.  In the HTTP body that commonly corresponds to an
    ``additional_tools`` developer input item, and some callers also carry an
    explicit marker in ``client_metadata``.  Keep passthrough and bridge paths
    aligned so upstream sees the same request class Codex intended.
    """
    metadata = body.get("client_metadata") if isinstance(body.get("client_metadata"), dict) else {}
    for key in (RESPONSES_LITE_HEADER, "responses_lite", "use_responses_lite"):
        value = body.get(key, metadata.get(key))
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "responses_lite"}:
            return True
    raw_input = body.get("input")
    if isinstance(raw_input, list):
        return any(isinstance(item, dict) and str(item.get("type") or "") in {"additional_tools", "additionalTools"} for item in raw_input)
    return False


def upstream_auth_headers(*, stream: bool = False, cfg: Settings = settings, api_key: str | None = None, responses_lite: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": "Bearer " + (api_key or cfg.upstream_api_key),
        "Content-Type": "application/json",
        "User-Agent": "bill015-local-proxy/1.0",
    }
    if stream:
        headers["Accept"] = "text/event-stream"
    if responses_lite:
        headers[RESPONSES_LITE_HEADER] = "true"
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
        "prompt_cache_options": body.get("prompt_cache_options") if isinstance(body.get("prompt_cache_options"), dict) else None,
        "prompt_cache_retention": None,
        "reasoning": body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {},
        "service_tier": body.get("service_tier") if isinstance(body.get("service_tier"), str) else None,
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


async def _next_key_after_rotation_error(
    selection: ApiKeySelection,
    tried_keys: set[str],
    cfg: Settings,
) -> ApiKeySelection | None:
    await asyncio.sleep(KEY_ROTATION_DELAY_SECONDS)
    next_selection = upstream_key_pool(cfg).rotate_after_failure(selection.key, tried_keys)
    if next_selection is not None:
        tried_keys.add(next_selection.key)
    return next_selection


async def normal_forward_stream(body: dict[str, Any], cfg: Settings = settings) -> AsyncIterator[bytes]:
    selection = upstream_key_pool(cfg).current()
    if selection is None:
        yield encode_sse({"type": "error", "error": {"message": f"Missing upstream API key env {cfg.upstream_api_key_env}"}}, "error")
        yield b"data: [DONE]\n\n"
        return
    tried_keys = {selection.key}

    try:
        async with httpx.AsyncClient(timeout=http_timeout(cfg)) as client:
            while True:
                async with client.stream(
                    "POST",
                    cfg.upstream_base_url + "/v1/responses",
                    headers=upstream_auth_headers(stream=True, cfg=cfg, api_key=selection.key, responses_lite=request_uses_responses_lite(body)),
                    json=prepare_passthrough_payload(body, cfg),
                ) as resp:
                    if resp.status_code != 200:
                        body_bytes = await resp.aread()
                        if key_rotation_error_reason(body_bytes):
                            next_selection = await _next_key_after_rotation_error(selection, tried_keys, cfg)
                            if next_selection is not None:
                                selection = next_selection
                                continue
                        detail = sanitize_upstream_error_detail(
                            resp.status_code, body_bytes, content_type=resp.headers.get("content-type", "")
                        )
                        for event in _passthrough_failed_events(resp.status_code, detail, body, cfg):
                            yield event
                        return

                    chunks: list[bytes] = []
                    async for chunk in resp.aiter_bytes():
                        chunks.append(chunk)
                    body_bytes = b"".join(chunks)
                    if key_rotation_error_reason(body_bytes):
                        next_selection = await _next_key_after_rotation_error(selection, tried_keys, cfg)
                        if next_selection is not None:
                            selection = next_selection
                            continue
                    yield body_bytes
                    return
    except Exception as e:
        yield encode_sse({"type": "error", "error": {"message": f"{type(e).__name__}: {e}", "type": "local_proxy_error"}}, "error")
        yield b"data: [DONE]\n\n"


async def normal_forward_json(body: dict[str, Any], cfg: Settings = settings) -> dict[str, Any]:
    selection = upstream_key_pool(cfg).current()
    if selection is None:
        raise HTTPException(status_code=500, detail=f"Missing upstream API key env {cfg.upstream_api_key_env}")
    tried_keys = {selection.key}

    async with httpx.AsyncClient(timeout=http_timeout(cfg)) as client:
        while True:
            r = await client.post(
                cfg.upstream_base_url + "/v1/responses",
                headers=upstream_auth_headers(cfg=cfg, api_key=selection.key, responses_lite=request_uses_responses_lite(body)),
                json=prepare_passthrough_payload(body, cfg),
            )
            try:
                obj = r.json()
            except Exception as e:
                if not 200 <= r.status_code < 300 and key_rotation_error_reason(r.text):
                    next_selection = await _next_key_after_rotation_error(selection, tried_keys, cfg)
                    if next_selection is not None:
                        selection = next_selection
                        continue
                raise HTTPException(
                    status_code=502,
                    detail=sanitize_upstream_error_detail(
                        r.status_code, r.text, content_type=r.headers.get("content-type", "")
                    ),
                ) from e

            if not 200 <= r.status_code < 300:
                if key_rotation_error_reason(obj):
                    next_selection = await _next_key_after_rotation_error(selection, tried_keys, cfg)
                    if next_selection is not None:
                        selection = next_selection
                        continue
                raise HTTPException(
                    status_code=r.status_code,
                    detail=sanitize_upstream_error_detail(r.status_code, obj, preserve_json_body=True),
                )
            return obj
