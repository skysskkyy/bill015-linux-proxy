from __future__ import annotations

import copy
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from .config import Settings, settings
from .sse import encode_sse


def http_timeout(cfg: Settings = settings) -> httpx.Timeout | None:
    """Build the effective upstream timeout from config."""
    total = cfg.upstream_timeout_seconds if cfg.upstream_timeout_seconds > 0 else None
    args_done = cfg.args_done_timeout_ms / 1000 if cfg.args_done_timeout_ms > 0 else None
    read = cfg.upstream_idle_timeout_ms / 1000 if cfg.upstream_idle_timeout_ms > 0 else (total or args_done)
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
                    yield encode_sse(
                        {
                            "type": "error",
                            "error": {
                                "message": body_bytes[:2000].decode("utf-8", errors="replace"),
                                "upstream_status": resp.status_code,
                                "type": "upstream_error",
                            },
                        },
                        "error",
                    )
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
    async with httpx.AsyncClient(timeout=http_timeout(cfg)) as client:
        r = await client.post(
            cfg.upstream_base_url + "/v1/responses",
            headers=upstream_auth_headers(cfg=cfg),
            json=prepare_passthrough_payload(body, cfg),
        )
    try:
        obj = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail={"upstream_status": r.status_code, "body": r.text[:2000]}) from e
    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(status_code=r.status_code, detail={"upstream_status": r.status_code, "body": obj})
    return obj
