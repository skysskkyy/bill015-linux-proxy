from __future__ import annotations

import httpx

from ..config import Settings, settings


def http_timeout(cfg: Settings = settings) -> httpx.Timeout:
    total = cfg.upstream_timeout_seconds if cfg.upstream_timeout_seconds > 0 else None
    read = max(cfg.args_done_timeout_ms / 1000, cfg.upstream_idle_timeout_ms / 1000, 30)
    return httpx.Timeout(timeout=total, connect=30.0, read=read, write=30.0, pool=30.0)


def upstream_auth_headers(*, api_key: str, cfg: Settings = settings, client_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if cfg.packy_cookie:
        headers["Cookie"] = cfg.packy_cookie
    # Never forward Codex Lite markers to upstream in the safe bridge.
    blocked = {"x-openai-internal-codex-responses-lite", "openai-beta"}
    for key, value in (client_headers or {}).items():
        low = key.lower()
        if low in blocked or low.startswith("x-openai-internal-"):
            continue
        if low in {"authorization", "cookie", "content-length", "host"}:
            continue
        if low.startswith("x-codex-") or low in {"originator", "user-agent"}:
            headers[key] = value
    return headers


def codex_request_headers(headers: dict[str, str] | None) -> dict[str, str]:
    return {str(k): str(v) for k, v in (headers or {}).items()}
