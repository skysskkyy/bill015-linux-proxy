from __future__ import annotations

from collections.abc import Mapping

import httpx

from ..config import Settings, settings

NATIVE_USER_AGENT = "codex_cli_rs/0.153.4 (Linux 6.18.33; x86_64) rust"
NATIVE_ORIGINATOR = "codex_cli_rs"

_DROP_HEADERS = {
    "authorization",
    "cookie",
    "content-length",
    "content-encoding",
    "host",
    "connection",
    "transfer-encoding",
    "proxy-authorization",
    "expect",
}

_PASSTHROUGH_HEADERS = {
    "originator",
    "session-id",
    "thread-id",
    "conversation-id",
    "x-client-request-id",
    "openai-beta",
    "accept-language",
    "user-agent",
    "chatgpt-account-id",
}


def http_timeout(cfg: Settings = settings) -> httpx.Timeout:
    total = cfg.upstream_timeout_seconds if cfg.upstream_timeout_seconds > 0 else None
    read = max(cfg.args_done_timeout_ms / 1000, cfg.upstream_idle_timeout_ms / 1000, 30)
    return httpx.Timeout(timeout=total, connect=30.0, read=read, write=30.0, pool=30.0)


def codex_request_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_name, raw_value in (headers or {}).items():
        name = str(raw_name).strip()
        value = str(raw_value).strip()
        if not name or not value:
            continue
        low = name.lower()
        if low in _DROP_HEADERS:
            continue
        if (
            low in _PASSTHROUGH_HEADERS
            or low.startswith("x-codex-")
            or low.startswith("x-openai-")
            or low.startswith("x-stainless-")
        ):
            out[low] = value
    return out


def upstream_auth_headers(*, api_key: str, cfg: Settings = settings, client_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = codex_request_headers(client_headers)
    headers["authorization"] = f"Bearer {api_key}"
    headers["content-type"] = "application/json"
    headers["accept"] = "text/event-stream"
    if cfg.packy_cookie:
        headers["cookie"] = cfg.packy_cookie
    headers.setdefault("user-agent", NATIVE_USER_AGENT)
    headers.setdefault("originator", NATIVE_ORIGINATOR)
    # Keep the upstream request on the standard Responses path, not Lite.
    headers.pop("x-openai-internal-codex-responses-lite", None)
    return headers
