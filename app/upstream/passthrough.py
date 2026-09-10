from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack
from typing import Any

import anyio
import httpx
from fastapi import HTTPException
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Message, Receive, Scope, Send

from ..config import Settings, settings
from ..ingest.turn import parse_codex_turn_metadata
from ..ops.audit import audit_logger, redact
from ..ops.capacity import CapacityLimiter, QueueFullError, QueueWaitTimeoutError
from ..ops.state import runtime_state
from ..protocol.models import local_response_id
from .client import codex_request_headers, http_timeout, upstream_auth_headers
from .keys import upstream_key_pool

_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}


def is_responses_compaction(body: dict[str, Any], headers: dict[str, str] | None = None) -> bool:
    metadata = parse_codex_turn_metadata(body.get("client_metadata"))
    if not metadata and not body.get("request_kind"):
        native_headers = codex_request_headers(headers)
        metadata = parse_codex_turn_metadata({
            "x-codex-turn-metadata": native_headers.get("x-codex-turn-metadata")
        })
    kind = metadata.get("request_kind") or body.get("request_kind")
    return kind == "compaction" or bool(metadata.get("compaction"))


class CompactionPassthroughResponse(Response):
    """Own the upstream connection and capacity lease for the entire ASGI response."""

    def __init__(
        self, body: dict[str, Any], client_headers: dict[str, str] | None = None, *,
        raw_body: bytes | None = None, mode: str, limiter: CapacityLimiter, cfg: Settings = settings,
    ) -> None:
        super().__init__()
        self.payload = body
        self.raw_body = raw_body if raw_body is not None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.client_headers = client_headers
        self.mode = mode
        self.limiter = limiter
        self.cfg = cfg

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = time.perf_counter()
        response_started = False
        response_finished = False
        record: dict[str, Any] = {
            "local_request_id": local_response_id(),
            "mode": self.mode,
            "request_kind": "compaction",
            "transport": "native_passthrough",
            "upstream_path": "/v1/responses",
            "model": self.payload.get("model"),
        }
        stack = AsyncExitStack()
        runtime_state.inc_request()

        async def tracked_send(message: Message) -> None:
            nonlocal response_started, response_finished
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                response_finished = True

        try:
            lease = await self.limiter.acquire()
            stack.push_async_callback(lease.release)
            record["queue_wait_ms"] = lease.queue_wait_ms
            selection = upstream_key_pool(self.cfg).current()
            if selection is None:
                raise HTTPException(status_code=500, detail="Missing upstream API key for compaction")
            headers = upstream_auth_headers(api_key=selection.key, cfg=self.cfg, client_headers=self.client_headers)
            # Unlike the emit_value bridge, passthrough must preserve native Lite markers.
            headers.update(codex_request_headers(self.client_headers))
            headers["accept"] = "text/event-stream" if self.payload.get("stream", True) else "application/json"
            headers["accept-encoding"] = "identity"
            async with asyncio.timeout(max(0.001, self.cfg.request_total_timeout_ms / 1000)):
                client = await stack.enter_async_context(httpx.AsyncClient(timeout=http_timeout(self.cfg)))
                upstream = await stack.enter_async_context(client.stream(
                    "POST", self.cfg.upstream_base_url.rstrip("/") + "/v1/responses",
                    headers=headers, content=self.raw_body,
                ))
                record["upstream_status"] = upstream.status_code
                record["upstream_content_type"] = upstream.headers.get("content-type", "")
                record["upstream_request_id"] = upstream.headers.get("x-request-id") or upstream.headers.get("request-id")
                excluded = _HOP_HEADERS | {part.strip().lower() for part in upstream.headers.get("connection", "").split(",")}
                response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
                response_headers["x-accel-buffering"] = "no"
                response = StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=response_headers)
                await response(scope, receive, tracked_send)
            if response_finished:
                if upstream.status_code >= 400:
                    record["error"] = f"upstream returned HTTP {upstream.status_code}"
                    runtime_state.mark_error(record["error"])
                else:
                    runtime_state.mark_success()
            else:
                record["client_disconnected"] = True
        except asyncio.CancelledError:
            record["client_disconnected"] = True
            raise
        except Exception as exc:
            if isinstance(exc, QueueFullError):
                status, detail = 429, "proxy queue is full; retry later"
            elif isinstance(exc, QueueWaitTimeoutError):
                status, detail = 503, "proxy queue wait timed out"
            elif isinstance(exc, TimeoutError):
                status, detail = 504, "proxy compaction exceeded request_total_timeout_ms"
                runtime_state.mark_request_timeout()
            elif isinstance(exc, HTTPException):
                status, detail = exc.status_code, str(exc.detail)
            else:
                status, detail = 502, f"{type(exc).__name__}: {exc}"
            record["error"] = redact(detail)
            runtime_state.mark_error(record["error"])
            if response_started:
                # Headers/body already went out: leave the stream incomplete for the
                # client to handle, rather than inventing a successful completion.
                raise
            await JSONResponse(status_code=status, content={
                "error": {"message": record["error"], "type": "local_proxy_error"}
            })(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await stack.aclose()
                finally:
                    record["duration_ms"] = int((time.perf_counter() - started) * 1000)
                    audit_logger.write(record)
        if self.background is not None:
            await self.background()
