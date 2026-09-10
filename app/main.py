from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .ingest import normalize_chat_request, normalize_responses_request, request_needs_passthrough
from .ops.audit import audit_logger
from .ops.capacity import CapacityLimiter, QueueFullError, QueueWaitTimeoutError
from .ops.state import runtime_state
from .protocol.models import Turn, TurnResult, local_response_id
from .replay import chat_json, chat_sse_generator, compact_output, response_json, responses_sse_generator
from .upstream import audit_from_result, dry_run_response, execute_turn, upstream_key_pool
from .upstream.client import codex_request_headers
from .upstream.passthrough import CompactionPassthroughResponse, is_responses_compaction


def project_version() -> str:
    try:
        return (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except Exception:
        return "0.0.0"


PROJECT_VERSION = project_version()
app = FastAPI(title="BILL-015 Linux Codex Desktop Proxy", version=PROJECT_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex or None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
capacity_limiter = CapacityLimiter(settings.max_concurrency, settings.max_queue_size, settings.queue_wait_timeout_ms, runtime_state)


def active_mode() -> str:
    snap = runtime_state.snapshot()
    mode = snap.get("mode_override") or settings.mode
    if settings.circuit_failures > 0 and mode in {"exploit", "verify"} and snap.get("consecutive_failures", 0) >= settings.circuit_failures:
        return "circuit-open"
    return mode


async def read_json_body(request: Request) -> dict[str, Any]:
    limit = max(1, settings.max_request_bytes)
    raw_content_length = request.headers.get("content-length")
    if raw_content_length:
        try:
            if int(raw_content_length) > limit:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid Content-Length header") from None
    body = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(body) + len(chunk) > limit:
            raise HTTPException(status_code=413, detail="request body too large")
        body.extend(chunk)
    request.state.raw_json_body = bytes(body)
    if not body:
        return {}
    try:
        obj = json.loads(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return obj


async def _const(value: Any) -> Any:
    return value


def require_admin(authorization: str | None) -> None:
    if not settings.admin_token:
        raise HTTPException(status_code=403, detail="LOCAL_PROXY_ADMIN_TOKEN is not configured")
    if authorization != "Bearer " + settings.admin_token:
        raise HTTPException(status_code=401, detail="invalid admin token")


async def run_and_record(turn: Turn, mode: str, local_request_id: str | None = None) -> TurnResult:
    runtime_state.inc_request()
    start = time.perf_counter()
    request_id = local_request_id or local_response_id()
    try:
        lease = await capacity_limiter.acquire()
    except QueueFullError as e:
        runtime_state.mark_error("proxy queue is full")
        raise HTTPException(status_code=429, detail="proxy queue is full; retry later") from e
    except QueueWaitTimeoutError as e:
        runtime_state.mark_error("proxy queue wait timed out")
        raise HTTPException(status_code=503, detail="proxy stayed busy longer than queue_wait_timeout_ms; retry later") from e
    try:
        try:
            async with asyncio.timeout(max(0.001, settings.request_total_timeout_ms / 1000)):
                result = await execute_turn(turn, mode, local_request_id=request_id)
            runtime_state.mark_success(args_done=result.args_done_seen, aborted=result.aborted)
            record = audit_from_result(result, turn, mode)
            record["queue_wait_ms"] = lease.queue_wait_ms
            audit_logger.write(record)
            return result
        except TimeoutError as e:
            runtime_state.mark_request_timeout()
            runtime_state.mark_error("proxy request exceeded request_total_timeout_ms")
            audit_logger.write(
                {
                    "local_request_id": request_id,
                    "mode": mode,
                    "error": "proxy request exceeded request_total_timeout_ms",
                    "queue_wait_ms": lease.queue_wait_ms,
                    "duration_ms": int((time.perf_counter() - start) * 1000),
                }
            )
            raise HTTPException(status_code=504, detail="proxy request exceeded request_total_timeout_ms") from e
        except HTTPException as e:
            runtime_state.mark_error(str(e.detail))
            audit_logger.write({"local_request_id": request_id, "mode": mode, "error": e.detail, "queue_wait_ms": lease.queue_wait_ms})
            raise
        except Exception as e:
            runtime_state.mark_error(f"{type(e).__name__}: {e}")
            audit_logger.write({"local_request_id": request_id, "mode": mode, "error": f"{type(e).__name__}: {e}"})
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
    finally:
        await lease.release()


@app.get("/")
async def root() -> dict[str, Any]:
    return {"service": "bill015-linux-proxy", "version": PROJECT_VERSION, "health": "/healthz", "models": "/v1/models"}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "version": PROJECT_VERSION,
        "mode": active_mode(),
        "configured_mode": settings.mode,
        "upstream_base_url": settings.upstream_base_url,
        "upstream_api_key_present": settings.upstream_configured,
        "host": settings.host,
        "port": settings.port,
        "max_concurrency": settings.max_concurrency,
        "max_queue_size": settings.max_queue_size,
        "safe_bridge": {
            "force_emit_value": settings.force_emit_value,
            "block_passthrough": settings.block_passthrough,
            "block_normal_mode": settings.block_normal_mode,
        },
        "config_warnings": settings.config_warnings,
        "metrics": runtime_state.snapshot(),
        "upstream_keys": upstream_key_pool(settings).snapshot(),
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    return runtime_state.snapshot()


@app.get("/v1")
async def v1_root() -> dict[str, Any]:
    return {
        "service": "bill015-linux-proxy",
        "object": "api_root",
        "compatible": ["openai_responses", "openai_chat_completions"],
        "responses": "/v1/responses",
        "chat_completions": "/v1/chat/completions",
        "models": "/v1/models",
        "mode": active_mode(),
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "created": now, "owned_by": "bill015-linux-proxy"} for mid in settings.public_model_ids()],
    }


async def handle_responses_body(
    body: dict[str, Any], request_headers: dict[str, str] | None = None, *, raw_body: bytes | None = None
):
    mode = active_mode()
    if mode == "circuit-open":
        raise HTTPException(status_code=503, detail="circuit breaker open after consecutive upstream failures")
    # Compaction metadata describes the operation, not a request to change endpoints.
    # Keep this before normalization so native input, tools, and instructions survive.
    if mode != "dry-run" and is_responses_compaction(body, request_headers):
        return CompactionPassthroughResponse(
            body, request_headers, raw_body=raw_body, mode=mode, limiter=capacity_limiter
        )
    try:
        turn = normalize_responses_request(body, request_headers=codex_request_headers(request_headers))
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"code": "local_proxy_vision_unsupported", "message": str(e)}) from e
    if mode == "dry-run":
        dry = dry_run_response(turn)
        if turn.want_stream:
            result = TurnResult(local_request_id=dry["id"], answer=json.dumps(dry, ensure_ascii=False), args_done_seen=False)
            return StreamingResponse(
                responses_sse_generator(_const(result), turn, dry["id"]),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        runtime_state.inc_request()
        runtime_state.mark_success()
        audit_logger.write(audit_from_result(TurnResult(local_request_id=dry["id"]), turn, "dry-run"))
        return JSONResponse(dry)

    needs_passthrough, passthrough_reason = request_needs_passthrough(body)
    if mode in {"exploit", "verify"} and needs_passthrough and settings.block_passthrough:
        raise HTTPException(
            status_code=422,
            detail=f"safe bridge policy blocked auto-passthrough because this request needs native upstream passthrough ({passthrough_reason}).",
        )
    if mode == "normal":
        if settings.block_normal_mode:
            raise HTTPException(status_code=409, detail="safe bridge policy blocked normal forwarding; use exploit/verify BILL-015 bridge mode.")
        mode = "exploit"

    if turn.want_stream:
        rid = local_response_id()
        return StreamingResponse(
            responses_sse_generator(run_and_record(turn, mode, rid), turn, rid),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await run_and_record(turn, mode)
    return JSONResponse(response_json(result, turn))


@app.post("/v1/responses")
async def responses(request: Request):
    body = await read_json_body(request)
    return await handle_responses_body(body, dict(request.headers), raw_body=request.state.raw_json_body)


@app.post("/v1/responses/compact")
async def responses_compact(request: Request):
    body = await read_json_body(request)
    body = dict(body)
    body["stream"] = False
    cm = dict(body.get("client_metadata") or {})
    raw = cm.get("x-codex-turn-metadata")
    try:
        md = json.loads(raw) if isinstance(raw, str) and raw else {}
    except Exception:
        md = {}
    if not isinstance(md, dict):
        md = {}
    md["request_kind"] = "compaction"
    cm["x-codex-turn-metadata"] = json.dumps(md, ensure_ascii=False)
    body["client_metadata"] = cm
    try:
        turn = normalize_responses_request(body, request_headers=codex_request_headers(dict(request.headers)))
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"code": "local_proxy_vision_unsupported", "message": str(e)}) from e
    turn.is_compaction = True
    turn.request_kind = "compaction"
    mode = active_mode()
    if mode == "dry-run":
        result = TurnResult(local_request_id=local_response_id(), answer="Compact dry-run summary.", args_done_seen=False)
    else:
        if mode == "normal":
            if settings.block_normal_mode:
                raise HTTPException(status_code=409, detail="safe bridge policy blocked normal compact forwarding")
            mode = "exploit"
        result = await run_and_record(turn, mode)
    return JSONResponse(compact_output(result, turn))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        turn = normalize_chat_request(await read_json_body(request), request_headers=codex_request_headers(dict(request.headers)))
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"code": "local_proxy_vision_unsupported", "message": str(e)}) from e
    mode = active_mode()
    if mode == "circuit-open":
        raise HTTPException(status_code=503, detail="circuit breaker open after consecutive upstream failures")
    if mode == "dry-run":
        dry = dry_run_response(turn)
        result = TurnResult(local_request_id=dry["id"], answer=json.dumps(dry, ensure_ascii=False))
        if turn.want_stream:
            return StreamingResponse(chat_sse_generator(_const(result), turn), media_type="text/event-stream")
        return JSONResponse(chat_json(result, turn))
    if mode == "normal":
        mode = "exploit"
    if turn.want_stream:
        return StreamingResponse(chat_sse_generator(run_and_record(turn, mode), turn), media_type="text/event-stream")
    result = await run_and_record(turn, mode)
    return JSONResponse(chat_json(result, turn))


@app.post("/admin/mode")
async def admin_mode(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin(authorization)
    body = await read_json_body(request)
    mode = str(body.get("mode", "")).strip().lower()
    if mode not in {"exploit", "verify", "normal", "dry-run"}:
        raise HTTPException(status_code=400, detail="mode must be exploit/verify/normal/dry-run")
    if settings.block_normal_mode and mode == "normal":
        raise HTTPException(status_code=400, detail="safe bridge policy forbids normal mode because it forwards billable upstream calls")
    runtime_state.current_mode_override = mode
    audit_logger.write({"admin_action": "mode", "mode": mode})
    return {"ok": True, "mode": mode}


@app.get("/admin/recent")
async def admin_recent(n: int = 20, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin(authorization)
    return {"items": audit_logger.recent(n)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "local_proxy_error"}})
