from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .audit import audit_logger
from .chat_events import chat_json, chat_sse_generator
from .config import settings
from .models import Bill015Result, NormalizedRequest, local_response_id
from .normalization import normalize_chat_request, normalize_responses_request, request_needs_passthrough
from .response_events import response_json, responses_sse_generator
from .state import runtime_state
from .upstream import audit_from_result, dry_run_response, execute_bill015
from .upstream_client import normal_forward_json, normal_forward_stream


def project_version() -> str:
    try:
        return (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except Exception:
        return "0.0.0"


PROJECT_VERSION = project_version()

app = FastAPI(title="BILL-015 Local Codex Proxy", version=PROJECT_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex or None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
semaphore = asyncio.Semaphore(settings.max_concurrency)


def active_mode() -> str:
    snap = runtime_state.snapshot()
    mode = snap.get("mode_override") or settings.mode
    if settings.circuit_failures > 0 and mode in {"exploit", "verify"} and snap.get("consecutive_failures", 0) >= settings.circuit_failures:
        return "circuit-open"
    return mode


async def read_json_body(request: Request) -> dict[str, Any]:
    body = await request.body()
    if len(body) > settings.max_request_bytes:
        raise HTTPException(status_code=413, detail="request body too large")
    if not body:
        return {}
    try:
        obj = json.loads(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return obj


def require_admin(authorization: str | None) -> None:
    if not settings.admin_token:
        raise HTTPException(status_code=403, detail="LOCAL_PROXY_ADMIN_TOKEN is not configured")
    expected = "Bearer " + settings.admin_token
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid admin token")


async def run_and_record(n: NormalizedRequest, mode: str, local_request_id: str | None = None) -> Bill015Result:
    runtime_state.inc_request()
    start = time.perf_counter()
    async with semaphore:
        try:
            result = await execute_bill015(n, mode, local_request_id=local_request_id)
            runtime_state.mark_success(args_done=result.args_done_seen, aborted=result.aborted)
            audit_logger.write(audit_from_result(result, n, mode))
            return result
        except HTTPException as e:
            runtime_state.mark_error(str(e.detail))
            audit_logger.write({
                "local_request_id": local_response_id(),
                "mode": mode,
                "client_api": n.client_api,
                "model": n.model,
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "error": e.detail,
                "prompt_chars": len(n.user_input),
            })
            raise
        except Exception as e:
            runtime_state.mark_error(f"{type(e).__name__}: {e}")
            audit_logger.write({
                "local_request_id": local_response_id(),
                "mode": mode,
                "client_api": n.client_api,
                "model": n.model,
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "error": f"{type(e).__name__}: {e}",
                "prompt_chars": len(n.user_input),
            })
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e


async def synthetic_result(answer: str, n: NormalizedRequest) -> Bill015Result:
    result = Bill015Result(local_request_id=local_response_id(), answer=answer, args_done_seen=True, aborted=False)
    audit_logger.write(audit_from_result(result, n, "dry-run"))
    runtime_state.inc_request()
    runtime_state.mark_success(args_done=False, aborted=False)
    return result


@app.get("/")
async def root() -> dict[str, Any]:
    return {"service": "bill015-local-proxy", "version": PROJECT_VERSION, "health": "/healthz", "models": "/v1/models"}


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
        "strict_zero": settings.strict_zero,
        "config_warnings": getattr(settings, "config_warnings", []),
        "metrics": runtime_state.snapshot(),
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    return runtime_state.snapshot()




@app.get("/v1")
async def v1_root() -> dict[str, Any]:
    return {
        "service": "bill015-local-proxy",
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
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "bill015-local-proxy"}
            for mid in settings.public_model_ids()
        ],
    }


def _force_compaction_metadata(body: dict[str, Any]) -> dict[str, Any]:
    body = dict(body)
    body["tools"] = []
    body["parallel_tool_calls"] = False
    cm = dict(body.get("client_metadata") or {})
    raw = cm.get("x-codex-turn-metadata")
    try:
        md = json.loads(raw) if isinstance(raw, str) and raw else {}
    except Exception:
        md = {}
    if not isinstance(md, dict):
        md = {}
    md["request_kind"] = "compaction"
    md.setdefault("compaction", {"trigger": "explicit", "reason": "responses_compact_endpoint", "implementation": "responses", "phase": "mid_turn", "strategy": "memento"})
    cm["x-codex-turn-metadata"] = json.dumps(md, ensure_ascii=False)
    body["client_metadata"] = cm
    return body


async def handle_responses_body(body: dict[str, Any]):
    n = normalize_responses_request(body)
    mode = active_mode()
    if mode == "circuit-open":
        raise HTTPException(status_code=503, detail="circuit breaker open after consecutive upstream failures")

    if mode == "dry-run":
        dry = dry_run_response(n)
        if n.want_stream:
            answer = json.dumps(dry, ensure_ascii=False, indent=2)
            rid = local_response_id()
            return StreamingResponse(responses_sse_generator(synthetic_result(answer, n), n, rid), media_type="text/event-stream", headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"})
        result = Bill015Result(local_request_id=dry["id"], answer=json.dumps(dry, ensure_ascii=False), args_done_seen=False, aborted=False)
        runtime_state.inc_request()
        runtime_state.mark_success(args_done=False, aborted=False)
        audit_logger.write(audit_from_result(result, n, "dry-run"))
        return JSONResponse(dry)

    needs_passthrough, passthrough_reason = request_needs_passthrough(body)
    if mode in {"exploit", "verify"} and needs_passthrough:
        if settings.strict_zero:
            runtime_state.inc_request()
            runtime_state.mark_error(f"strict_zero blocked auto-passthrough: {passthrough_reason}")
            audit_logger.write({
                "local_request_id": local_response_id(),
                "mode": "passthrough-blocked",
                "client_api": "responses",
                "model": n.model,
                "reason": passthrough_reason,
                "prompt_chars": len(n.user_input),
            })
            raise HTTPException(
                status_code=422,
                detail=(
                    "strict_zero blocked auto-passthrough because this request needs native upstream "
                    f"passthrough ({passthrough_reason}). This prevents accidental billable/non-aborted calls."
                ),
            )
        runtime_state.inc_request()
        runtime_state.mark_fallback()
        audit_logger.write({
            "local_request_id": local_response_id(),
            "mode": "auto-passthrough",
            "client_api": "responses",
            "model": n.model,
            "reason": passthrough_reason,
            "prompt_chars": len(n.user_input),
        })
        if n.want_stream:
            return StreamingResponse(normal_forward_stream(body), media_type="text/event-stream", headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"})
        return JSONResponse(await normal_forward_json(body))

    if mode == "normal":
        if settings.strict_zero:
            runtime_state.inc_request()
            runtime_state.mark_error("strict_zero blocked normal forwarding")
            audit_logger.write({
                "local_request_id": local_response_id(),
                "mode": "normal-blocked",
                "client_api": "responses",
                "model": n.model,
                "prompt_chars": len(n.user_input),
            })
            raise HTTPException(status_code=409, detail="strict_zero blocked normal forwarding; use exploit/verify BILL-015 bridge mode.")
        runtime_state.inc_request()
        runtime_state.mark_fallback()
        audit_logger.write({"local_request_id": local_response_id(), "mode": "normal", "client_api": "responses", "model": n.model, "prompt_chars": len(n.user_input)})
        if n.want_stream:
            return StreamingResponse(normal_forward_stream(body), media_type="text/event-stream", headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"})
        return JSONResponse(await normal_forward_json(body))

    if n.want_stream:
        rid = local_response_id()
        return StreamingResponse(responses_sse_generator(run_and_record(n, mode, rid), n, rid), media_type="text/event-stream", headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"})
    result = await run_and_record(n, mode)
    return JSONResponse(response_json(result, n))


@app.post("/v1/responses")
async def responses(request: Request):
    return await handle_responses_body(await read_json_body(request))


@app.post("/v1/responses/compact")
async def responses_compact(request: Request):
    # Compatibility with relays that expose ResponsesCompact. Native Codex logs
    # mostly send compaction to /v1/responses with request_kind=compaction, but
    # this alias preserves the same local zero/low-usage bridge path if a client
    # uses /compact.
    return await handle_responses_body(_force_compaction_metadata(await read_json_body(request)))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await read_json_body(request)
    n = normalize_chat_request(body)
    mode = active_mode()
    if mode == "circuit-open":
        raise HTTPException(status_code=503, detail="circuit breaker open after consecutive upstream failures")

    if mode == "dry-run":
        dry = dry_run_response(n)
        answer = json.dumps(dry, ensure_ascii=False, indent=2)
        if n.want_stream:
            return StreamingResponse(chat_sse_generator(synthetic_result(answer, n), n), media_type="text/event-stream")
        result = await synthetic_result(answer, n)
        return JSONResponse(chat_json(result, n))

    if mode == "normal":
        # Chat is normalized into Responses for compatibility instead of direct /chat upstream.
        mode = "exploit"

    if n.want_stream:
        return StreamingResponse(chat_sse_generator(run_and_record(n, mode), n), media_type="text/event-stream")
    result = await run_and_record(n, mode)
    return JSONResponse(chat_json(result, n))


@app.post("/admin/mode")
async def admin_mode(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin(authorization)
    body = await read_json_body(request)
    mode = str(body.get("mode", "")).strip().lower()
    if mode not in {"exploit", "verify", "normal", "dry-run"}:
        raise HTTPException(status_code=400, detail="mode must be exploit/verify/normal/dry-run")
    if settings.strict_zero and mode == "normal":
        raise HTTPException(status_code=400, detail="strict_zero forbids normal mode because it forwards billable upstream calls")
    runtime_state.current_mode_override = mode
    audit_logger.write({"admin_action": "mode", "mode": mode})
    return {"ok": True, "mode": mode}


@app.get("/admin/recent")
async def admin_recent(n: int = 20, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin(authorization)
    return {"items": audit_logger.recent(n)}


@app.post("/admin/replay")
async def admin_replay(request: Request, authorization: str | None = Header(default=None)):
    require_admin(authorization)
    body = await read_json_body(request)
    # Safe default: replay is dry-run unless explicit allow_real=true.
    allow_real = bool(body.pop("allow_real", False))
    n = normalize_responses_request(body)
    if not allow_real:
        return JSONResponse(dry_run_response(n, mode="dry-run"))
    result = await run_and_record(n, "verify")
    return JSONResponse(response_json(result, n))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail, "type": "local_proxy_error"}})

