from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.ops.capacity import CapacityLimiter
from app.ops.state import RuntimeState
from app.upstream.keys import ApiKeyPool
from app.upstream.passthrough import CompactionPassthroughResponse


class Chunks(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.fixture
def upstream(monkeypatch):
    state = RuntimeState()
    limiter = CapacityLimiter(1, 0, 100, state)
    fixture = SimpleNamespace(requests=[], records=[], state=state, limiter=limiter, handler=None)
    real_client = httpx.AsyncClient

    async def handle(request):
        fixture.requests.append(request)
        response = fixture.handler(request)
        if asyncio.iscoroutine(response):
            response = await response
        return response

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("app.upstream.passthrough.httpx.AsyncClient", client)
    monkeypatch.setattr("app.upstream.passthrough.upstream_key_pool", lambda cfg: ApiKeyPool(["test-upstream-key"]))
    monkeypatch.setattr("app.upstream.execute.upstream_key_pool", lambda cfg: ApiKeyPool(["test-upstream-key"]))
    monkeypatch.setattr("app.upstream.passthrough.runtime_state", state)
    monkeypatch.setattr("app.main.runtime_state", state)
    monkeypatch.setattr("app.main.capacity_limiter", limiter)
    monkeypatch.setattr("app.upstream.passthrough.audit_logger.write", fixture.records.append)
    monkeypatch.setattr(settings, "mode", "exploit")
    monkeypatch.setattr(settings, "upstream_base_url", "https://upstream.invalid")
    monkeypatch.setattr(settings, "packy_cookie", "")
    monkeypatch.setattr(settings, "block_passthrough", True)
    return fixture


def compaction_body(stream=True):
    return {
        "model": "client-model",
        "stream": stream,
        "instructions": "Preserve the goal and pending work in the summary.",
        "input": [{
            "type": "agent_message", "id": "original-id",
            "content": [{"type": "encrypted_content", "encrypted_content": "opaque-test-content"}],
        }],
        "tools": [{"type": "web_search"}],
        "include": ["reasoning.encrypted_content"],
        "client_metadata": {"x-codex-turn-metadata": json.dumps({"request_kind": "compaction"})},
    }


@pytest.mark.parametrize("marker", ["metadata_string", "metadata_object", "top_level", "header"])
def test_compaction_preserves_request_and_sse(upstream, monkeypatch, marker):
    body = compaction_body()
    headers = {
        "authorization": "Bearer client-secret", "cookie": "client-cookie",
        "content-type": "application/json", "session-id": "session-test",
        "x-openai-internal-codex-responses-lite": "true",
    }
    if marker == "metadata_object":
        body["client_metadata"]["x-codex-turn-metadata"] = {"compaction": {"implementation": "responses"}}
    elif marker == "top_level":
        body.pop("client_metadata")
        body["request_kind"] = "compaction"
    elif marker == "header":
        body.pop("client_metadata")
        headers["x-codex-turn-metadata"] = json.dumps({"request_kind": "compaction"})
    raw_body = json.dumps(body, indent=2).encode()
    wire = (
        b': upstream heartbeat\n\n'
        b'data: {"type":"response.output_text.delta","delta":"summary"}\n\n'
        b'data: {"type":"response.output_item.done","item":{"type":"compaction","encrypted_content":"blob"}}\n\n'
        b'data: {"type":"response.completed","response":{"id":"resp_native","usage":{"output_tokens":42}}}\n\n'
    )
    stream = Chunks(wire[:17], wire[17:])
    upstream.handler = lambda request: httpx.Response(200, stream=stream, headers={
        "content-type": "text/event-stream", "x-request-id": "req-native",
        "connection": "keep-alive, x-hop", "x-hop": "remove-me",
    })
    monkeypatch.setattr(settings, "force_default_model", True)
    monkeypatch.setattr(settings, "default_model", "must-not-replace-model")
    with TestClient(app) as client:
        response = client.post("/v1/responses", content=raw_body, headers=headers)
    assert response.status_code == 200
    assert response.content == wire
    assert response.headers["x-request-id"] == "req-native"
    assert "x-hop" not in response.headers
    assert "connection" not in response.headers
    assert len(upstream.requests) == 1
    sent = upstream.requests[0]
    assert str(sent.url) == "https://upstream.invalid/v1/responses"
    assert sent.content == raw_body
    assert sent.headers["authorization"] == "Bearer test-upstream-key"
    assert "cookie" not in sent.headers
    assert sent.headers["session-id"] == "session-test"
    assert sent.headers["x-openai-internal-codex-responses-lite"] == "true"
    assert stream.closed
    assert upstream.state.active_requests == 0
    assert upstream.state.success_count == 1
    assert upstream.records[-1]["upstream_request_id"] == "req-native"


@pytest.mark.parametrize("status", [200, 400, 429, 503])
def test_compaction_preserves_json_and_http_errors(upstream, status):
    body = compaction_body(stream=False)
    wire = b'{ "output": [{"type":"message","content":[{"type":"output_text","text":"summary"}]}] }' if status == 200 else b'{ "error": {"code":"native_error","message":"original detail"} }'
    stream = Chunks(wire)
    upstream.handler = lambda request: httpx.Response(status, stream=stream, headers={
        "content-type": "application/json", "retry-after": "2", "content-length": str(len(wire)),
    })
    with TestClient(app) as client:
        response = client.post("/v1/responses", json=body)
    assert response.status_code == status
    assert response.content == wire
    assert response.headers["retry-after"] == "2"
    assert json.loads(upstream.requests[0].content) == body
    assert stream.closed
    assert upstream.state.active_requests == 0
    assert upstream.records[-1]["upstream_status"] == status


def test_compaction_sse_error_passes_through(upstream):
    wire = b'data: {"type":"error","error":{"code":"stream_read_error","message":"upstream detail"}}\n\n'
    upstream.handler = lambda request: httpx.Response(200, stream=Chunks(wire), headers={"content-type": "text/event-stream"})
    with TestClient(app) as client:
        response = client.post("/v1/responses", json=compaction_body())
    assert response.content == wire
    assert len(upstream.requests) == 1


def test_explicit_compact_still_uses_compact_endpoint(upstream):
    wire = json.dumps({"output": [{"type": "compaction", "id": "cmp_test", "encrypted_content": "blob"}]}).encode()
    upstream.handler = lambda request: httpx.Response(200, stream=Chunks(wire), headers={"content-type": "application/json"})
    with TestClient(app) as client:
        response = client.post("/v1/responses/compact", json={"model": "gpt-6-astra", "input": "history"})
    assert response.status_code == 200
    assert upstream.requests[0].url.path == "/v1/responses/compact"
    assert response.json()["object"] == "response.compaction"
    assert response.json()["output"][0]["encrypted_content"] == "blob"


def test_compaction_dry_run_does_not_contact_upstream(upstream, monkeypatch):
    monkeypatch.setattr(settings, "mode", "dry-run")
    with TestClient(app) as client:
        response = client.post("/v1/responses", json=compaction_body(stream=False))
    assert response.status_code == 200
    assert upstream.requests == []


def test_ordinary_responses_still_enforce_passthrough_guard(upstream):
    body = compaction_body(stream=False)
    body.pop("client_metadata")
    with TestClient(app) as client:
        response = client.post("/v1/responses", json=body)
    assert response.status_code == 422
    assert upstream.requests == []


def test_compaction_respects_circuit_breaker(upstream, monkeypatch):
    monkeypatch.setattr(settings, "circuit_failures", 1)
    upstream.state.mark_error("synthetic failure")
    with TestClient(app) as client:
        response = client.post("/v1/responses", json=compaction_body())
    assert response.status_code == 503
    assert upstream.requests == []


def test_compaction_queue_full_does_not_contact_upstream(upstream):
    async def run():
        lease = await upstream.limiter.acquire()
        sent = []
        async def send(message):
            sent.append(message)
        async def receive():
            return {"type": "http.disconnect"}
        response = CompactionPassthroughResponse(compaction_body(), mode="exploit", limiter=upstream.limiter)
        try:
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            assert sent[0]["status"] == 429
        finally:
            await lease.release()
    asyncio.run(run())
    assert upstream.requests == []
    assert upstream.state.active_requests == 0


@pytest.mark.parametrize("ending", ["read_error", "cancel", "timeout"])
def test_compaction_stream_lifetime_releases_resources(upstream, monkeypatch, ending):
    class InterruptedStream(Chunks):
        async def __aiter__(self):
            yield b': first chunk\n\n'
            if ending == "read_error":
                raise httpx.ReadError("synthetic read error")
            await asyncio.Event().wait()
    stream = InterruptedStream()
    upstream.handler = lambda request: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    if ending == "timeout":
        monkeypatch.setattr(settings, "request_total_timeout_ms", 20)
    async def run():
        sent = []
        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body":
                assert upstream.state.active_requests == 1
                if ending == "cancel":
                    raise asyncio.CancelledError
        async def receive():
            await asyncio.Event().wait()
        response = CompactionPassthroughResponse(compaction_body(), mode="exploit", limiter=upstream.limiter)
        error = {"read_error": httpx.ReadError, "cancel": asyncio.CancelledError, "timeout": TimeoutError}[ending]
        with pytest.raises(error):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert not any(m["type"] == "http.response.body" and not m.get("more_body") for m in sent)
        assert stream.closed
        assert upstream.state.active_requests == 0
        lease = await upstream.limiter.acquire()
        await lease.release()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["connect", "timeout"])
def test_compaction_failure_before_headers_returns_http_error(upstream, monkeypatch, failure):
    async def handler(request):
        if failure == "connect":
            raise httpx.ConnectError("synthetic connection failure")
        await asyncio.Event().wait()
    upstream.handler = handler
    monkeypatch.setattr(settings, "request_total_timeout_ms", 20)
    with TestClient(app) as client:
        response = client.post("/v1/responses", json=compaction_body())
    assert response.status_code == (502 if failure == "connect" else 504)
    assert upstream.state.active_requests == 0


def test_compaction_client_disconnect_closes_upstream(upstream):
    class WaitingStream(Chunks):
        async def __aiter__(self):
            yield b': first chunk\n\n'
            await asyncio.Event().wait()
    stream = WaitingStream()
    upstream.handler = lambda request: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    async def run():
        first_chunk = asyncio.Event()
        async def send(message):
            if message["type"] == "http.response.body":
                first_chunk.set()
        async def receive():
            await first_chunk.wait()
            return {"type": "http.disconnect"}
        response = CompactionPassthroughResponse(compaction_body(), mode="exploit", limiter=upstream.limiter)
        async with asyncio.timeout(1):
            await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
    asyncio.run(run())
    assert stream.closed
    assert upstream.state.active_requests == 0
    assert upstream.records[-1]["client_disconnected"] is True
