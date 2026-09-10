from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.ingest.catalog import build_catalog
from app.protocol.models import Turn
from app.upstream.errors import (
    CYBER_POLICY_ERROR_MESSAGE,
    RATE_LIMIT_RETRY_DELAY_SECONDS,
    is_cyber_policy_rotation_error,
    is_rate_limit_exceeded,
    is_server_error,
    key_rotation_error_reason,
    retryable_upstream_error_reason,
)
from app.upstream.execute import execute_turn
from app.upstream.keys import ApiKeyPool


def test_cyber_policy_matcher_requires_logged_fields():
    ok = {"error": {"code": "cyber_policy", "message": CYBER_POLICY_ERROR_MESSAGE}}
    assert is_cyber_policy_rotation_error(ok)
    assert key_rotation_error_reason(ok) == "cyber_policy"
    nope = {"error": {"code": "cyber_policy", "message": "please rephrase"}}
    assert not is_cyber_policy_rotation_error(nope)
    assert key_rotation_error_reason(nope) is None


def test_rate_limit_exceeded_matches_sse_capacity_event():
    sse = {
        "type": "error",
        "error": {
            "type": "service_unavailable_error",
            "code": "rate_limit_exceeded",
            "message": "Temporary capacity limit. Please try again in 2s.",
            "param": None,
        },
        "sequence_number": 2,
    }
    assert is_rate_limit_exceeded(sse)
    assert RATE_LIMIT_RETRY_DELAY_SECONDS == 2.0
    assert not is_rate_limit_exceeded({"error": {"code": "cyber_policy", "message": CYBER_POLICY_ERROR_MESSAGE}})
    assert not is_rate_limit_exceeded({"error": {"code": "concurrent_request_limit_exceeded", "message": "Too many concurrent requests."}})
    assert is_rate_limit_exceeded(b'{"error":{"code":"rate_limit_exceeded","message":"Temporary capacity limit. Please try again in 2s."}}')


def test_server_error_matches_openai_retryable_sse():
    sse = {
        "type": "error",
        "error": {
            "type": "server_error",
            "code": "server_error",
            "message": "An error occurred while processing your request. You can retry your request.",
            "param": None,
        },
        "sequence_number": 2,
    }
    assert is_server_error(sse)
    assert retryable_upstream_error_reason(sse) == "server_error"
    assert not is_server_error({"error": {"code": "rate_limit_exceeded", "message": "Temporary capacity limit."}})


@pytest.mark.parametrize(
    ("event", "reason"),
    [
        (
            {
                "type": "error",
                "error": {
                    "type": "service_unavailable_error",
                    "code": "rate_limit_exceeded",
                    "message": "Temporary capacity limit. Please try again in 2s.",
                    "param": None,
                },
                "sequence_number": 2,
            },
            "rate_limit_exceeded",
        ),
        (
            {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "code": "server_error",
                    "message": "An error occurred while processing your request. You can retry your request.",
                    "param": None,
                },
                "sequence_number": 2,
            },
            "server_error",
        ),
    ],
)
def test_execute_retries_retryable_sse_instead_of_raising(monkeypatch, event, reason):
    calls = {"n": 0, "slept": []}
    success = (
        "event: response.created\n"
        'data: {"type":"response.created","response":{"id":"resp_ok"}}\n'
        "\n"
        "event: response.output_item.added\n"
        'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value","call_id":"c1"}}\n'
        "\n"
        "event: response.function_call_arguments.done\n"
        'data: {"type":"response.function_call_arguments.done","call_id":"c1","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"ok\\",\\"tool_calls\\":[]}"}\n'
        "\n"
    )

    async def fake_sleep(seconds):
        calls["slept"].append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            body = "event: error\ndata: " + json.dumps(event) + "\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, text=success, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("app.upstream.execute.httpx.AsyncClient", fake_client)
    monkeypatch.setattr("app.upstream.execute.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("app.upstream.execute.upstream_key_pool", lambda cfg=None: ApiKeyPool(["sk-test"]))
    monkeypatch.setattr("app.upstream.execute.settings.tool_bridge_local_web_research_preflight", False)

    turn = Turn(
        model="gpt-6-astra",
        want_stream=True,
        client_api="responses",
        items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        current_user="hi",
        catalog=build_catalog([], [], query="hi"),
    )
    result = asyncio.run(execute_turn(turn, "exploit"))
    assert calls["n"] == 2
    assert calls["slept"] == [2.0]
    assert result.retry_reasons == [reason]
    assert result.answer == "ok"


def test_key_pool_rotates_and_cools_failed_key():
    pool = ApiKeyPool(["aaa", "bbb", "ccc"])
    first = pool.current()
    assert first is not None and first.key == "aaa"
    nxt = pool.rotate_after_failure("aaa", {"aaa"}, cooldown_seconds=600)
    assert nxt is not None and nxt.key == "bbb"
    again = pool.current()
    assert again is not None and again.key == "bbb"
