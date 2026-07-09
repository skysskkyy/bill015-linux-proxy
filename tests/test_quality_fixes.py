from __future__ import annotations

import asyncio

from fastapi import HTTPException


def test_repaired_function_arguments_are_marked_malformed():
    from app.tool_bridge import parse_function_arguments

    answer, malformed, repaired, mode, calls = parse_function_arguments(
        '{"mode":"answer","answer":"OK","tool_calls":[]} trailing noise'
    )

    assert answer == "OK"
    assert mode == "answer"
    assert calls == []
    assert malformed is True
    assert repaired is True


def test_normal_forward_json_preserves_upstream_error_status(monkeypatch):
    from app import upstream
    from app.config import settings

    class DummyResponse:
        status_code = 429
        text = '{"error":{"message":"rate limited"}}'

        def json(self):
            return {"error": {"message": "rate limited"}}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(settings, "upstream_api_key_file_value", "sk-test")
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)

    async def run():
        try:
            await upstream.normal_forward_json({"model": "gpt-test", "input": "hello"})
        except HTTPException as exc:
            return exc
        raise AssertionError("normal_forward_json did not raise for upstream 429")

    exc = asyncio.run(run())
    assert exc.status_code == 429
    assert exc.detail["upstream_status"] == 429
    assert exc.detail["body"]["error"]["message"] == "rate limited"


def test_redact_covers_common_secret_key_names():
    from app.audit import redact

    value = redact(
        {
            "access_token": "abc123456789012345",
            "refresh_token": "def123456789012345",
            "admin_token": "ghi123456789012345",
            "db_password": "super-secret-password",
            "nested": {"client_secret": "secret-value"},
            "input_tokens": 123,
        }
    )

    assert value["access_token"] == "<redacted>"
    assert value["refresh_token"] == "<redacted>"
    assert value["admin_token"] == "<redacted>"
    assert value["db_password"] == "<redacted>"
    assert value["nested"]["client_secret"] == "<redacted>"
    assert value["input_tokens"] == 123


def test_tool_bridge_strict_mode_drops_unregistered_tools(monkeypatch):
    from app.config import settings
    from app.tool_bridge import resolve_bridge_tool_call

    monkeypatch.setattr(settings, "tool_bridge_allow_unknown_tools", False)

    assert resolve_bridge_tool_call({"name": "not_registered", "arguments": "{}"}, {}) is None

    registered = resolve_bridge_tool_call(
        {"name": "shell_command", "arguments": '{"command":"pwd"}'},
        {"shell_command": {"call_type": "function", "output_name": "shell_command", "raw_type": "function"}},
    )
    assert registered is not None
    assert registered.name == "shell_command"
