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
    from app import upstream_client
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
    monkeypatch.setattr(upstream_client.httpx, "AsyncClient", DummyClient)

    async def run():
        try:
            await upstream_client.normal_forward_json({"model": "gpt-test", "input": "hello"})
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


def test_emit_value_schema_embeds_typed_tool_oneof():
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload
    from app.tool_bridge import parse_function_arguments

    tools = [
        {
            "type": "function",
            "name": "shell_command",
            "description": "run shell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout_ms": {"type": "integer"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
        {
            "type": "namespace",
            "name": "codex_app",
            "tools": [
                {
                    "type": "function",
                    "name": "read_thread_terminal",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                }
            ],
        },
        {"type": "custom", "name": "apply_patch", "description": "patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}},
    ]

    n = normalize_responses_request({"model": "gpt-5.5", "input": "list files", "tools": tools})
    payload = build_bill015_payload(n)
    item_schema = payload["tools"][0]["parameters"]["properties"]["tool_calls"]["items"]
    variants = item_schema["oneOf"]

    shell = next(v for v in variants if v["properties"]["name"]["enum"] == ["shell_command"])
    codex = next(v for v in variants if v["properties"]["namespace"]["enum"] == ["codex_app"])
    patch = next(v for v in variants if v["properties"]["name"]["enum"] == ["apply_patch"])

    assert shell["properties"]["arguments"]["properties"]["command"]["type"] == "string"
    assert "timeout_ms" in shell["properties"]["arguments"]["required"]
    assert shell["properties"]["arguments"]["properties"]["timeout_ms"]["type"] == ["integer", "null"]
    assert codex["properties"]["name"]["enum"] == ["read_thread_terminal"]
    assert patch["properties"]["type"]["enum"] == ["custom"]

    _, _, _, mode, calls = parse_function_arguments(
        '{"mode":"tool_call","answer":"","tool_calls":[{"type":"function","namespace":"","name":"shell_command","arguments":{"command":"pwd","timeout_ms":10000},"input":""}]}',
        tool_registry=n.tool_registry,
    )
    assert mode == "tool_call"
    assert calls[0].arguments == '{"command": "pwd", "timeout_ms": 10000}'


def test_latest_user_request_stays_first_after_tool_feedback():
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "旧问题：解释 bill015_local_proxy"}]},
                {"type": "function_call", "name": "shell_command", "call_id": "call_1", "arguments": "{\"command\":\"pwd\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "Exit code: 0\nOutput:\nS:\\hack\\packyapi.com"},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "新问题：修复第二句话回答旧问题"}]},
            ],
        }
    )

    content = build_bill015_payload(n)["input"][1]["content"]

    assert "Recent local Codex tool results" in content
    assert "Current user request:\n新问题：修复第二句话回答旧问题" in content
    assert content.index("新问题：修复第二句话回答旧问题") < content.index("旧问题：解释 bill015_local_proxy")


def test_strict_zero_blocks_auto_passthrough(monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app
    from app.state import runtime_state

    monkeypatch.setattr(settings, "mode", "exploit")
    monkeypatch.setattr(settings, "strict_zero", True)
    runtime_state.current_mode_override = None

    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}],
            "stream": False,
        },
    )

    assert response.status_code == 422
    assert "strict_zero blocked auto-passthrough" in response.text


def test_config_schema_reports_unknown_fields():
    from app.config_schema import validate_local_config

    warnings = validate_local_config({"limits": {"max_concurrency": 2, "old_unused_limit": 1}})

    assert warnings
    assert "limits.old_unused_limit" in warnings[0]



def test_upstream_html_error_is_sanitized():
    from app.upstream_errors import sanitize_upstream_error_detail

    html = "<!DOCTYPE html><html><head><title>packyapi.com | 520: Web server is returning an unknown error</title></head><body>" + ("x" * 5000) + "</body></html>"
    detail = sanitize_upstream_error_detail(520, html, content_type="text/html; charset=utf-8")

    assert detail["upstream_status"] == 520
    assert "520" in detail["message"]
    assert "packyapi.com" in detail["message"]
    assert "<!DOCTYPE" not in detail["message"]
    assert "<html" not in detail["body_preview"].lower()
    assert detail["body_chars"] == len(html)
    assert len(detail["body_sha256_16"]) == 16
    assert "body" not in detail


def test_response_failed_event_sanitizes_http_exception_detail():
    from app.models import NormalizedRequest
    from app.response_events import responses_sse_generator
    from app.sse import parse_sse_lines
    from app.upstream_errors import sanitize_upstream_error_detail

    async def fail():
        html = "<!DOCTYPE html><html><head><title>packyapi.com | 520: Web server is returning an unknown error</title></head></html>"
        raise HTTPException(status_code=502, detail=sanitize_upstream_error_detail(520, html, content_type="text/html"))

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="hello", want_stream=True, client_api="responses", is_primary_path=True)

    async def run():
        chunks = []
        async for chunk in responses_sse_generator(fail(), n, "resp_local_htmlerr"):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    text = asyncio.run(run())
    events = list(parse_sse_lines(text.splitlines(True)))
    objs = [ev.json for ev in events if ev.json]

    assert [obj["type"] for obj in objs][-2:] == ["response.failed", "error"]
    assert events[-1].data == "[DONE]"
    assert "<!DOCTYPE" not in text
    assert "<html" not in text.lower()
    assert "packyapi.com | 520" in text


def test_normal_forward_stream_non_200_emits_response_failed_done(monkeypatch):
    from app import upstream_client
    from app.config import settings
    from app.sse import parse_sse_lines

    html = "<!DOCTYPE html><html><head><title>packyapi.com | 520: Web server is returning an unknown error</title></head><body>cf</body></html>"

    class DummyStreamResponse:
        status_code = 520
        headers = {"content-type": "text/html; charset=utf-8"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return html.encode("utf-8")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return DummyStreamResponse()

    monkeypatch.setattr(settings, "upstream_api_key_file_value", "sk-test")
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(upstream_client.httpx, "AsyncClient", DummyClient)

    async def run():
        chunks = []
        async for chunk in upstream_client.normal_forward_stream({"model": "gpt-test", "input": "hello", "stream": True}):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    text = asyncio.run(run())
    events = list(parse_sse_lines(text.splitlines(True)))
    objs = [ev.json for ev in events if ev.json]

    assert [obj["type"] for obj in objs] == ["response.failed", "error"]
    assert events[-1].data == "[DONE]"
    assert objs[0]["response"]["error"]["upstream_status"] == 520
    assert "<!DOCTYPE" not in text
    assert "<html" not in text.lower()


def test_execute_bill015_retries_pre_stream_http_500(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest

    class DummyStreamResponse:
        headers = {"content-type": "text/event-stream"}

        def __init__(self, status_code: int):
            self.status_code = status_code

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return b'{"error":{"message":"upstream error: do request failed","type":"new_api_error","code":"do_request_failed"}}'

        async def aiter_lines(self):
            lines = [
                'event: response.output_item.added',
                'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
                '',
                'event: response.function_call_arguments.done',
                'data: {"type":"response.function_call_arguments.done","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"OK\\",\\"tool_calls\\":[]}"}',
                '',
            ]
            for line in lines:
                yield line

        async def aclose(self):
            return None

    class DummyClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            DummyClient.calls += 1
            return DummyStreamResponse(500 if DummyClient.calls == 1 else 200)

    monkeypatch.setattr(settings, "upstream_api_key_file_value", "sk-test")
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "upstream_retries", 2)
    monkeypatch.setattr(settings, "upstream_retry_backoff_ms", 0)
    monkeypatch.setattr(settings, "strict_zero", False)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="Return OK", want_stream=False, client_api="responses", is_primary_path=True)

    result = asyncio.run(upstream.execute_bill015(n, "exploit"))

    assert DummyClient.calls == 2
    assert result.retry_count == 1
    assert result.retry_reasons == ["http_500"]
    assert result.answer == "OK"
    assert result.args_done_seen is True
    assert result.aborted is True
