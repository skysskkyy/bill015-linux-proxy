from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

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


def test_unknown_tools_disabled_by_default_without_local_config(tmp_path):
    env = os.environ.copy()
    env["BILL015_CONFIG_PATH"] = str(tmp_path / "missing-config.json")
    env.pop("BILL015_TOOL_BRIDGE_ALLOW_UNKNOWN_TOOLS", None)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from app.config import settings; "
                "print(settings.tool_bridge_allow_unknown_tools, settings.max_output_tokens, "
                "settings.compaction_max_output_tokens, settings.max_answer_chars, "
                "settings.upstream_timeout_seconds, settings.args_done_timeout_ms, "
                "settings.upstream_idle_timeout_ms, settings.upstream_retries)"
            ),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "False 8192 8192 65536 300.0 300000 180000 0"


def test_strict_unknown_tool_schema_allows_only_tool_search_discovery_without_registry(monkeypatch):
    from app.config import settings
    from app.payloads import build_emit_value_schema
    from app.tool_bridge import parse_function_arguments

    monkeypatch.setattr(settings, "tool_bridge_allow_unknown_tools", False)
    schema = build_emit_value_schema(settings, {})
    params = schema["parameters"]["properties"]

    assert params["mode"]["enum"] == ["answer", "tool_call"]
    assert "tool_search" in params["tool_calls"]["description"]

    answer, _, _, mode, calls = parse_function_arguments(
        '{"mode":"tool_call","answer":"","tool_calls":[{"type":"function","namespace":"","name":"not_registered","arguments":{},"input":""}]}',
        tool_registry={},
    )
    assert mode == "answer"
    assert calls == []
    assert "no valid tool_calls" in answer

    _, _, _, search_mode, search_calls = parse_function_arguments(
        json.dumps({
            "mode": "tool_call",
            "answer": "",
            "tool_calls": [
                {
                    "type": "tool_search",
                    "namespace": "",
                    "name": "tool_search",
                    "arguments": json.dumps({"query": "PowerShell terminal tools", "limit": 8}),
                    "input": "",
                }
            ],
        }),
        tool_registry={},
    )
    assert search_mode == "tool_call"
    assert search_calls[0].call_type == "tool_search"


def test_emit_value_schema_is_upstream_compatible_and_tools_are_validated_locally():
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

    assert "oneOf" not in json.dumps(payload["tools"], ensure_ascii=False)
    assert item_schema["properties"]["name"]["type"] == "string"
    assert item_schema["properties"]["arguments"]["type"] == "string"
    assert "shell_command" in payload["instructions"]
    assert "codex_app.read_thread_terminal" in payload["instructions"]
    assert "apply_patch" in payload["instructions"]
    assert "prefer apply_patch" in payload["instructions"]
    assert "Do not call shell_command, node_repl, or PowerShell just to write" in payload["instructions"]

    _, _, _, mode, calls = parse_function_arguments(
        '{"mode":"tool_call","answer":"","tool_calls":[{"type":"function","namespace":"","name":"shell_command","arguments":{"command":"pwd","timeout_ms":10000},"input":""}]}',
        tool_registry=n.tool_registry,
    )
    assert mode == "tool_call"
    assert calls[0].arguments == '{"command": "pwd", "timeout_ms": 10000}'


def test_loop_guard_stops_repeated_successful_tool_call():
    from app.models import Bill015Result
    from app.normalization import normalize_responses_request
    from app.sse import parse_sse_lines
    from app.upstream import collect_bill015_result_from_events

    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "input": [
                {"type": "function_call", "name": "shell_command", "call_id": "call_done", "arguments": "{\"command\":\"pwd\"}"},
                {"type": "function_call_output", "call_id": "call_done", "output": "Exit code: 0\nOutput:\nS:\\hack\\packyapi.com"},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell_command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False},
                }
            ],
        }
    )
    args = json.dumps(
        {
            "mode": "tool_call",
            "answer": "",
            "tool_calls": [{"type": "function", "name": "shell_command", "arguments": "{\"command\":\"pwd\"}", "input": ""}],
        }
    )
    data = json.dumps({"type": "response.function_call_arguments.done", "arguments": args}, ensure_ascii=False)
    events = parse_sse_lines(
        [
            "event: response.function_call_arguments.done\n",
            f"data: {data}\n",
            "\n",
        ]
    )

    result = collect_bill015_result_from_events(events, n)

    assert isinstance(result, Bill015Result)
    assert result.bridge_mode == "answer"
    assert result.tool_calls == []
    assert "loop guard" in result.answer


def test_tool_search_auto_expansion_is_capped_to_one_extra_search():
    from app.tool_bridge import parse_function_arguments

    _, _, _, mode, calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "",
                "tool_calls": [
                    {
                        "type": "tool_search",
                        "namespace": "",
                        "name": "tool_search",
                        "arguments": json.dumps({"query": "Playwright browser cookies DOM network tools", "limit": 8}),
                        "input": "",
                    }
                ],
            }
        ),
        tool_registry={"tool_search": {"call_type": "tool_search", "output_name": "tool_search", "raw_type": "tool_search"}},
    )

    assert mode == "tool_call"
    assert len([c for c in calls if c.call_type == "tool_search"]) <= 2


def test_payload_output_budgets_and_compaction_budget(monkeypatch):
    from app.config import settings
    from app.models import NormalizedRequest
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "max_output_tokens", 8192)
    monkeypatch.setattr(settings, "compaction_max_output_tokens", 4096)

    n = normalize_responses_request({"model": "gpt-5.5", "input": "write a long patch", "max_output_tokens": 20000})
    assert build_bill015_payload(n)["max_output_tokens"] == 8192

    comp = NormalizedRequest(
        model="gpt-5.5",
        instructions="",
        user_input="compact",
        want_stream=True,
        client_api="responses",
        is_primary_path=True,
        is_compaction=True,
        raw_input="compact",
        max_output_tokens=20000,
    )
    assert build_bill015_payload(comp)["max_output_tokens"] == 4096


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

    payload = build_bill015_payload(n)
    state_content = payload["input"][0]["content"]
    current_content = payload["input"][-1]["content"]

    assert "Recent local Codex tool results" in state_content
    assert "旧问题：解释 bill015_local_proxy" in state_content
    assert "Current user request:\n新问题：修复第二句话回答旧问题" in current_content
    assert payload["input"][-1]["role"] == "user"
    assert state_content.count("新问题：修复第二句话回答旧问题") == 0
    assert current_content.count("新问题：修复第二句话回答旧问题") == 1


def test_system_developer_context_goes_to_instructions_and_current_user_is_last():
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "instructions": "top-level root rule",
            "input": [
                {"type": "message", "role": "system", "content": "system rule A"},
                {"type": "message", "role": "developer", "content": "developer rule B"},
                {"type": "message", "role": "user", "content": "old request"},
                {"type": "message", "role": "system", "content": "system rule C"},
                {"type": "message", "role": "user", "content": "current request"},
            ],
        }
    )
    payload = build_bill015_payload(n)

    upstream_instructions = payload["instructions"]
    assert "top-level root rule" in upstream_instructions
    assert "system rule A" in upstream_instructions
    assert "developer rule B" in upstream_instructions
    assert "system rule C" in upstream_instructions
    assert upstream_instructions.index("system rule A") < upstream_instructions.index("developer rule B") < upstream_instructions.index("system rule C")

    dumped_input = json.dumps(payload["input"], ensure_ascii=False)
    assert "system rule A" not in dumped_input
    assert "developer rule B" not in dumped_input
    assert payload["input"][-1] == {"role": "user", "content": "Current user request:\ncurrent request"}


def test_latest_parallel_tool_batch_keeps_more_than_three_outputs():
    from app.normalization import normalize_responses_request

    input_items = [{"type": "message", "role": "user", "content": "run batch"}]
    for idx in range(5):
        input_items.append(
            {
                "type": "function_call",
                "name": "shell_command",
                "call_id": f"call_{idx}",
                "arguments": f'{{"command":"echo marker-{idx}"}}',
            }
        )
    for idx in range(5):
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": f"call_{idx}",
                "output": f"Exit code: 0\nOutput:\nmarker-{idx}",
            }
        )

    n = normalize_responses_request({"model": "gpt-5.5", "input": input_items})

    assert len(n.tool_history.latest_outputs) == 5
    for idx in range(5):
        assert f"marker-{idx}" in n.latest_tool_summary


def test_budgeted_transcript_uses_current_and_tail_history_not_raw_char_prefix():
    from app.normalization import normalize_responses_request

    old = "OLD_HISTORY_START " + ("老历史 " * 20_000)
    current = "CURRENT_UNIQUE_REQUEST"
    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "max_output_tokens": 4096,
            "input": [
                {"type": "message", "role": "user", "content": old},
                {"type": "message", "role": "assistant", "content": "middle"},
                {"type": "message", "role": "user", "content": current},
            ],
        }
    )
    content = n.user_input

    assert "current user message moved above" in content
    assert current not in content
    # The very old raw prefix should not dominate the token-budgeted transcript.
    assert "OLD_HISTORY_START" not in content


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
