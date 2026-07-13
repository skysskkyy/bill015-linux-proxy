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
                "settings.upstream_idle_timeout_ms, settings.upstream_retries, "
                "settings.reasoning_effort, settings.reasoning_summary, "
                "settings.tool_bridge_auto_expand_search)"
            ),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "False 8192 8192 65536 300.0 300000 180000 0   False"


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
    assert mode == "tool_call"
    assert answer == ""
    assert calls[0].call_type == "tool_search"
    assert "not_registered" in calls[0].arguments

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


def test_invalid_or_display_named_tool_calls_fall_back_to_discovery_not_user_error():
    from app.tool_bridge import parse_function_arguments

    answer, _, _, mode, calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "我需要读取当前 Chrome 页面。",
                "tool_calls": [
                    {
                        "type": "function",
                        "tool_name": "Chrome Integration",
                        "parameters": {"action": "extract Level 0 settings text"},
                        "input": "",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        tool_registry={"tool_search": {"call_type": "tool_search", "output_name": "tool_search", "raw_type": "tool_search"}},
    )

    assert mode == "tool_call"
    assert answer == "我需要读取当前 Chrome 页面。"
    assert len(calls) == 1
    assert calls[0].call_type == "tool_search"
    assert "Chrome Integration" in calls[0].arguments
    assert "Level 0" in calls[0].arguments

    blank_answer, _, _, blank_mode, blank_calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "",
                "tool_calls": [{"type": "function", "tool_name": "Chrome Integration", "parameters": {"action": "extract text"}}],
            }
        ),
        tool_registry={"tool_search": {"call_type": "tool_search", "output_name": "tool_search", "raw_type": "tool_search"}},
    )
    assert blank_mode == "tool_call"
    assert blank_answer == ""
    assert blank_calls[0].call_type == "tool_search"


def test_nested_native_call_shape_is_recovered():
    from app.tool_bridge import parse_function_arguments

    registry = {
        "mcp__chrome.extract_text": {
            "call_type": "function",
            "output_name": "extract_text",
            "namespace": "mcp__chrome",
            "raw_type": "function",
        }
    }
    _, _, _, mode, calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "",
                "tool_calls": [{"native_call": {"namespace": "mcp__chrome", "name": "extract_text"}, "parameters": {"level": 0}}],
            }
        ),
        tool_registry=registry,
    )

    assert mode == "tool_call"
    assert calls[0].namespace == "mcp__chrome"
    assert calls[0].name == "extract_text"
    assert calls[0].arguments == '{"level": 0}'


def test_emit_value_schema_is_upstream_compatible_and_tools_are_validated_locally(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload
    from app.tool_bridge import parse_function_arguments

    monkeypatch.setattr(settings, "bridge_strategy", "emit_value")
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
    assert "enum" in item_schema["properties"]["name"]
    assert "shell_command" in item_schema["properties"]["name"]["enum"]
    assert "read_thread_terminal" in item_schema["properties"]["name"]["enum"]
    assert "codex_app.read_thread_terminal" in item_schema["properties"]["name"]["enum"]
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


def test_native_tool_first_payload_exposes_real_tools_and_final_answer(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "bridge_strategy", "native_tool_first")
    monkeypatch.setattr(settings, "final_answer_tool_name", "submit_final_answer")
    monkeypatch.setattr(settings, "native_tool_choice", "required")
    tools = [
        {
            "type": "function",
            "name": "shell_command",
            "description": "run shell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
        {"type": "custom", "name": "apply_patch", "description": "patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}},
    ]

    n = normalize_responses_request({"model": "gpt-5.5", "input": "list files", "tools": tools, "parallel_tool_calls": True})
    payload = build_bill015_payload(n)

    tool_names = [tool.get("name") for tool in payload["tools"]]
    assert "shell_command" in tool_names
    assert "apply_patch" in tool_names
    assert "submit_final_answer" in tool_names
    assert payload["tools"][0]["name"] == "shell_command"
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert "emit_value" not in tool_names
    assert "submit_final_answer" in payload["instructions"]
    assert "Do not wrap native tool calls" in payload["instructions"]


def test_local_proxy_noise_messages_are_not_replayed_to_upstream():
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "input": [
                {"role": "assistant", "content": [{"type": "output_text", "text": "I need to resolve the right local tool first."}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "[local proxy loop guard] stopped"}]},
                {"role": "user", "content": "continue"},
            ],
        }
    )
    payload = build_bill015_payload(n)
    text = json.dumps(payload["input"], ensure_ascii=False)

    assert "I need to resolve the right local tool first" not in text
    assert "[local proxy loop guard]" not in text
    assert "continue" in text


def test_loop_guard_does_not_abort_repeated_successful_tool_call():
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
    assert result.bridge_mode == "tool_call"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "shell_command"
    assert "loop guard" not in result.answer
    assert any("allowed_repeated_success_to_avoid_abort" in reason for reason in result.retry_reasons)


def test_loop_guard_filters_repeated_success_from_mixed_batch():
    from app.models import Bill015Result, BridgeToolCall
    from app.normalization import normalize_responses_request
    from app.upstream import apply_tool_loop_guard

    n = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "tool_search_call", "call_id": "call_search", "execution": "client", "arguments": {"query": "browser tools", "limit": 8}},
                {"type": "tool_search_output", "call_id": "call_search", "execution": "client", "status": "completed", "tools": []},
            ],
        }
    )
    result = Bill015Result(
        local_request_id="resp_local_mixed_loop",
        bridge_mode="tool_call",
        tool_calls=[
            BridgeToolCall(id="call_repeat", name="tool_search", arguments='{"query":"browser tools","limit":8}', call_type="tool_search"),
            BridgeToolCall(id="call_next", name="js", namespace="mcp__node_repl", arguments='{"code":"1+1"}'),
        ],
    )

    apply_tool_loop_guard(result, n)

    assert result.bridge_mode == "tool_call"
    assert [(call.namespace, call.name) for call in result.tool_calls] == [("mcp__node_repl", "js")]
    assert any("filtered_repeated_success:tool_search" in reason for reason in result.retry_reasons)


def test_loop_guard_allows_corrected_retry_after_failed_tool_name():
    from app.models import Bill015Result, BridgeToolCall
    from app.normalization import normalize_responses_request
    from app.upstream import apply_tool_loop_guard

    n = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "function_call", "name": "js", "namespace": "mcp__node_repl", "call_id": "call_bad", "arguments": "{\"code\":\"bad()\"}"},
                {"type": "function_call_output", "call_id": "call_bad", "output": "Error: bad is not defined"},
            ],
        }
    )
    result = Bill015Result(
        local_request_id="resp_local_retry",
        bridge_mode="tool_call",
        tool_calls=[BridgeToolCall(id="call_good", name="js", namespace="mcp__node_repl", arguments="{\"code\":\"fixed()\"}")],
    )

    apply_tool_loop_guard(result, n)

    assert result.bridge_mode == "tool_call"
    assert result.tool_calls


def test_loop_guard_stops_exact_failed_tool_replay():
    from app.models import Bill015Result, BridgeToolCall
    from app.normalization import normalize_responses_request
    from app.upstream import apply_tool_loop_guard

    n = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "function_call", "name": "js", "namespace": "mcp__node_repl", "call_id": "call_bad", "arguments": "{\"code\":\"bad()\"}"},
                {"type": "function_call_output", "call_id": "call_bad", "output": "Error: bad is not defined"},
            ],
        }
    )
    result = Bill015Result(
        local_request_id="resp_local_retry",
        bridge_mode="tool_call",
        tool_calls=[BridgeToolCall(id="call_bad_again", name="js", namespace="mcp__node_repl", arguments="{\"code\":\"bad()\"}")],
    )

    apply_tool_loop_guard(result, n)

    assert result.bridge_mode == "tool_call"
    assert result.tool_calls
    assert any("repeated_failed_tool_allowed" in reason for reason in result.retry_reasons)


def test_tool_search_auto_expansion_is_disabled_by_default():
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
    assert len([c for c in calls if c.call_type == "tool_search"]) == 1


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
    assert [item.get("type", "message") for item in payload["input"]] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert payload["input"][0]["content"][0]["text"] == "旧问题：解释 bill015_local_proxy"
    assert payload["input"][1]["name"] == "shell_command"
    assert payload["input"][2]["output"].startswith("Exit code: 0")
    assert payload["input"][-1]["role"] == "user"
    assert payload["input"][-1]["content"][0]["text"] == "新问题：修复第二句话回答旧问题"


def test_system_developer_context_stays_typed_in_normal_input():
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
    dumped_input = json.dumps(payload["input"], ensure_ascii=False)
    assert "system rule A" in dumped_input
    assert "developer rule B" in dumped_input
    assert "system rule C" in dumped_input
    assert payload["input"][0] == {"type": "message", "role": "system", "content": "system rule A"}
    assert payload["input"][1] == {"type": "message", "role": "developer", "content": "developer rule B"}
    assert payload["input"][-1] == {"type": "message", "role": "user", "content": "current request"}


def test_large_developer_context_is_not_budget_clipped_in_normal_turn():
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    marker = "UNIQUE_DEV_CONTEXT_TAIL"
    large_developer = "developer prefix " + ("重要规则 " * 20_000) + marker
    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "instructions": "top-level root rule",
            "input": [
                {"type": "message", "role": "developer", "content": large_developer},
                {"type": "message", "role": "user", "content": "current request"},
            ],
        }
    )
    payload = build_bill015_payload(n)

    dumped_input = json.dumps(payload["input"], ensure_ascii=False)
    assert marker in dumped_input
    assert "token-budget clipped" not in dumped_input
    assert marker not in payload["instructions"]


def test_bill015_payload_preserves_native_input_and_request_controls():
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect"}]},
                {"type": "function_call", "name": "shell_command", "call_id": "call_1", "arguments": "{\"command\":\"pwd\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "Exit code: 0\nOutput:\nS:/hack"},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
            ],
            "parallel_tool_calls": True,
            "prompt_cache_key": "thread-cache-key",
            "client_metadata": {"thread_id": "thread_1"},
            "text": {"verbosity": "low"},
        }
    )
    payload = build_bill015_payload(n)

    assert payload["input"][1]["type"] == "function_call"
    assert payload["input"][2]["type"] == "function_call_output"
    assert payload["input"][-1]["content"][0]["text"] == "continue"
    assert "Local proxy state for continuity" not in json.dumps(payload["input"], ensure_ascii=False)
    assert payload["parallel_tool_calls"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["prompt_cache_key"] == "thread-cache-key"
    assert payload["client_metadata"] == {"thread_id": "thread_1"}
    assert payload["text"] == {"verbosity": "low"}
    assert "If asked what model you are" not in payload["instructions"]


def test_bill015_payload_omits_reasoning_when_client_and_config_are_default(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "reasoning_effort", "")
    monkeypatch.setattr(settings, "reasoning_summary", "")
    n = normalize_responses_request({"model": "gpt-5.5", "input": "hello"})

    payload = build_bill015_payload(n)

    assert "reasoning" not in payload


def test_bill015_payload_normalizes_call_output_pairs_without_truncating_outputs():
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    long_output = "A" * 20000
    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "input": [
                {"type": "function_call_output", "call_id": "orphan", "output": "should be removed"},
                {"type": "function_call", "name": "shell_command", "call_id": "pending", "arguments": "{\"command\":\"sleep 1\"}"},
                {"type": "function_call", "name": "shell_command", "call_id": "done", "arguments": "{\"command\":\"big\"}"},
                {"type": "function_call_output", "call_id": "done", "output": long_output},
                {"type": "tool_search_call", "call_id": "search_pending", "execution": "client", "arguments": {"query": "node_repl"}},
            ],
        }
    )
    payload = build_bill015_payload(n)
    items = payload["input"]

    assert all(item.get("call_id") != "orphan" for item in items)
    pending_idx = next(i for i, item in enumerate(items) if item.get("call_id") == "pending")
    assert items[pending_idx + 1] == {"type": "function_call_output", "call_id": "pending", "output": "aborted"}
    done_output = next(item for item in items if item.get("type") == "function_call_output" and item.get("call_id") == "done")
    assert done_output["output"] == long_output
    search_idx = next(i for i, item in enumerate(items) if item.get("call_id") == "search_pending")
    assert items[search_idx + 1] == {
        "type": "tool_search_output",
        "call_id": "search_pending",
        "status": "completed",
        "execution": "client",
        "tools": [],
    }


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


def test_response_stream_uses_in_progress_heartbeat():
    from app.models import Bill015Result, NormalizedRequest
    from app.response_events import responses_sse_generator
    from app.sse import parse_sse_lines

    async def slow_result():
        await asyncio.sleep(0.02)
        return Bill015Result(local_request_id="resp_local_slow", answer="OK", args_done_seen=True)

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="hello", want_stream=True, client_api="responses", is_primary_path=True)

    async def run():
        chunks = []
        async for chunk in responses_sse_generator(slow_result(), n, "resp_local_slow"):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    text = asyncio.run(run())
    events = [ev.json for ev in parse_sse_lines(text.splitlines(True)) if ev.json]
    types = [event["type"] for event in events]

    assert types[:2] == ["response.created", "response.in_progress"]
    assert ": keep-alive" not in text


def test_tool_call_stream_can_include_commentary_message_before_tools():
    from app.models import Bill015Result, BridgeToolCall, NormalizedRequest
    from app.response_events import response_json, responses_sse_generator
    from app.sse import parse_sse_lines

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="inspect", want_stream=True, client_api="responses", is_primary_path=True)
    result = Bill015Result(
        local_request_id="resp_local_mixed",
        bridge_mode="tool_call",
        answer="我先检查目录结构。",
        tool_calls=[BridgeToolCall(id="call_mixed", name="shell_command", arguments='{"command":"Get-ChildItem"}')],
        args_done_seen=True,
    )

    async def run():
        chunks = []
        async for chunk in responses_sse_generator(asyncio.sleep(0, result), n, "resp_local_mixed"):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    text = asyncio.run(run())
    objs = [ev.json for ev in parse_sse_lines(text.splitlines(True)) if ev.json]
    added = [obj for obj in objs if obj["type"] == "response.output_item.added"]
    done = [obj for obj in objs if obj["type"] == "response.output_item.done"]
    completed = [obj for obj in objs if obj["type"] == "response.completed"][-1]["response"]

    assert [(obj["output_index"], obj["item"]["type"]) for obj in added] == [(0, "message"), (1, "function_call")]
    assert done[0]["item"]["type"] == "message"
    assert done[0]["item"]["phase"] == "commentary"
    assert done[1]["item"]["type"] == "function_call"
    assert [item["type"] for item in completed["output"]] == ["message", "function_call"]
    assert completed["output"][0]["phase"] == "commentary"
    assert completed["output"][0]["content"][0]["text"] == "我先检查目录结构。"

    body = response_json(result, n)
    assert [item["type"] for item in body["output"]] == ["message", "function_call"]


def test_http_timeout_read_is_not_shorter_than_args_done(monkeypatch):
    from app.config import settings
    from app.upstream_client import http_timeout

    monkeypatch.setattr(settings, "upstream_timeout_seconds", 300.0)
    monkeypatch.setattr(settings, "args_done_timeout_ms", 300000)
    monkeypatch.setattr(settings, "upstream_idle_timeout_ms", 180000)

    timeout = http_timeout(settings)

    assert timeout.read == 300.0


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


def test_native_tool_first_direct_function_and_final_answer_are_parsed(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest
    from app.sse import SSEEvent

    monkeypatch.setattr(settings, "final_answer_tool_name", "submit_final_answer")
    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)

    tool_result = upstream.collect_bill015_result_from_events(
        [
            SSEEvent("response.output_item.added", '{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_shell","name":"shell_command"}}'),
            SSEEvent("response.function_call_arguments.done", '{"type":"response.function_call_arguments.done","item_id":"fc_1","call_id":"call_shell","arguments":"{\\"command\\":\\"pwd\\"}"}'),
        ],
        n,
    )
    assert tool_result.bridge_mode == "tool_call"
    assert tool_result.tool_calls[0].id == "call_shell"
    assert tool_result.tool_calls[0].name == "shell_command"
    assert tool_result.tool_calls[0].arguments == '{"command":"pwd"}'
    assert tool_result.aborted is True

    final_result = upstream.collect_bill015_result_from_events(
        [
            SSEEvent("response.output_item.added", '{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_2","call_id":"call_final","name":"submit_final_answer"}}'),
            SSEEvent("response.function_call_arguments.done", '{"type":"response.function_call_arguments.done","item_id":"fc_2","call_id":"call_final","arguments":"{\\"answer\\":\\"DONE\\"}"}'),
        ],
        n,
    )
    assert final_result.bridge_mode == "answer"
    assert final_result.answer == "DONE"
    assert final_result.tool_calls == []
    assert final_result.aborted is True


def test_native_tool_first_custom_and_tool_search_boundaries_are_parsed():
    from app import upstream
    from app.models import NormalizedRequest
    from app.sse import SSEEvent

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)

    custom_result = upstream.collect_bill015_result_from_events(
        [
            SSEEvent("response.output_item.added", '{"type":"response.output_item.added","item":{"type":"custom_tool_call","id":"ct_1","call_id":"call_patch","name":"apply_patch"}}'),
            SSEEvent("response.custom_tool_call_input.done", '{"type":"response.custom_tool_call_input.done","item_id":"ct_1","input":"*** Begin Patch\\n*** End Patch"}'),
        ],
        n,
    )
    assert custom_result.bridge_mode == "tool_call"
    assert custom_result.tool_calls[0].call_type == "custom"
    assert custom_result.tool_calls[0].name == "apply_patch"
    assert custom_result.tool_calls[0].arguments == "*** Begin Patch\n*** End Patch"

    search_result = upstream.collect_bill015_result_from_events(
        [
            SSEEvent("response.output_item.done", '{"type":"response.output_item.done","item":{"type":"tool_search_call","id":"ts_1","call_id":"call_search","execution":"client","arguments":{"query":"browser tools","limit":8}}}'),
        ],
        n,
    )
    assert search_result.bridge_mode == "tool_call"
    assert search_result.tool_calls[0].call_type == "tool_search"
    assert search_result.tool_calls[0].name == "tool_search"
    assert '"query": "browser tools"' in search_result.tool_calls[0].arguments
