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
    assert proc.stdout.strip() == "False 8192 8192 65536 900.0 900000 900000 2   False"


def test_legacy_strict_zero_environment_controls_safe_bridge_defaults(tmp_path):
    env = os.environ.copy()
    env["BILL015_CONFIG_PATH"] = str(tmp_path / "missing-config.json")
    env["BILL015_STRICT_ZERO"] = "false"
    env.pop("BILL015_FORCE_EMIT_VALUE", None)
    env.pop("BILL015_BLOCK_PASSTHROUGH", None)
    env.pop("BILL015_BLOCK_NORMAL_MODE", None)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from app.config import settings; "
                "print(settings.force_emit_value, settings.block_passthrough, settings.block_normal_mode)"
            ),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "False False False"


def test_capacity_limiter_bounds_active_and_queue_and_releases_cleanly():
    from app.capacity import CapacityLimiter, QueueFullError, QueueWaitTimeoutError
    from app.state import RuntimeState

    async def run():
        state = RuntimeState()
        limiter = CapacityLimiter(max_active=1, max_queue=1, queue_timeout_ms=20, state=state)
        first = await limiter.acquire()
        second_task = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        with __import__("pytest").raises(QueueFullError):
            await limiter.acquire()
        with __import__("pytest").raises(QueueWaitTimeoutError):
            await second_task
        assert state.snapshot()["active_requests"] == 1
        assert state.snapshot()["queued_requests"] == 0
        assert state.snapshot()["rejected_busy_total"] == 1
        assert state.snapshot()["queue_timeout_total"] == 1
        await first.release()
        assert state.snapshot()["active_requests"] == 0
        replacement = await limiter.acquire()
        await replacement.release()
        return state.snapshot()

    snapshot = asyncio.run(run())
    assert snapshot["active_requests"] == 0
    assert snapshot["queued_requests"] == 0


def test_run_and_record_total_timeout_releases_capacity(monkeypatch):
    from app import main
    from app.capacity import CapacityLimiter
    from app.models import NormalizedRequest
    from app.state import RuntimeState

    state = RuntimeState()
    limiter = CapacityLimiter(max_active=1, max_queue=0, queue_timeout_ms=20, state=state)
    monkeypatch.setattr(main, "runtime_state", state)
    monkeypatch.setattr(main, "capacity_limiter", limiter)
    monkeypatch.setattr(main.settings, "request_total_timeout_ms", 10)
    monkeypatch.setattr(main.audit_logger, "write", lambda record: None)

    async def never_finishes(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(main, "execute_bill015", never_finishes)
    request = NormalizedRequest(
        model="gpt-test",
        instructions="",
        user_input="x",
        want_stream=False,
        client_api="responses",
        is_primary_path=True,
    )

    with __import__("pytest").raises(HTTPException) as exc:
        asyncio.run(main.run_and_record(request, "exploit"))
    assert exc.value.status_code == 504
    snapshot = state.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["request_timeout_total"] == 1


def test_strict_unknown_tool_schema_disables_tui_dynamic_discovery_without_registry(monkeypatch):
    from app.config import settings
    from app.payloads import build_emit_value_schema
    from app.tool_bridge import parse_function_arguments

    monkeypatch.setattr(settings, "tool_bridge_allow_unknown_tools", False)
    schema = build_emit_value_schema(settings, {})
    params = schema["parameters"]["properties"]

    assert params["mode"]["enum"] == ["answer"]
    assert params["tool_calls"]["maxItems"] == 0

    answer, _, _, mode, calls = parse_function_arguments(
        '{"mode":"tool_call","answer":"","tool_calls":[{"type":"function","namespace":"","name":"not_registered","arguments":{},"input":""}]}',
        tool_registry={},
    )
    assert mode == "answer"
    assert calls == []
    assert answer

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
    assert search_mode == "answer"
    assert search_calls == []


def test_invalid_or_display_named_tool_calls_do_not_trigger_tui_dynamic_discovery():
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
        allow_dynamic_tools=False,
    )

    assert mode == "answer"
    assert answer == "我需要读取当前 Chrome 页面。"
    assert calls == []

    blank_answer, _, _, blank_mode, blank_calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "",
                "tool_calls": [{"type": "function", "tool_name": "Chrome Integration", "parameters": {"action": "extract text"}}],
            }
        ),
        tool_registry={"tool_search": {"call_type": "tool_search", "output_name": "tool_search", "raw_type": "tool_search"}},
        allow_dynamic_tools=False,
    )
    assert blank_mode == "answer"
    assert blank_answer
    assert blank_calls == []


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


def test_custom_tool_arguments_are_not_lost_when_input_is_empty():
    from app.tool_bridge import parse_function_arguments

    registry = {"exec": {"call_type": "custom", "output_name": "exec", "raw_type": "custom"}}
    _, _, _, mode, calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "",
                "tool_calls": [
                    {
                        "type": "custom",
                        "namespace": "",
                        "name": "exec",
                        "arguments": "Get-ChildItem",
                        "input": "",
                    }
                ],
            }
        ),
        tool_registry=registry,
    )

    assert mode == "tool_call"
    assert calls[0].call_type == "custom"
    assert calls[0].name == "exec"
    assert calls[0].arguments == "Get-ChildItem"


def test_tool_schema_prioritizes_core_tools_beyond_old_96_cap():
    from app.tool_bridge import build_typed_tool_call_schema

    registry = {
        f"dummy_{idx:03d}": {"call_type": "function", "output_name": f"dummy_{idx:03d}", "raw_type": "function"}
        for idx in range(180)
    }
    registry["zzzz_exec"] = {"call_type": "custom", "output_name": "exec", "raw_type": "custom"}
    registry["zzzz_browser"] = {"call_type": "function", "output_name": "navigate", "namespace": "mcp__chrome_browser", "raw_type": "function"}
    registry["zzzz_tool_search"] = {"call_type": "tool_search", "output_name": "tool_search", "raw_type": "tool_search"}

    schema = build_typed_tool_call_schema(registry, max_tools=32)
    names = schema["properties"]["name"]["enum"]

    assert "exec" in names
    assert "tool_search" in names
    assert "mcp__chrome_browser.navigate" in names


def test_local_web_research_intent_detector_avoids_project_search_false_positives():
    from app.local_web_research import detect_local_web_research_intent

    assert detect_local_web_research_intent("帮我上网查一下今天 OpenAI 有什么最新新闻")
    assert detect_local_web_research_intent("search the web for the latest Python release")
    assert detect_local_web_research_intent("打开 https://example.com 看看页面内容")
    assert not detect_local_web_research_intent("修复这个项目的网络搜索功能")
    assert not detect_local_web_research_intent("在仓库里搜索 web_search 字符串")


def test_local_web_research_guidance_prefers_concrete_browser_tool_for_gpt56(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload, use_responses_lite_upstream

    monkeypatch.setattr(settings, "bridge_strategy", "emit_value")
    monkeypatch.setattr(settings, "strict_zero", True)

    body = {
        "model": "gpt-5.6-sol",
        "client_metadata": {"client": "codex-cli"},
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp__chrome",
                        "description": "Chrome browser control",
                        "tools": [
                            {
                                "type": "function",
                                "name": "extract_text",
                                "description": "Read the current Chrome browser page text",
                                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                            }
                        ],
                    },
                    {
                        "type": "tool_search",
                        "execution": "client",
                        "description": "discover deferred tools",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
                    },
                ],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "search the web for today's Codex release notes"}]},
        ],
    }

    n = normalize_responses_request(body)
    payload = build_bill015_payload(n)

    assert n.responses_lite is True
    assert use_responses_lite_upstream(n, settings) is False
    assert "LOCAL WEB RESEARCH POLICY" in payload["instructions"]
    assert "mcp__chrome.extract_text" in payload["instructions"]
    assert "do not call tool_search first" in payload["instructions"]
    assert json.dumps(payload["input"], ensure_ascii=False).count('"type": "additional_tools"') == 0


def test_local_web_research_preflight_emits_tool_search_without_upstream_key(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.upstream import execute_bill015

    monkeypatch.setattr(settings, "tool_bridge_local_web_research_preflight", True)

    n = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "stream": False,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "帮我上网查一下今天东京天气"}]}],
            "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "description": "discover deferred tools",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
                }
            ],
        }
    )

    result = asyncio.run(execute_bill015(n, "exploit"))

    assert result.bridge_mode == "tool_call"
    assert result.tool_calls[0].call_type == "tool_search"
    assert result.tool_calls[0].name == "tool_search"
    assert "local web research browser chrome playwright" in result.tool_calls[0].arguments
    assert "今天东京天气" in result.tool_calls[0].arguments
    assert "local_web_research_preflight" in result.retry_reasons


def test_local_web_research_preflight_does_not_repeat_completed_discovery(monkeypatch):
    from app.config import settings
    from app.local_web_research import build_local_web_discovery_arguments, local_web_research_preflight
    from app.normalization import normalize_responses_request

    monkeypatch.setattr(settings, "tool_bridge_local_web_research_preflight", True)
    query = "帮我上网查一下今天东京天气"
    n = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": query}]},
                {"type": "tool_search_call", "call_id": "search_done", "execution": "client", "arguments": json.loads(build_local_web_discovery_arguments(query))},
                {"type": "tool_search_output", "call_id": "search_done", "execution": "client", "status": "completed", "tools": []},
            ],
            "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "description": "discover deferred tools",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
                }
            ],
        }
    )

    assert local_web_research_preflight(n, settings) is None


def test_local_web_research_after_tool_search_output_uses_exposed_tool(monkeypatch):
    from app.config import settings
    from app.local_web_research import build_local_web_discovery_arguments
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "bridge_strategy", "emit_value")
    monkeypatch.setattr(settings, "strict_zero", True)

    query = "帮我上网查一下今天东京天气"
    n = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": query}]},
                {"type": "tool_search_call", "call_id": "search_done", "execution": "client", "arguments": json.loads(build_local_web_discovery_arguments(query))},
                {
                    "type": "tool_search_output",
                    "call_id": "search_done",
                    "execution": "client",
                    "status": "completed",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "mcp__node_repl",
                            "description": "Node-backed browser/HTTP helper",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "js",
                                    "description": "Run JavaScript with fetch/playwright support",
                                    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False},
                                }
                            ],
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "description": "discover deferred tools",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
                }
            ],
        }
    )
    payload = build_bill015_payload(n)

    assert "mcp__node_repl.js" in n.tool_registry
    assert "LOCAL WEB RESEARCH POLICY" in payload["instructions"]
    assert "mcp__node_repl.js" in payload["instructions"]
    assert "do not call tool_search first" in payload["instructions"]


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
    assert "Do not write a preamble/progress sentence merely because you are calling a tool" in payload["instructions"]
    assert "When native Codex would say a short preamble" not in payload["instructions"]
    assert '"answer":"",' in payload["instructions"]
    assert "genuinely useful" in payload["tools"][0]["parameters"]["properties"]["answer"]["description"]

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
    monkeypatch.setattr(settings, "strict_zero", False)
    monkeypatch.setattr(settings, "force_emit_value", False)
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


def test_strict_zero_forces_emit_value_even_if_native_tool_first_requested(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "bridge_strategy", "native_tool_first")
    monkeypatch.setattr(settings, "strict_zero", True)
    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "input": "Return OK.",
            "tools": [
                {
                    "type": "function",
                    "name": "shell_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                }
            ],
        }
    )

    payload = build_bill015_payload(n)

    assert payload["tool_choice"] == {"type": "function", "name": "emit_value"}
    assert [tool.get("name") for tool in payload["tools"]] == ["emit_value"]


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


def test_malformed_emit_value_arguments_becomes_answer_not_stream_disconnect():
    from app.normalization import normalize_responses_request
    from app.sse import parse_sse_lines
    from app.upstream import collect_bill015_result_from_events

    n = normalize_responses_request({"model": "gpt-5.5", "input": "continue"})
    bad_args = '{"mode":"tool_call","answer":"x","tool_calls":[{"type":"custom","name":"apply_patch","input":"*** Begin Patch\nunterminated'
    data = json.dumps({"type": "response.function_call_arguments.done", "arguments": bad_args}, ensure_ascii=False)
    events = parse_sse_lines(["event: response.function_call_arguments.done\n", f"data: {data}\n", "\n"])

    result = collect_bill015_result_from_events(events, n)

    assert result.bridge_mode == "answer"
    assert result.tool_calls == []
    assert result.malformed_function_args is True
    assert "JSON" in result.answer
    assert result.error is None


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


def test_tool_search_auto_expansion_is_disabled_by_default_but_desktop_dynamic_call_survives():
    from app.tool_bridge import parse_function_arguments

    answer, _, _, mode, calls = parse_function_arguments(
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
    assert len(calls) == 1
    assert calls[0].call_type == "tool_search"
    assert calls[0].name == "tool_search"


def test_emit_value_schema_preserves_native_tool_search_for_tui(monkeypatch):
    from app.config import settings
    from app.payloads import build_emit_value_schema

    monkeypatch.setattr(settings, "tool_bridge_allow_unknown_tools", False)
    schema = build_emit_value_schema(
        settings,
        {"tool_search": {"call_type": "tool_search", "output_name": "tool_search", "raw_type": "tool_search"}},
        bridge_target="tui",
    )
    params = schema["parameters"]["properties"]

    assert params["mode"]["enum"] == ["answer", "tool_call"]
    assert "maxItems" not in params["tool_calls"]
    assert "tool_search" in params["tool_calls"]["items"]["properties"]["name"]["enum"]


def test_responses_lite_additional_tools_populate_cli_registry(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload, use_responses_lite_upstream
    from app.tool_bridge import parse_function_arguments

    monkeypatch.setattr(settings, "bridge_strategy", "emit_value")
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(settings, "tool_bridge_allow_unknown_tools", False)

    body = {
        "model": "gpt-5.6-sol",
        "client_metadata": {"client": "codex-cli"},
        # Mirrors Codex core's Responses Lite request shape: native tools are
        # not in top-level tools; they are carried in an additional_tools item.
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
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
                    {
                        "type": "tool_search",
                        "execution": "client",
                        "description": "discover deferred tools",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "patch",
                        "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                    },
                ],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
        ],
    }

    n = normalize_responses_request(body)
    payload = build_bill015_payload(n)
    assert n.responses_lite is True
    assert use_responses_lite_upstream(n, settings) is False
    # Strict-zero still ingests Codex Responses Lite input, but does not forward
    # the Lite wire shape/header upstream. The upstream sees the same single
    # BILL-015 bridge function that gpt-5.5 used without quota deltas.
    assert payload["tools"][0]["name"] == "emit_value"
    assert "instructions" in payload
    assert payload["input"][0]["type"] == "message"
    assert json.dumps(payload["input"], ensure_ascii=False).count('"type": "additional_tools"') == 0
    item_schema = payload["tools"][0]["parameters"]["properties"]["tool_calls"]["items"]
    names = item_schema["properties"]["name"]["enum"]

    assert n.tool_bridge_target == "tui"
    assert "shell_command" in n.tool_registry
    assert "apply_patch" in n.tool_registry
    assert "tool_search" in n.tool_registry
    assert "shell_command" in names
    assert "apply_patch" in names
    assert "tool_search" in names
    assert payload["tools"][0]["parameters"]["properties"]["mode"]["enum"] == ["answer", "tool_call"]
    assert "context" not in payload.get("reasoning", {})
    developer_text = payload["instructions"]
    assert "shell_command" in developer_text
    assert "apply_patch" in developer_text

    _, _, _, mode, calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "",
                "tool_calls": [{"type": "function", "name": "shell_command", "arguments": {"command": "Get-ChildItem"}, "input": ""}],
            }
        ),
        tool_registry=n.tool_registry,
        allow_dynamic_tools=False,
    )
    assert mode == "tool_call"
    assert calls[0].name == "shell_command"


def test_responses_lite_additional_tools_preserves_non_tool_developer_text(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "bridge_strategy", "emit_value")
    monkeypatch.setattr(settings, "strict_zero", True)

    n = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "instructions": "IMPORTANT 5.6 developer hint",
                    "tools": [{"type": "function", "name": "shell_command", "parameters": {"type": "object", "properties": {}}}],
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
            ],
        }
    )
    payload = build_bill015_payload(n)
    dumped_input = json.dumps(payload["input"], ensure_ascii=False)

    assert '"type": "additional_tools"' not in dumped_input
    assert "IMPORTANT 5.6 developer hint" in dumped_input
    assert payload["input"][0]["role"] == "developer"


def test_responses_lite_non_strict_still_uses_codex_lite_transport(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload, use_responses_lite_upstream

    monkeypatch.setattr(settings, "bridge_strategy", "emit_value")
    monkeypatch.setattr(settings, "strict_zero", False)
    monkeypatch.setattr(settings, "force_emit_value", False)

    body = {
        "model": "gpt-5.6-sol",
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": []},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ],
    }

    n = normalize_responses_request(body)
    payload = build_bill015_payload(n)

    assert n.responses_lite is True
    assert use_responses_lite_upstream(n, settings) is True
    assert "tools" not in payload
    assert "instructions" not in payload
    assert payload["input"][0]["type"] == "additional_tools"
    assert payload["reasoning"]["context"] == "all_turns"


def test_execute_bill015_strict_zero_suppresses_responses_lite_upstream_header(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.normalization import normalize_responses_request

    captured: dict[str, object] = {}
    success_lines = [
        "event: response.output_item.added",
        'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
        "",
        "event: response.function_call_arguments.done",
        'data: {"type":"response.function_call_arguments.done","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"OK\\",\\"tool_calls\\":[]}"}',
        "",
    ]

    class DummyStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            for line in success_lines:
                yield line

        async def aclose(self):
            captured["closed"] = True

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            captured["headers"] = kwargs["headers"]
            captured["json"] = kwargs["json"]
            return DummyStreamResponse()

    _configure_test_key_pool(monkeypatch, settings, "sk-lite-safe", [])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)

    n = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "client_metadata": {"x-openai-internal-codex-responses-lite": "true", "thread_id": "thread_1"},
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "function", "name": "shell_command", "parameters": {"type": "object", "properties": {}}}],
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            ],
        },
        request_headers={"x-openai-internal-codex-responses-lite": "true", "session-id": "session_1"},
    )

    result = asyncio.run(upstream.execute_bill015(n, "exploit"))

    assert result.answer == "OK"
    assert captured["closed"] is True
    headers = captured["headers"]
    payload = captured["json"]
    assert "x-openai-internal-codex-responses-lite" not in {str(k).lower(): v for k, v in headers.items()}
    assert payload["tools"][0]["name"] == "emit_value"
    assert "instructions" in payload
    assert payload.get("client_metadata") == {"thread_id": "thread_1"}
    assert payload["input"][0]["type"] == "message"
    assert json.dumps(payload["input"], ensure_ascii=False).count('"type": "additional_tools"') == 0


def test_responses_lite_multi_agent_v1_and_v2_tools_roundtrip_on_cli(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload, use_responses_lite_upstream
    from app.tool_bridge import parse_function_arguments

    monkeypatch.setattr(settings, "bridge_strategy", "emit_value")
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(settings, "tool_bridge_allow_unknown_tools", False)

    body = {
        "model": "gpt-5.6-sol",
        "client_metadata": {"client": "codex-cli"},
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "multi_agent_v1",
                        "description": "Tools for spawning and managing sub-agents.",
                        "tools": [
                            {
                                "type": "function",
                                "name": "spawn_agent",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"message": {"type": "string"}},
                                    "required": ["message"],
                                    "additionalProperties": False,
                                },
                            },
                            {
                                "type": "function",
                                "name": "wait_agent",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"targets": {"type": "array", "items": {"type": "string"}}},
                                    "required": ["targets"],
                                    "additionalProperties": False,
                                },
                            },
                        ],
                    },
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_name": {"type": "string"},
                                "message": {"type": "string"},
                            },
                            "required": ["task_name", "message"],
                            "additionalProperties": False,
                        },
                    },
                ],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "delegate two checks"}]},
        ],
    }

    n = normalize_responses_request(body)
    payload = build_bill015_payload(n)
    names = payload["tools"][0]["parameters"]["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"]

    assert n.tool_bridge_target == "tui"
    assert n.responses_lite is True
    assert use_responses_lite_upstream(n, settings) is False
    assert "multi_agent_v1.spawn_agent" in names
    assert "wait_agent" in names
    assert "spawn_agent" in names

    _, _, _, v1_mode, v1_calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "namespace": "multi_agent_v1",
                        "name": "spawn_agent",
                        "arguments": {"message": "inspect tools"},
                        "input": "",
                    }
                ],
            }
        ),
        tool_registry=n.tool_registry,
    )
    assert v1_mode == "tool_call"
    assert (v1_calls[0].namespace, v1_calls[0].name) == ("multi_agent_v1", "spawn_agent")

    _, _, _, v2_mode, v2_calls = parse_function_arguments(
        json.dumps(
            {
                "mode": "tool_call",
                "answer": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "namespace": "",
                        "name": "spawn_agent",
                        "arguments": {"task_name": "inspect_tools", "message": "inspect tools"},
                        "input": "",
                    }
                ],
            }
        ),
        tool_registry=n.tool_registry,
    )
    assert v2_mode == "tool_call"
    assert (v2_calls[0].namespace, v2_calls[0].name) == (None, "spawn_agent")


def test_responses_lite_passthrough_sets_codex_header(monkeypatch):
    import asyncio

    from app import upstream_client
    from app.config import settings

    captured_headers = []

    class DummyResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        def json(self):
            return {"id": "resp_ok"}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            captured_headers.append(kwargs.get("headers", {}))
            return DummyResponse()

    monkeypatch.setattr(settings, "upstream_api_key_file_value", "sk-test")
    monkeypatch.setattr(settings, "upstream_api_keys_file_value", [])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(upstream_client.httpx, "AsyncClient", DummyClient)

    result = asyncio.run(
        upstream_client.normal_forward_json(
            {
                "model": "gpt-test",
                "input": [
                    {"type": "additional_tools", "role": "developer", "tools": []},
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                ],
            },
            request_headers={
                "session-id": "session_1",
                "thread-id": "thread_1",
                "x-client-request-id": "thread_1",
                "x-openai-subagent": "collab_spawn",
                "x-codex-window-id": "thread_1:0",
                "authorization": "Bearer client-secret-must-not-forward",
            },
        )
    )

    assert result == {"id": "resp_ok"}
    assert captured_headers[-1]["x-openai-internal-codex-responses-lite"] == "true"
    assert captured_headers[-1]["session-id"] == "session_1"
    assert captured_headers[-1]["thread-id"] == "thread_1"
    assert captured_headers[-1]["x-client-request-id"] == "thread_1"
    assert captured_headers[-1]["x-openai-subagent"] == "collab_spawn"
    assert captured_headers[-1]["x-codex-window-id"] == "thread_1:0"
    assert captured_headers[-1]["Authorization"] == "Bearer sk-test"


def test_emit_value_schema_preserves_desktop_dynamic_tools(monkeypatch):
    from app.config import settings
    from app.payloads import build_emit_value_schema

    monkeypatch.setattr(settings, "tool_bridge_allow_unknown_tools", False)
    schema = build_emit_value_schema(
        settings,
        {"tool_search": {"call_type": "tool_search", "output_name": "tool_search", "raw_type": "tool_search"}},
        bridge_target="desktop",
    )
    params = schema["parameters"]["properties"]

    assert params["mode"]["enum"] == ["answer", "tool_call"]
    assert "maxItems" not in params["tool_calls"]
    assert "tool_search" in params["tool_calls"]["items"]["properties"]["name"]["enum"]


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
            "prompt_cache_options": {"type": "ephemeral"},
            "service_tier": "flex",
            "truncation": "disabled",
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
    assert payload["prompt_cache_options"] == {"type": "ephemeral"}
    assert payload["service_tier"] == "flex"
    assert payload["truncation"] == "disabled"
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


def test_bill015_payload_rewrites_invalid_function_call_item_ids():
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    n = normalize_responses_request(
        {
            "model": "gpt-5.5",
            "input": [
                {"type": "function_call", "id": "item_a14d5ab7e87a439abb75e01d", "call_id": "call_bad_id", "name": "shell_command", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_bad_id", "output": "ok"},
            ],
        }
    )

    payload = build_bill015_payload(n)
    call = next(item for item in payload["input"] if item.get("type") == "function_call")
    output = next(item for item in payload["input"] if item.get("type") == "function_call_output")

    assert call["id"].startswith("fc_")
    assert call["id"] != "item_a14d5ab7e87a439abb75e01d"
    assert call["call_id"] == output["call_id"] == "call_bad_id"


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


def test_image_inputs_are_rejected_explicitly_by_default(monkeypatch):
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
    assert response.json()["error"]["message"]["code"] == "local_proxy_vision_unsupported"


def test_image_inputs_local_extract_mode_is_loss_explicit(monkeypatch):
    from app.config import settings
    from app.normalization import sanitize_unsupported_image_inputs

    monkeypatch.setattr(settings, "multimodal_strategy", "local_extract")
    body, count = sanitize_unsupported_image_inputs(
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}]}
    )
    dumped = json.dumps(body, ensure_ascii=False)
    assert count == 1
    assert "data:image" not in dumped
    assert "does not support image/screenshot uploads" in dumped


def test_request_aware_tool_selection_keeps_core_and_defers_excess():
    from app.tool_bridge import select_tool_registry

    shared = {}
    for i in range(300):
        name = f"boring_tool_{i}"
        shared[name] = {"call_type": "function", "output_name": name, "schema": {"description": "generic"}}
    shared["apply_patch"] = {"call_type": "custom", "output_name": "apply_patch", "schema": {"description": "edit files"}}
    shared["special_database_lookup"] = {"call_type": "function", "output_name": "special_database_lookup", "schema": {"description": "database lookup"}}

    selected, stats = select_tool_registry(shared, "please do a special database lookup", max_tools=32)
    assert "apply_patch" in selected
    assert "special_database_lookup" in selected
    assert stats["canonical_tool_count"] == 302
    assert stats["schema_tool_count"] == 32
    assert stats["deferred_tool_count"] == 270
    assert stats["dropped_tool_count"] == 0


def test_emit_value_integrity_limits_block_oversized_tool_call(monkeypatch):
    from app.config import settings
    from app.tool_bridge import parse_function_arguments

    monkeypatch.setattr(settings, "max_tool_argument_chars", 16)
    raw = json.dumps({"mode": "tool_call", "answer": "", "tool_calls": [{"type": "custom", "name": "apply_patch", "input": "x" * 50, "arguments": "{}"}]})
    answer, malformed, _, mode, calls = parse_function_arguments(
        raw, settings, {"apply_patch": {"call_type": "custom", "output_name": "apply_patch", "raw_type": "custom"}}
    )
    assert malformed is True
    assert mode == "answer"
    assert not calls
    assert "完整性上限" in answer


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


def test_execute_bill015_retries_pre_stream_http_500_with_safe_bridge(monkeypatch):
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
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(settings, "force_emit_value", True)
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


def test_upstream_completed_without_args_done_uses_completed_text():
    from app import upstream
    from app.models import NormalizedRequest
    from app.sse import SSEEvent

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)

    result = upstream.collect_bill015_result_from_events(
        [
            SSEEvent(
                "response.completed",
                '{"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"DONE"}]}]}}',
            ),
        ],
        n,
    )

    assert result.answer == "DONE"
    assert result.bridge_mode == "answer"
    assert result.args_done_seen is False
    assert "upstream_completed_without_args_done_used_text" in result.retry_reasons


def test_upstream_empty_close_without_args_done_is_an_error():
    from app import upstream
    from app.models import NormalizedRequest
    from app.sse import SSEEvent

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)

    result = upstream.collect_bill015_result_from_events([SSEEvent("message", "[DONE]")], n)

    assert result.answer == ""
    assert result.error == "upstream ended without a final answer or tool call"
    assert result.bridge_mode == "answer"
    assert result.args_done_seen is False
    assert "upstream_completed_without_args_done" in result.retry_reasons


def test_execute_bill015_empty_stream_exhaustion_returns_answer_not_http_error(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest

    class EmptyStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            if False:
                yield ""

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
            return EmptyStreamResponse()

    monkeypatch.setattr(settings, "upstream_api_key_file_value", "sk-test")
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "empty_stream_retries", 2)
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="Return OK", want_stream=False, client_api="responses", is_primary_path=True)
    result = asyncio.run(upstream.execute_bill015(n, "exploit"))

    assert DummyClient.calls == 3
    assert result.bridge_mode == "answer"
    assert result.error is None
    assert result.args_done_seen is True
    assert "empty_stream_exhausted:3" in result.retry_reasons
    assert "没有返回最终答案或工具调用" in result.answer



def _configure_test_key_pool(monkeypatch, settings, primary: str, additional: list[str]) -> None:
    monkeypatch.delenv(settings.upstream_api_key_env, raising=False)
    monkeypatch.setattr(settings, "upstream_api_key_file_value", primary)
    monkeypatch.setattr(settings, "upstream_api_keys_file_value", additional)


def test_cyber_policy_error_matcher_uses_logged_error_fields():
    from app.upstream_errors import CYBER_POLICY_ERROR_MESSAGE, is_cyber_policy_rotation_error, key_rotation_error_reason

    event = {
        "type": "error",
        "error": {
            "type": "invalid_request",
            "code": "cyber_policy",
            "message": CYBER_POLICY_ERROR_MESSAGE,
            "param": None,
        },
        "sequence_number": 225,
    }
    assert is_cyber_policy_rotation_error(event)
    assert key_rotation_error_reason(event) == "cyber_policy"
    assert key_rotation_error_reason(f"event: error\ndata: {json.dumps(event)}\n\n") == "cyber_policy"
    assert key_rotation_error_reason({"error": {"message": CYBER_POLICY_ERROR_MESSAGE}}) is None
    assert key_rotation_error_reason({"error": {"code": "cyber_policy", "message": "If this seems wrong, try rephrasing your request"}}) is None
    assert key_rotation_error_reason({"type": "error", "error": {"sequence_number": 113}}) is None


def test_execute_bill015_rotates_key_on_http_cyber_policy_under_strict_zero(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest
    from app.upstream_errors import CYBER_POLICY_ERROR_MESSAGE

    success_lines = [
        'event: response.output_item.added',
        'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
        '',
        'event: response.function_call_arguments.done',
        'data: {"type":"response.function_call_arguments.done","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"OK\\",\\"tool_calls\\":[]}"}',
        '',
    ]

    class DummyStreamResponse:
        headers = {"content-type": "application/json"}

        def __init__(self, status_code: int):
            self.status_code = status_code

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return json.dumps(
                {"error": {"type": "invalid_request", "code": "cyber_policy", "message": CYBER_POLICY_ERROR_MESSAGE, "param": None}}
            ).encode()

        async def aiter_lines(self):
            for line in success_lines:
                yield line

        async def aclose(self):
            return None

    class DummyClient:
        authorizations = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            self.authorizations.append(kwargs["headers"]["Authorization"])
            return DummyStreamResponse(400 if len(self.authorizations) == 1 else 200)

    _configure_test_key_pool(monkeypatch, settings, "sk-http-1", ["sk-http-2", "sk-http-3"])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "upstream_retries", 0)
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(upstream.asyncio, "sleep", fake_sleep)

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="Return OK", want_stream=False, client_api="responses", is_primary_path=True)
    result = asyncio.run(upstream.execute_bill015(n, "exploit"))

    assert DummyClient.authorizations == ["Bearer sk-http-1", "Bearer sk-http-2"]
    assert result.answer == "OK"
    assert result.key_switch_count == 1
    assert result.upstream_key_index == 2
    assert result.upstream_key_count == 3
    assert result.retry_count == 1
    assert result.retry_reasons == ["cyber_policy:key_switch:1->2"]
    assert sleeps == []


def test_execute_bill015_rotates_key_on_sse_cyber_policy(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest
    from app.upstream_errors import CYBER_POLICY_ERROR_MESSAGE

    error_lines = [
        'event: response.created',
        'data: {"type":"response.created","response":{"id":"resp_bad"}}',
        '',
        'event: error',
        'data: '
        + json.dumps(
            {"type": "error", "error": {"type": "invalid_request", "code": "cyber_policy", "message": CYBER_POLICY_ERROR_MESSAGE, "param": None}}
        ),
        '',
    ]
    success_lines = [
        'event: response.output_item.added',
        'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
        '',
        'event: response.function_call_arguments.done',
        'data: {"type":"response.function_call_arguments.done","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"SSE OK\\",\\"tool_calls\\":[]}"}',
        '',
    ]

    class DummyStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            for line in self.lines:
                yield line

        async def aclose(self):
            return None

    class DummyClient:
        authorizations = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            self.authorizations.append(kwargs["headers"]["Authorization"])
            return DummyStreamResponse(error_lines if len(self.authorizations) == 1 else success_lines)

    _configure_test_key_pool(monkeypatch, settings, "sk-sse-1", ["sk-sse-2"])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "upstream_retries", 0)
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(upstream.asyncio, "sleep", fake_sleep)

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="Return OK", want_stream=False, client_api="responses", is_primary_path=True)
    result = asyncio.run(upstream.execute_bill015(n, "exploit"))

    assert DummyClient.authorizations == ["Bearer sk-sse-1", "Bearer sk-sse-2"]
    assert result.answer == "SSE OK"
    assert result.key_switch_count == 1
    assert result.upstream_response_id is None
    assert sleeps == []


def test_execute_bill015_stops_after_cyber_policy_exhausts_key_pool(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest
    from app.upstream_errors import CYBER_POLICY_ERROR_MESSAGE

    class DummyStreamResponse:
        status_code = 400
        headers = {"content-type": "application/json"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return json.dumps({"error": {"type": "invalid_request", "code": "cyber_policy", "message": CYBER_POLICY_ERROR_MESSAGE}}).encode()

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
            return DummyStreamResponse()

    _configure_test_key_pool(monkeypatch, settings, "sk-exhaust-1", ["sk-exhaust-2"])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "upstream_retries", 0)
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(upstream.asyncio, "sleep", fake_sleep)
    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)

    with __import__("pytest").raises(HTTPException):
        asyncio.run(upstream.execute_bill015(n, "exploit"))
    assert DummyClient.calls == 2
    assert sleeps == []


def test_normal_forward_json_rotates_only_for_cyber_policy(monkeypatch):
    from app import upstream_client
    from app.config import settings
    from app.upstream_errors import CYBER_POLICY_ERROR_MESSAGE

    class DummyResponse:
        headers = {"content-type": "application/json"}
        text = ""

        def __init__(self, status_code, obj):
            self.status_code = status_code
            self.obj = obj
            self.text = json.dumps(obj)

        def json(self):
            return self.obj

    class DummyClient:
        authorizations = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            self.authorizations.append(kwargs["headers"]["Authorization"])
            if len(self.authorizations) == 1:
                return DummyResponse(400, {"error": {"type": "invalid_request", "code": "cyber_policy", "message": CYBER_POLICY_ERROR_MESSAGE}})
            return DummyResponse(200, {"id": "resp_ok"})

    _configure_test_key_pool(monkeypatch, settings, "sk-json-1", ["sk-json-2"])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(upstream_client.httpx, "AsyncClient", DummyClient)
    result = asyncio.run(upstream_client.normal_forward_json({"model": "gpt-test", "input": "hello"}))
    assert result == {"id": "resp_ok"}
    assert DummyClient.authorizations == ["Bearer sk-json-1", "Bearer sk-json-2"]



def test_execute_bill015_does_not_rotate_on_http_rephrase_without_cyber_policy(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest

    marker = "If this seems wrong, try rephrasing your request"
    success_lines = [
        'event: response.output_item.added',
        'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
        '',
        'event: response.function_call_arguments.done',
        'data: {"type":"response.function_call_arguments.done","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"HTTP OK\\",\\"tool_calls\\":[]}"}',
        '',
    ]

    class DummyStreamResponse:
        headers = {"content-type": "application/json"}

        def __init__(self, status_code):
            self.status_code = status_code

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return json.dumps({"error": {"message": marker}}).encode()

        async def aiter_lines(self):
            for line in success_lines:
                yield line

        async def aclose(self):
            return None

    class DummyClient:
        authorizations = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            self.authorizations.append(kwargs["headers"]["Authorization"])
            return DummyStreamResponse(400 if len(self.authorizations) == 1 else 200)

    _configure_test_key_pool(monkeypatch, settings, "sk-rephrase-http-1", ["sk-rephrase-http-2"])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "upstream_retries", 0)
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)
    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)

    with __import__("pytest").raises(HTTPException):
        asyncio.run(upstream.execute_bill015(n, "exploit"))
    assert DummyClient.authorizations == ["Bearer sk-rephrase-http-1"]


def test_execute_bill015_does_not_rotate_on_sse_rephrase_without_cyber_policy(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest

    marker = "If this seems wrong, try rephrasing your request"
    error_lines = [
        'event: error',
        json.dumps({"type": "error", "error": {"message": marker}}).join(["data: ", ""]),
        '',
    ]
    success_lines = [
        'event: response.output_item.added',
        'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
        '',
        'event: response.function_call_arguments.done',
        'data: {"type":"response.function_call_arguments.done","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"SSE OK\\",\\"tool_calls\\":[]}"}',
        '',
    ]

    class DummyStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            for line in self.lines:
                yield line

        async def aclose(self):
            return None

    class DummyClient:
        authorizations = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            self.authorizations.append(kwargs["headers"]["Authorization"])
            return DummyStreamResponse(error_lines if len(self.authorizations) == 1 else success_lines)

    _configure_test_key_pool(monkeypatch, settings, "sk-rephrase-sse-1", ["sk-rephrase-sse-2"])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "upstream_retries", 0)
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)
    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)

    with __import__("pytest").raises(HTTPException):
        asyncio.run(upstream.execute_bill015(n, "exploit"))
    assert DummyClient.authorizations == ["Bearer sk-rephrase-sse-1"]


def test_bill015_payload_passes_client_reasoning_through(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "reasoning_effort", "")
    monkeypatch.setattr(settings, "reasoning_summary", "")
    n = normalize_responses_request({"model": "gpt-5.5", "input": "hello", "reasoning": {"effort": "high", "summary": "auto"}})

    payload = build_bill015_payload(n)

    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}


def test_bill015_payload_maps_top_level_reasoning_effort_to_responses_reasoning(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_chat_request, normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "reasoning_effort", "")
    monkeypatch.setattr(settings, "reasoning_summary", "")

    responses_request = normalize_responses_request(
        {"model": "gpt-5.5", "input": "hello", "reasoning_effort": "medium", "reasoning_summary": "auto"}
    )
    chat_request = normalize_chat_request(
        {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "high"}
    )

    assert build_bill015_payload(responses_request)["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert build_bill015_payload(chat_request)["reasoning"] == {"effort": "high"}


def test_bill015_payload_maps_codex_model_reasoning_aliases(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "reasoning_effort", "")
    monkeypatch.setattr(settings, "reasoning_summary", "")

    top_level = normalize_responses_request({"model": "gpt-5.6-sol", "input": "hello", "model_reasoning_effort": "xhigh"})
    metadata = normalize_responses_request(
        {
            "model": "gpt-5.6-sol",
            "input": "hello",
            "client_metadata": {"x-codex-turn-metadata": json.dumps({"model_reasoning_effort": "max"})},
        }
    )

    assert build_bill015_payload(top_level)["reasoning"] == {"effort": "xhigh"}
    assert build_bill015_payload(metadata)["reasoning"] == {"effort": "max"}


def test_bill015_payload_defaults_gpt56_to_high_reasoning(monkeypatch):
    from app.config import settings
    from app.normalization import normalize_responses_request
    from app.payloads import build_bill015_payload

    monkeypatch.setattr(settings, "reasoning_effort", "")
    monkeypatch.setattr(settings, "reasoning_summary", "")

    n = normalize_responses_request({"model": "gpt-5.6-sol", "input": "solve carefully"})

    assert build_bill015_payload(n)["reasoning"] == {"effort": "high"}


def test_context_compaction_preserves_latest_user_and_tool_batch():
    from app.context_compaction import compact_payload_history, payload_input_tokens

    payload = {
        "model": "gpt-test",
        "instructions": "root",
        "input": [
            {"type": "message", "role": "user", "content": "OLD " * 40000},
            {"type": "function_call", "call_id": "call_latest", "name": "shell_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_latest", "output": "LATEST_TOOL_OUTPUT " + ("x" * 20000)},
            {"type": "message", "role": "user", "content": "CURRENT_REQUEST"},
        ],
        "tools": [],
    }
    compacted, removed, clipped = compact_payload_history(payload, 12000, latest_tool_output_max_chars=6000)
    dumped = json.dumps(compacted, ensure_ascii=False)
    assert payload_input_tokens(compacted) < payload_input_tokens(payload)
    assert removed > 0 or clipped > 0
    assert "CURRENT_REQUEST" in dumped
    assert "call_latest" in dumped
    assert "LATEST_TOOL_OUTPUT" in dumped
    assert "CONTEXT CHECKPOINT COMPACTION" in dumped
    summary_items = [
        item
        for item in compacted["input"]
        if isinstance(item, dict) and "CONTEXT CHECKPOINT COMPACTION" in json.dumps(item, ensure_ascii=False)
    ]
    assert summary_items
    assert summary_items[0]["role"] == "user"
    assert summary_items[0]["content"][0]["type"] == "input_text"


def test_context_compaction_aggressive_clips_oversized_latest_user():
    from app.context_compaction import compact_payload_history, payload_input_tokens

    payload = {
        "model": "gpt-test",
        "instructions": "root",
        "input": [
            {"type": "message", "role": "user", "content": "old context " * 30000},
            {"type": "message", "role": "user", "content": "CURRENT_HEAD " + ("x" * 220000) + " CURRENT_TAIL"},
        ],
        "tools": [],
    }

    compacted, removed, clipped = compact_payload_history(payload, 8000, aggressive=True)
    dumped = json.dumps(compacted, ensure_ascii=False)

    assert removed > 0
    assert clipped > 0
    assert payload_input_tokens(compacted) <= 8000
    assert "CURRENT_HEAD" in dumped
    assert "CURRENT_TAIL" in dumped
    assert "LOCAL LOSSY COMPACTION NOTICE" in dumped


def test_context_length_error_matcher():
    from app.upstream_errors import is_context_length_exceeded

    assert is_context_length_exceeded({"type": "error", "error": {"code": "context_length_exceeded"}})
    assert is_context_length_exceeded('data: {"type":"error","error":{"code":"context_length_exceeded"}}\n\n')
    assert not is_context_length_exceeded({"type": "error", "error": {"code": "bad_request"}})


def test_preemptive_compaction_exact_count_only_near_limit():
    from types import SimpleNamespace

    from app.upstream import _needs_exact_preemptive_token_count

    precompact_limit = 115_200

    assert _needs_exact_preemptive_token_count(SimpleNamespace(estimated_input_tokens=0), precompact_limit)
    assert not _needs_exact_preemptive_token_count(SimpleNamespace(estimated_input_tokens=10_000), precompact_limit)
    assert not _needs_exact_preemptive_token_count(SimpleNamespace(estimated_input_tokens=int(precompact_limit * 0.64)), precompact_limit)
    assert _needs_exact_preemptive_token_count(SimpleNamespace(estimated_input_tokens=int(precompact_limit * 0.70)), precompact_limit)


def test_clip_text_token_budget_reuses_original_count(monkeypatch):
    from app import normalization

    text = "abcdef " * 400
    original_calls = 0

    def fake_estimate(value, *, json_like=False):
        nonlocal original_calls
        if value == text:
            original_calls += 1
        return max(1, len(str(value)) // 4)

    monkeypatch.setattr(normalization, "estimate_text_tokens", fake_estimate)

    clipped = normalization._clip_text_to_token_budget(text, 100, keep="head_tail")

    assert clipped != text
    assert "token-budget omitted middle content" in clipped
    assert original_calls == 1


def test_native_transcript_omits_latest_user_without_flattening_body(monkeypatch):
    from app import normalization

    original_flatten = normalization.flatten_content
    current_body = "CURRENT_BODY " * 10_000
    current_body_flatten_calls = 0

    def wrapped_flatten(content):
        nonlocal current_body_flatten_calls
        if content == current_body:
            current_body_flatten_calls += 1
        return original_flatten(content)

    monkeypatch.setattr(normalization, "flatten_content", wrapped_flatten)

    transcript = normalization.native_input_transcript(
        [
            {"type": "message", "role": "user", "content": "old"},
            {"type": "message", "role": "user", "content": current_body},
        ],
        omit_latest_user_body=True,
    )

    assert "current user message moved above" in transcript
    assert "CURRENT_BODY" not in transcript
    assert current_body_flatten_calls == 0


def test_execute_bill015_recovers_args_done_timeout(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest

    success_lines = [
        'event: response.output_item.added',
        'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
        '',
        'event: response.function_call_arguments.done',
        'data: {"type":"response.function_call_arguments.done","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"RECOVERED\\",\\"tool_calls\\":[]}"}',
        '',
    ]

    class DummyStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            for line in self.lines:
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
            first_attempt_lines = [
                'event: response.output_item.added',
                'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
                '',
            ]
            return DummyStreamResponse(first_attempt_lines if DummyClient.calls == 1 else success_lines)

    _configure_test_key_pool(monkeypatch, settings, "sk-timeout-1", [])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(settings, "stream_recovery_retries", 1)
    monkeypatch.setattr(settings, "upstream_retry_backoff_ms", 0)
    deadlines = iter([0.0, None])
    monkeypatch.setattr(upstream, "_args_done_deadline", lambda cfg=settings: next(deadlines))
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)
    result = asyncio.run(upstream.execute_bill015(n, "exploit"))

    assert DummyClient.calls == 2
    assert result.answer == "RECOVERED"
    assert result.stream_timeout_retry_count == 1
    assert "args_done_timeout_retry:1" in result.retry_reasons


def test_execute_bill015_recovers_midstream_readtimeout(monkeypatch):
    from app import upstream
    from app.config import settings
    from app.models import NormalizedRequest

    success_lines = [
        'event: response.output_item.added',
        'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}',
        '',
        'event: response.function_call_arguments.done',
        'data: {"type":"response.function_call_arguments.done","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"READ OK\\",\\"tool_calls\\":[]}"}',
        '',
    ]

    class DummyStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __init__(self, fail):
            self.fail = fail

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            if self.fail:
                yield 'event: response.output_item.added'
                yield 'data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value"}}'
                yield ''
                raise upstream.httpx.ReadTimeout("simulated idle upstream")
            for line in success_lines:
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
            return DummyStreamResponse(fail=DummyClient.calls == 1)

    _configure_test_key_pool(monkeypatch, settings, "sk-readtimeout-1", [])
    monkeypatch.setattr(settings, "upstream_base_url", "https://example.invalid")
    monkeypatch.setattr(settings, "strict_zero", True)
    monkeypatch.setattr(settings, "stream_recovery_retries", 1)
    monkeypatch.setattr(settings, "upstream_retry_backoff_ms", 0)
    monkeypatch.setattr(upstream.httpx, "AsyncClient", DummyClient)

    n = NormalizedRequest(model="gpt-test", instructions="", user_input="x", want_stream=False, client_api="responses", is_primary_path=True)
    result = asyncio.run(upstream.execute_bill015(n, "exploit"))

    assert DummyClient.calls == 2
    assert result.answer == "READ OK"
    assert result.stream_timeout_retry_count == 1
    assert "ReadTimeout_retry:1" in result.retry_reasons


def test_reinforce_final_action_for_reasoning_only_retry():
    from app.context_compaction import reinforce_final_action

    payload = reinforce_final_action({"instructions": "base", "input": []})
    assert "immediately call the required final/action tool" in payload["instructions"]
