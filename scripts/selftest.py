from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep selftest offline and deterministic.
os.environ["LOCAL_PROXY_MODE"] = "dry-run"
os.environ["LOCAL_PROXY_LOG_DIR"] = str(ROOT / "proxy_evidence" / "selftest")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.proxy import (  # noqa: E402
    Bill015Result,
    BridgeToolCall,
    build_bill015_payload,
    build_client_tool_catalog,
    collect_bill015_result_from_events,
    normalize_responses_request,
    parse_function_arguments,
    responses_sse_generator,
)
from app.sse import parse_sse_lines  # noqa: E402


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    client = TestClient(app)

    r = client.get("/healthz")
    assert_true(r.status_code == 200, "/healthz failed")
    print("[ok] /healthz", r.json()["mode"])

    r = client.get("/v1/models")
    assert_true(r.status_code == 200 and r.json()["data"], "/v1/models failed")
    print("[ok] /v1/models", [x["id"] for x in r.json()["data"]])

    body = {"model": "codex-gpt55", "input": "Return the word OK.", "stream": False}
    r = client.post("/v1/responses", json=body)
    assert_true(r.status_code == 200, "/v1/responses dry-run non-stream failed")
    data = r.json()
    assert_true(data["mode"] == "dry-run" and data["payload"]["tool_choice"]["name"] == "emit_value", "dry-run payload wrong")
    assert_true(data["usage_estimate"]["input_tokens"] > 0, "dry-run usage estimate missing")
    print("[ok] /v1/responses dry-run non-stream")

    body["stream"] = True
    r = client.post("/v1/responses", json=body)
    assert_true(r.status_code == 200, "/v1/responses dry-run stream failed")
    text = r.text
    assert_true("response.created" in text and "response.completed" in text and "[DONE]" in text, "stream events missing")
    print("[ok] /v1/responses dry-run stream")

    r = client.post("/v1/responses/compact", json={"model":"gpt-5.5","input":"compact me","stream":False})
    assert_true(r.status_code == 200 and "CONTEXT CHECKPOINT COMPACTION" in json.dumps(r.json(), ensure_ascii=False), "/v1/responses/compact dry-run failed")
    compact_json = r.json()
    assert_true("output" in compact_json and isinstance(compact_json["output"], list) and "object" not in compact_json, "/v1/responses/compact must return native compact output JSON")
    assert_true(compact_json["output"][0]["type"] == "message" and compact_json["output"][0]["content"][0]["type"] == "output_text", "/v1/responses/compact output item shape wrong")
    print("[ok] /v1/responses/compact compatibility")

    n = normalize_responses_request({"model": "gpt-5.5", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}], "stream": True})
    payload = build_bill015_payload(n)
    assert_true(payload["tools"][0]["strict"] is True, "strict schema missing")
    assert_true(payload["tool_choice"]["name"] == "emit_value", "tool_choice missing")
    model54_n = normalize_responses_request({"model": "gpt-5.4", "input": "model switch", "stream": True})
    model54_payload = build_bill015_payload(model54_n)
    assert_true(model54_n.model == "gpt-5.4" and model54_payload["model"] == "gpt-5.4" and "GPT-5.4" not in json.dumps(model54_payload), "model passthrough/identity mapping failed")
    alias54_n = normalize_responses_request({"model": "codex-gpt54", "input": "model alias", "stream": True})
    assert_true(alias54_n.model == "gpt-5.4", "codex-gpt54 alias failed")
    print("[ok] BILL-015 payload builder/model passthrough")

    codex_md = json.dumps({"request_kind":"compaction","compaction":{"trigger":"auto","reason":"context_limit","implementation":"responses","phase":"mid_turn","strategy":"memento"}})
    comp_n = normalize_responses_request({
        "model":"gpt-5.5",
        "instructions":"root instructions",
        "input":[
            {"type":"message","role":"developer","content":[{"type":"input_text","text":"<skills_instructions>skill list here</skills_instructions>"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"Earlier task"}]},
            {"type":"function_call","name":"shell_command","arguments":"{\"command\":\"pwd\"}","call_id":"call_x"},
            {"type":"function_call_output","call_id":"call_x","output":"ok"},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary."}]},
        ],
        "tools":[],
        "parallel_tool_calls":False,
        "stream":True,
        "client_metadata":{"x-codex-turn-metadata": codex_md},
    })
    assert_true(comp_n.is_compaction and comp_n.request_kind == "compaction", "compaction detection failed")
    assert_true("<skills_instructions>" in comp_n.instructions, "developer/skills context not preserved")
    comp_payload = build_bill015_payload(comp_n)
    assert_true("CONTEXT CHECKPOINT COMPACTION" in comp_payload["instructions"], "compaction system prompt missing")
    assert_true("<skills_instructions>" in json.dumps(comp_payload, ensure_ascii=False), "compaction did not forward native developer/skills context")
    assert_true("Do not request tools" in comp_payload["instructions"], "compaction tool prohibition missing")
    assert_true("Codex native tool catalog" not in comp_payload["instructions"], "compaction included normal tool catalog")
    assert_true(comp_n.estimated_input_tokens > 0, "usage estimate missing")
    assert_true(comp_n.usage_estimate.cached_tokens >= int(comp_n.usage_estimate.input_tokens * 0.80), "compaction cached usage too low")
    print("[ok] native compaction/skills preservation")

    plain_usage_n = normalize_responses_request({"model": "gpt-5.5", "input": "hello world"})
    zh_usage_n = normalize_responses_request({"model": "gpt-5.5", "input": "这是一个用于测试 token 估算的中文长文本。" * 20})
    tools_usage_n = normalize_responses_request({"model": "gpt-5.5", "input": "hello world", "tools": [{
        "type": "function",
        "name": "shell_command",
        "description": "run shell",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False},
    }]})
    long_output_n = normalize_responses_request({"model": "gpt-5.5", "input": [
        {"type": "function_call", "name": "shell_command", "call_id": "call_long", "arguments": "{\"command\":\"big\"}"},
        {"type": "function_call_output", "call_id": "call_long", "output": "Exit code: 0\n" + ("0123456789abcdef\n" * 2000)},
    ]})
    tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    image_usage_n = normalize_responses_request({"model": "gpt-5.5", "input": [{
        "type": "message",
        "role": "user",
        "content": [{"type": "input_image", "image_url": "data:image/png;base64," + tiny_png, "detail": "high"}],
    }]})
    assert_true(plain_usage_n.estimated_input_tokens > 0, "plain usage missing")
    assert_true(zh_usage_n.estimated_input_tokens > plain_usage_n.estimated_input_tokens, "Chinese usage under-estimated")
    assert_true(tools_usage_n.estimated_input_tokens > plain_usage_n.estimated_input_tokens, "tools schema usage missing")
    assert_true(long_output_n.usage_estimate.tool_history_tokens > plain_usage_n.estimated_input_tokens, "long tool output usage missing")
    assert_true(image_usage_n.usage_estimate.image_tokens > 0, "image usage missing")
    print("[ok] structured usage estimator")

    shell_loop_n = normalize_responses_request({
        "model": "gpt-5.5",
        "input": [
            {"type": "function_call", "name": "shell_command", "call_id": "call_1", "arguments": "{\"command\":\"pwd\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "Exit code: 0\nWall time: 0.1s\nOutput:\nS:\\hack\\packyapi.com"},
        ],
    })
    assert_true("shell_command" in shell_loop_n.latest_tool_summary, "shell_command missing from tool feedback")
    assert_true("Exit code: 0" in shell_loop_n.latest_tool_summary, "shell exit code missing from tool feedback")
    assert_true(shell_loop_n.latest_tool_failed is False, "successful shell marked failed")
    shell_payload = build_bill015_payload(shell_loop_n)
    assert_true("Recent local Codex tool results" in shell_payload["input"][0]["content"] and shell_payload["input"][-1]["content"].startswith("Current user request:"), "tool feedback/current request layout wrong")
    print("[ok] shell_command tool-loop feedback")

    patch_loop_n = normalize_responses_request({
        "model": "gpt-5.5",
        "input": [
            {"type": "custom_tool_call", "name": "apply_patch", "call_id": "call_patch", "input": "*** Begin Patch\n*** Add File: x.txt\n+ok\n*** End Patch\n"},
            {"type": "custom_tool_call_output", "call_id": "call_patch", "output": "Success. Updated the following files:\nA x.txt"},
        ],
    })
    assert_true("apply_patch" in patch_loop_n.latest_tool_summary and "Success. Updated" in patch_loop_n.latest_tool_summary, "apply_patch feedback missing")
    print("[ok] apply_patch tool-loop feedback")

    failed_loop_n = normalize_responses_request({
        "model": "gpt-5.5",
        "input": [
            {"type": "function_call", "name": "shell_command", "call_id": "call_fail", "arguments": "{\"command\":\"python missing.py\"}"},
            {"type": "function_call_output", "call_id": "call_fail", "output": "Exit code: 1\nTraceback (most recent call last):\nFileNotFoundError: missing.py"},
        ],
    })
    assert_true(failed_loop_n.latest_tool_failed is True, "failed tool not detected")
    assert_true("retry with corrected arguments" in failed_loop_n.latest_tool_summary, "failure guidance missing")
    print("[ok] failed tool-loop feedback")

    pending_loop_n = normalize_responses_request({
        "model": "gpt-5.5",
        "input": [
            {"type": "function_call", "name": "shell_command", "call_id": "call_pending", "arguments": "{\"command\":\"sleep 1\"}"},
        ],
    })
    assert_true(pending_loop_n.pending_tool_call_count == 1, "pending call not detected")
    assert_true("Do not assume their result" in pending_loop_n.latest_tool_summary, "pending guidance missing")
    print("[ok] pending tool-loop feedback")

    answer, malformed, repaired, mode, calls = parse_function_arguments('{"mode":"answer","answer":"OK","tool_calls":[]}')
    assert_true(answer == "OK" and mode == "answer" and not calls and not malformed and not repaired, "argument parser failed")
    t_answer, _, _, t_mode, t_calls = parse_function_arguments(
        '{"mode":"tool_call","answer":"","tool_calls":[{"name":"shell_command","arguments":"{\\\"command\\\":\\\"pwd\\\"}"}]}',
        tool_registry={"shell_command": {"call_type": "function", "output_name": "shell_command", "raw_type": "function"}},
    )
    assert_true(t_mode == "tool_call" and len(t_calls) == 1 and t_calls[0].name == "shell_command", "tool_call parser failed")
    print("[ok] function argument parser")

    tools = [
        {"type": "function", "name": "shell_command", "description": "run shell", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}},
        {"type": "custom", "name": "apply_patch", "description": "patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}},
        {"type": "namespace", "name": "codex_app", "tools": [{"type": "function", "name": "read_thread_terminal", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}]},
        {"type": "namespace", "name": "mcp__playwright", "tools": [
            {"type": "function", "name": "browser_tabs", "parameters": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False}},
            {"type": "function", "name": "browser_navigate", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"], "additionalProperties": False}},
        ]},
        {"type": "namespace", "name": "mcp__jshook", "tools": [
            {"type": "function", "name": "call_tool", "parameters": {"type": "object", "properties": {"server": {"type": "string"}, "tool": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["server", "tool", "arguments"], "additionalProperties": False}},
        ]},
        {"type": "tool_search", "execution": "client", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "number"}}, "required": ["query"], "additionalProperties": False}},
        {"type": "web_search", "external_web_access": True, "search_content_types": ["text", "image"]},
    ]
    catalog, registry = build_client_tool_catalog(tools)
    assert_true('"type":"custom"' in catalog and 'codex_app.read_thread_terminal' in catalog and 'mcp__playwright.browser_tabs' in catalog and '"native_call"' in catalog, "tool catalog lost custom/namespace tools")
    assert_true('"type":"tool_search"' in catalog and registry["tool_search"]["call_type"] == "tool_search", "tool_search registry wrong")
    assert_true('"type":"web_search"' in catalog and registry["web_search"]["call_type"] == "web_search", "web_search registry wrong")
    custom_answer, _, _, custom_mode, custom_calls = parse_function_arguments(json.dumps({
        "mode": "tool_call",
        "answer": "",
        "tool_calls": [{"type": "custom", "name": "apply_patch", "arguments": "", "input": "*** Begin Patch\n*** Add File: x.txt\n+ok\n*** End Patch\n"}],
    }), tool_registry=registry)
    assert_true(custom_mode == "tool_call" and custom_calls[0].call_type == "custom" and custom_calls[0].arguments.startswith("*** Begin Patch"), "custom tool_call parser failed")
    parallel_answer, _, _, parallel_mode, parallel_calls = parse_function_arguments(json.dumps({
        "mode": "tool_call",
        "answer": "",
        "tool_calls": [
            {"type": "function", "name": "shell_command", "arguments": "{\"command\":\"pwd\"}", "input": ""},
            {"type": "function", "name": "codex_app.read_thread_terminal", "arguments": "{}", "input": ""},
        ],
    }), tool_registry=registry)
    assert_true(parallel_mode == "tool_call" and len(parallel_calls) == 2 and parallel_calls[1].name == "read_thread_terminal" and parallel_calls[1].namespace == "codex_app", "parallel/namespaced tool parser failed")
    _, _, _, mcp_mode, mcp_calls = parse_function_arguments(json.dumps({
        "mode": "tool_call",
        "answer": "",
        "tool_calls": [
            {"type": "function", "namespace": "mcp__playwright", "name": "browser_tabs", "arguments": "{\"action\":\"list\"}", "input": ""},
            {"type": "function", "namespace": "", "name": "mcp__jshook.call_tool", "arguments": "{\"server\":\"x\",\"tool\":\"y\",\"arguments\":{}}", "input": ""},
            {"type": "function", "namespace": "", "name": "mcp__node_repl__js", "arguments": "{\"code\":\"1+1\"}", "input": ""},
        ],
    }), tool_registry=registry | {"mcp__node_repl.js": {"call_type": "function", "output_name": "js", "namespace": "mcp__node_repl", "raw_type": "function"}})
    mcp_pairs = [(c.namespace, c.name) for c in mcp_calls if c.call_type != "tool_search"]
    mcp_search_queries = "\n".join(c.arguments for c in mcp_calls if c.call_type == "tool_search")
    assert_true(
        mcp_mode == "tool_call"
        and mcp_pairs == [("mcp__playwright", "browser_tabs"), ("mcp__jshook", "call_tool"), ("mcp__node_repl", "js")]
        and not mcp_search_queries,
        "generic MCP namespace parser/deferred expansion failed",
    )
    _, _, _, web_rewrite_mode, web_rewrite_calls = parse_function_arguments(json.dumps({
        "mode":"tool_call",
        "answer":"",
        "tool_calls":[{"type":"web_search","name":"web_search","arguments":"{\"query\":\"QuantumNous new-api responses compact\"}","input":""}],
    }), tool_registry=registry)
    assert_true(web_rewrite_mode == "tool_call" and web_rewrite_calls[0].call_type == "tool_search" and web_rewrite_calls[0].name == "tool_search" and "local web search" in web_rewrite_calls[0].arguments, "web_search was not rewritten to local tool_search")
    browser_answer, _, _, browser_mode, browser_calls = parse_function_arguments(json.dumps({
        "mode":"tool_call",
        "answer":"",
        "tool_calls":[{"type":"tool_search","name":"tool_search","arguments":"{\"query\":\"Playwright browser cookies DOM network tools\",\"limit\":8}","input":""}],
    }), tool_registry=registry)
    browser_queries = "\n".join(c.arguments for c in browser_calls if c.call_type == "tool_search")
    assert_true(browser_mode == "tool_call" and len([c for c in browser_calls if c.call_type == "tool_search"]) <= 2 and "playwright browser navigate evaluate" in browser_queries, "browser/MCP tool_search expansion failed")
    deferred_n = normalize_responses_request({"model":"gpt-5.5","input":[{"type":"tool_search_output","call_id":"call_ts","status":"completed","execution":"client","tools":[{"type":"namespace","name":"mcp__node_repl","tools":[{"type":"function","name":"js","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"],"additionalProperties":False}}]}]}],"tools":tools})
    assert_true("mcp__node_repl.js" in deferred_n.tools_catalog and "js" in deferred_n.tool_registry, "deferred tool_search_output tools not cataloged")
    assert_true("mcp__node_repl.js" in deferred_n.latest_tool_summary, "deferred tools missing from tool feedback")
    print("[ok] native tool catalog custom/namespace/parallel/deferred parser")

    async def collect_tool_stream() -> str:
        nr = normalize_responses_request({"model": "gpt-5.5", "input": "patch", "stream": True, "tools": tools})
        result = Bill015Result(local_request_id="resp_local_test", bridge_mode="tool_call", tool_calls=[
            BridgeToolCall(id="call_customtest", name="apply_patch", arguments="*** Begin Patch\n*** Add File: x.txt\n+ok\n*** End Patch\n", call_type="custom"),
            BridgeToolCall(id="call_functest", name="shell_command", arguments="{\"command\":\"pwd\"}", call_type="function"),
            BridgeToolCall(id="call_mcptest", name="browser_tabs", namespace="mcp__playwright", arguments="{\"action\":\"list\"}", call_type="function"),
            BridgeToolCall(id="call_toolsearchtest", name="tool_search", arguments="{\"query\":\"node_repl js\",\"limit\":8}", call_type="tool_search"),
            BridgeToolCall(id="call_websearchtest", name="tool_search", arguments="{\"query\":\"local web search/fetch using browser chrome node_repl playwright curl PowerShell: QuantumNous new-api\",\"limit\":8}", call_type="tool_search"),
        ], args_done_seen=True)
        chunks = []
        async for chunk in responses_sse_generator(asyncio.sleep(0, result), nr, "resp_local_test"):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    stream_text = asyncio.run(collect_tool_stream())
    assert_true("response.custom_tool_call_input.delta" in stream_text and "response.function_call_arguments.delta" in stream_text and '"type":"custom_tool_call"' in stream_text, "native tool stream events missing")
    assert_true('"namespace":"mcp__playwright"' in stream_text and '"name":"browser_tabs"' in stream_text and '"name":"mcp__playwright.browser_tabs"' not in stream_text, "native MCP namespace stream format wrong")
    assert_true('"type":"tool_search_call"' in stream_text and '"execution":"client"' in stream_text, "native tool_search_call stream missing")
    assert_true('"type":"web_search_call"' not in stream_text, "web_search should be rewritten to local tool_search, not emitted as web_search_call")
    assert_true(stream_text.count('"type":"tool_search_call"') >= 2, "rewritten web_search did not produce a second local tool_search_call")
    assert_true('"usage"' in stream_text and '"input_tokens":' in stream_text and '"input_tokens_details":' in stream_text and '"output_tokens_details":' in stream_text, "synthetic usage details missing from response.completed")
    tool_events = list(parse_sse_lines(stream_text.splitlines(True)))
    tool_objs = [ev.json for ev in tool_events if ev.json]
    assert_true([obj["sequence_number"] for obj in tool_objs] == list(range(len(tool_objs))), "Responses sequence_number not contiguous")
    added_indexes = [obj["output_index"] for obj in tool_objs if obj["type"] == "response.output_item.added"]
    done_indexes = [obj["output_index"] for obj in tool_objs if obj["type"] == "response.output_item.done"]
    assert_true(added_indexes == [0, 1, 2, 3, 4] and done_indexes == [0, 1, 2, 3, 4], "parallel output_index sequence wrong")
    completed = [obj for obj in tool_objs if obj["type"] == "response.completed"][-1]["response"]
    assert_true(len(completed["output"]) == 5 and completed["output"][0]["type"] == "custom_tool_call" and completed["output"][2]["namespace"] == "mcp__playwright" and completed["output"][3]["type"] == "tool_search_call", "completed output items wrong")
    assert_true({"error", "incomplete_details", "reasoning", "text", "metadata"}.issubset(completed.keys()), "completed response native fields missing")
    print("[ok] native Responses tool-call stream synthesis")

    async def collect_message_stream() -> str:
        nr = normalize_responses_request({"model": "gpt-5.5", "input": "answer", "stream": True})
        result = Bill015Result(local_request_id="resp_local_msgtest", answer="hello world", args_done_seen=True)
        chunks = []
        async for chunk in responses_sse_generator(asyncio.sleep(0, result), nr, "resp_local_msgtest"):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    message_events = list(parse_sse_lines(asyncio.run(collect_message_stream()).splitlines(True)))
    message_types = [ev.json["type"] for ev in message_events if ev.json]
    assert_true(message_types[:2] == ["response.created", "response.output_item.added"], "message lifecycle prefix wrong")
    assert_true("response.output_text.delta" in message_types and "response.output_item.done" in message_types and message_types[-1] == "response.completed", "message lifecycle missing events")
    assert_true(message_events[-1].data == "[DONE]", "message stream missing DONE")
    print("[ok] native Responses message lifecycle")

    async def collect_failed_stream() -> str:
        async def fail():
            raise HTTPException(status_code=502, detail="boom")
        nr = normalize_responses_request({"model": "gpt-5.5", "input": "fail", "stream": True})
        chunks = []
        async for chunk in responses_sse_generator(fail(), nr, "resp_local_failtest"):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    failed_events = list(parse_sse_lines(asyncio.run(collect_failed_stream()).splitlines(True)))
    failed_types = [ev.json["type"] for ev in failed_events if ev.json]
    assert_true(failed_types[-2:] == ["response.failed", "error"] and failed_events[-1].data == "[DONE]", "failed lifecycle wrong")
    print("[ok] native Responses failed lifecycle")

    async def collect_generic_error_stream() -> str:
        async def fail_runtime():
            raise RuntimeError("synthetic upstream timeout")
        nr = normalize_responses_request({"model": "gpt-5.5", "input": "fail", "stream": True})
        chunks = []
        async for chunk in responses_sse_generator(fail_runtime(), nr, "resp_local_runtimefail"):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    generic_failed_events = list(parse_sse_lines(asyncio.run(collect_generic_error_stream()).splitlines(True)))
    generic_failed_types = [ev.json["type"] for ev in generic_failed_events if ev.json]
    assert_true(generic_failed_types[-2:] == ["response.failed", "error"] and generic_failed_events[-1].data == "[DONE]", "generic exception stream did not close cleanly")
    print("[ok] generic upstream exception stream closes cleanly")

    async def collect_heartbeat_stream() -> str:
        old_interval = settings.client_heartbeat_interval_ms
        settings.client_heartbeat_interval_ms = 50
        async def slow_result():
            await asyncio.sleep(0.12)
            return Bill015Result(local_request_id="resp_local_slow", answer="slow ok", args_done_seen=True)
        nr = normalize_responses_request({"model": "gpt-5.5", "input": "slow", "stream": True})
        chunks = []
        try:
            async for chunk in responses_sse_generator(slow_result(), nr, "resp_local_slow"):
                chunks.append(chunk.decode("utf-8"))
        finally:
            settings.client_heartbeat_interval_ms = old_interval
        return "".join(chunks)

    heartbeat_text = asyncio.run(collect_heartbeat_stream())
    assert_true(": keep-alive" in heartbeat_text and "response.completed" in heartbeat_text, "heartbeat stream missing keep-alive or completion")
    print("[ok] streaming keep-alive heartbeat")

    events = list(parse_sse_lines([
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"answer\\\":\"}\n",
        "\n",
        "event: response.function_call_arguments.done\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"mode\\\":\\\"answer\\\",\\\"answer\\\":\\\"OK\\\",\\\"tool_calls\\\":[]}\"}\n",
        "\n",
    ]))
    assert_true(len(events) == 2 and events[1].json["type"] == "response.function_call_arguments.done", "SSE parser failed")
    print("[ok] SSE parser")

    mock_lines = [
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_up_mock\"}}\n",
        "\n",
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"name\":\"emit_value\"}}\n",
        "\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"answer\\\":\\\"O\"}\n",
        "\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"K\\\"}\"}\n",
        "\n",
        "event: response.function_call_arguments.done\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"mode\\\":\\\"answer\\\",\\\"answer\\\":\\\"OK\\\",\\\"tool_calls\\\":[]}\"}\n",
        "\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\"}\n",
        "\n",
    ]
    mock_result = collect_bill015_result_from_events(parse_sse_lines(mock_lines), n)
    assert_true(mock_result.answer == "OK" and mock_result.args_done_seen and mock_result.aborted and not mock_result.upstream_completed_seen, "mock abort state failed")
    print("[ok] mock BILL-015 args_done abort state")

    r = client.post("/v1/chat/completions", json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "Return OK"}], "stream": False})
    assert_true(r.status_code == 200 and r.json()["choices"], "/v1/chat/completions dry-run failed")
    print("[ok] /v1/chat/completions compatibility")

    print("[done] offline selftest passed")


if __name__ == "__main__":
    main()
