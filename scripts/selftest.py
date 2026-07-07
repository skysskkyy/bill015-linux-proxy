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

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.proxy import Bill015Result, BridgeToolCall, build_bill015_payload, build_client_tool_catalog, collect_bill015_result_from_events, normalize_responses_request, parse_function_arguments, responses_sse_generator  # noqa: E402
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
    print("[ok] /v1/responses dry-run non-stream")

    body["stream"] = True
    r = client.post("/v1/responses", json=body)
    assert_true(r.status_code == 200, "/v1/responses dry-run stream failed")
    text = r.text
    assert_true("response.created" in text and "response.completed" in text and "[DONE]" in text, "stream events missing")
    print("[ok] /v1/responses dry-run stream")

    r = client.post("/v1/responses/compact", json={"model":"gpt-5.5","input":"compact me","stream":False})
    assert_true(r.status_code == 200 and "CONTEXT CHECKPOINT COMPACTION" in json.dumps(r.json(), ensure_ascii=False), "/v1/responses/compact dry-run failed")
    print("[ok] /v1/responses/compact compatibility")

    n = normalize_responses_request({"model": "gpt-5.5", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}], "stream": True})
    payload = build_bill015_payload(n)
    assert_true(payload["tools"][0]["strict"] is True, "strict schema missing")
    assert_true(payload["tool_choice"]["name"] == "emit_value", "tool_choice missing")
    print("[ok] BILL-015 payload builder")

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
    assert_true("CONTEXT CHECKPOINT COMPACTION" in comp_payload["input"][0]["content"], "compaction system prompt missing")
    assert_true("Do not request tools" in comp_payload["input"][0]["content"], "compaction tool prohibition missing")
    assert_true("Codex native tool catalog" not in comp_payload["input"][0]["content"], "compaction included normal tool catalog")
    assert_true(comp_n.estimated_input_tokens > 0, "usage estimate missing")
    print("[ok] native compaction/skills preservation")

    answer, malformed, repaired, mode, calls = parse_function_arguments('{"mode":"answer","answer":"OK","tool_calls":[]}')
    assert_true(answer == "OK" and mode == "answer" and not calls and not malformed and not repaired, "argument parser failed")
    t_answer, _, _, t_mode, t_calls = parse_function_arguments('{"mode":"tool_call","answer":"","tool_calls":[{"name":"shell_command","arguments":"{\\\"command\\\":\\\"pwd\\\"}"}]}')
    assert_true(t_mode == "tool_call" and len(t_calls) == 1 and t_calls[0].name == "shell_command", "tool_call parser failed")
    print("[ok] function argument parser")

    tools = [
        {"type": "function", "name": "shell_command", "description": "run shell", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}},
        {"type": "custom", "name": "apply_patch", "description": "patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}},
        {"type": "namespace", "name": "codex_app", "tools": [{"type": "function", "name": "read_thread_terminal", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}]},
        {"type": "tool_search", "execution": "client", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "number"}}, "required": ["query"], "additionalProperties": False}},
        {"type": "web_search", "external_web_access": True, "search_content_types": ["text", "image"]},
    ]
    catalog, registry = build_client_tool_catalog(tools)
    assert_true('"type":"custom"' in catalog and 'codex_app.read_thread_terminal' in catalog, "tool catalog lost custom/namespace tools")
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
    assert_true(parallel_mode == "tool_call" and len(parallel_calls) == 2 and parallel_calls[1].name == "codex_app.read_thread_terminal", "parallel/namespaced tool parser failed")
    _, _, _, web_rewrite_mode, web_rewrite_calls = parse_function_arguments(json.dumps({
        "mode":"tool_call",
        "answer":"",
        "tool_calls":[{"type":"web_search","name":"web_search","arguments":"{\"query\":\"QuantumNous new-api responses compact\"}","input":""}],
    }), tool_registry=registry)
    assert_true(web_rewrite_mode == "tool_call" and web_rewrite_calls[0].call_type == "tool_search" and web_rewrite_calls[0].name == "tool_search" and "local web search" in web_rewrite_calls[0].arguments, "web_search was not rewritten to local tool_search")
    deferred_n = normalize_responses_request({"model":"gpt-5.5","input":[{"type":"tool_search_output","call_id":"call_ts","status":"completed","execution":"client","tools":[{"type":"namespace","name":"mcp__node_repl","tools":[{"type":"function","name":"js","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"],"additionalProperties":False}}]}]}],"tools":tools})
    assert_true("mcp__node_repl.js" in deferred_n.tools_catalog and "js" in deferred_n.tool_registry, "deferred tool_search_output tools not cataloged")
    print("[ok] native tool catalog custom/namespace/parallel/deferred parser")

    async def collect_tool_stream() -> str:
        nr = normalize_responses_request({"model": "gpt-5.5", "input": "patch", "stream": True, "tools": tools})
        result = Bill015Result(local_request_id="resp_local_test", bridge_mode="tool_call", tool_calls=[
            BridgeToolCall(id="call_customtest", name="apply_patch", arguments="*** Begin Patch\n*** Add File: x.txt\n+ok\n*** End Patch\n", call_type="custom"),
            BridgeToolCall(id="call_functest", name="shell_command", arguments="{\"command\":\"pwd\"}", call_type="function"),
            BridgeToolCall(id="call_toolsearchtest", name="tool_search", arguments="{\"query\":\"node_repl js\",\"limit\":8}", call_type="tool_search"),
            BridgeToolCall(id="call_websearchtest", name="tool_search", arguments="{\"query\":\"local web search/fetch using browser chrome node_repl playwright curl PowerShell: QuantumNous new-api\",\"limit\":8}", call_type="tool_search"),
        ], args_done_seen=True)
        chunks = []
        async for chunk in responses_sse_generator(asyncio.sleep(0, result), nr, "resp_local_test"):
            chunks.append(chunk.decode("utf-8"))
        return "".join(chunks)

    stream_text = asyncio.run(collect_tool_stream())
    assert_true("response.custom_tool_call_input.delta" in stream_text and '"type":"custom_tool_call"' in stream_text and "response.function_call_arguments.done" in stream_text, "native tool stream events missing")
    assert_true('"type":"tool_search_call"' in stream_text and '"execution":"client"' in stream_text, "native tool_search_call stream missing")
    assert_true('"type":"web_search_call"' not in stream_text, "web_search should be rewritten to local tool_search, not emitted as web_search_call")
    assert_true(stream_text.count('"type":"tool_search_call"') >= 2, "rewritten web_search did not produce a second local tool_search_call")
    assert_true('"usage"' in stream_text and '"input_tokens":' in stream_text, "synthetic usage missing from response.completed")
    print("[ok] native Responses tool-call stream synthesis")

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


