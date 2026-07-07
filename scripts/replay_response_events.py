from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Awaitable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOCAL_PROXY_MODE", "dry-run")
os.environ.setdefault("LOCAL_PROXY_LOG_DIR", str(ROOT / "proxy_evidence" / "event_replay"))

from fastapi import HTTPException  # noqa: E402

from app.models import Bill015Result, BridgeToolCall  # noqa: E402
from app.proxy import normalize_responses_request  # noqa: E402
from app.response_events import response_json, responses_sse_generator  # noqa: E402
from app.sse import parse_sse_lines  # noqa: E402


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


async def collect(result_coro: Awaitable[Bill015Result], rid: str = "resp_local_eventtest") -> tuple[str, list[Any]]:
    n = normalize_responses_request({"model": "gpt-5.5", "input": "event replay", "stream": True})
    chunks: list[str] = []
    async for chunk in responses_sse_generator(result_coro, n, rid):
        chunks.append(chunk.decode("utf-8"))
    text = "".join(chunks)
    events = list(parse_sse_lines(text.splitlines(True)))
    return text, events


async def ok_result(result: Bill015Result) -> Bill015Result:
    return result


async def fail_result() -> Bill015Result:
    raise HTTPException(status_code=502, detail="boom")


def json_events(events: list[Any]) -> list[dict[str, Any]]:
    return [ev.json for ev in events if ev.json]


def types(events: list[Any]) -> list[str]:
    return [obj["type"] for obj in json_events(events)]


def assert_done_and_sequence(events: list[Any]) -> None:
    assert_true(events[-1].data == "[DONE]", "last frame is not [DONE]")
    objs = json_events(events)
    seq = [obj["sequence_number"] for obj in objs]
    assert_true(seq == list(range(len(seq))), f"sequence_number not contiguous: {seq}")


def completed_response(events: list[Any]) -> dict[str, Any]:
    for obj in reversed(json_events(events)):
        if obj["type"] in {"response.completed", "response.incomplete", "response.failed"}:
            return obj["response"]
    raise AssertionError("terminal response event missing")


async def main() -> None:
    text, events = await collect(ok_result(Bill015Result(local_request_id="resp_local_eventtest", answer="hello", args_done_seen=True)))
    t = types(events)
    assert_done_and_sequence(events)
    expected_prefix = ["response.created", "response.in_progress", "response.output_item.added", "response.content_part.added"]
    assert_true(t[:4] == expected_prefix, "message lifecycle prefix wrong")
    assert_true("response.output_text.delta" in t and "response.output_text.done" in t and "response.content_part.done" in t, "message text lifecycle missing")
    assert_true(t[-1] == "response.completed", "message terminal event wrong")
    assert_true(completed_response(events)["output"][0]["type"] == "message", "message completed output wrong")
    print("[ok] message lifecycle")

    calls = [
        BridgeToolCall(id="call_func", name="shell_command", arguments="{\"command\":\"pwd\"}", call_type="function"),
        BridgeToolCall(id="call_custom", name="apply_patch", arguments="*** Begin Patch\n*** End Patch\n", call_type="custom"),
        BridgeToolCall(id="call_search", name="tool_search", arguments="{\"query\":\"node_repl js\",\"limit\":8}", call_type="tool_search"),
    ]
    tool_text, events = await collect(ok_result(Bill015Result(local_request_id="resp_local_eventtest", bridge_mode="tool_call", tool_calls=calls, args_done_seen=True)))
    t = types(events)
    assert_done_and_sequence(events)
    assert_true("response.function_call_arguments.delta" in t and "response.function_call_arguments.done" in t, "function_call lifecycle missing")
    assert_true("response.custom_tool_call_input.delta" in t and "response.custom_tool_call_input.done" in t, "custom_tool_call lifecycle missing")
    assert_true(t.count("response.output_item.added") == 3 and t.count("response.output_item.done") == 3, "parallel output item count wrong")
    assert_true("response.web_search_call" not in tool_text, "web_search event should not appear")
    completed = completed_response(events)
    assert_true([item["type"] for item in completed["output"]] == ["function_call", "custom_tool_call", "tool_search_call"], "completed output types wrong")
    assert_true([obj["output_index"] for obj in json_events(events) if obj["type"] == "response.output_item.added"] == [0, 1, 2], "output_index sequence wrong")
    assert_true("input_tokens_details" in completed["usage"] and "output_tokens_details" in completed["usage"], "usage details missing")
    print("[ok] tool lifecycles and parallel outputs")

    _, events = await collect(fail_result())
    t = types(events)
    assert_done_and_sequence(events)
    assert_true(t[-2:] == ["response.failed", "error"], "failed lifecycle wrong")
    assert_true(completed_response(events)["status"] == "failed", "failed response status wrong")
    print("[ok] failed lifecycle")

    _, events = await collect(ok_result(Bill015Result(local_request_id="resp_local_eventtest", answer="x\n[local proxy truncated answer at max_answer_chars]", args_done_seen=True)))
    t = types(events)
    assert_done_and_sequence(events)
    assert_true(t[-1] == "response.incomplete", "incomplete terminal event missing")
    assert_true(completed_response(events)["incomplete_details"]["reason"] == "max_output_tokens", "incomplete reason wrong")
    print("[ok] incomplete lifecycle")

    n = normalize_responses_request({"model": "gpt-5.5", "input": "json parity", "stream": False})
    result = Bill015Result(local_request_id="resp_local_jsonparity", answer="hello", args_done_seen=True)
    obj = response_json(result, n)
    assert_true({"error", "incomplete_details", "reasoning", "text", "metadata"}.issubset(obj.keys()), "response_json missing native fields")
    print("[ok] non-stream response object fields")

    print("[done] response event replay passed")


if __name__ == "__main__":
    asyncio.run(main())
