from __future__ import annotations

import asyncio

from app.ingest.catalog import build_catalog
from app.protocol.models import BridgeToolCall, Turn, TurnResult
from app.protocol.sse import parse_sse_lines
from app.replay.events import responses_sse_generator


def _turn() -> Turn:
    return Turn(
        model="gpt-5.5",
        want_stream=True,
        client_api="responses",
        items=[],
        current_user="edit file",
        catalog=build_catalog([{"type": "custom", "name": "apply_patch"}, {"type": "tool_search", "name": "tool_search"}], []),
    )


async def _collect(result: TurnResult) -> list[str]:
    chunks: list[str] = []
    async for raw in responses_sse_generator(_ready(result), _turn(), result.local_request_id):
        chunks.append(raw.decode("utf-8"))
    return chunks


async def _ready(value: TurnResult) -> TurnResult:
    return value


def test_replay_ends_with_completed_and_dispatches_on_output_item_done():
    result = TurnResult(
        local_request_id="resp_local_test",
        commentary="editing",
        tool_calls=[
            BridgeToolCall(id="call_patch", name="apply_patch", arguments="", call_type="custom", input="*** Begin Patch\n*** End Patch"),
            BridgeToolCall(id="call_search", name="tool_search", arguments="", call_type="tool_search", execution="client", search_arguments={"query": "browser", "limit": 8}),
        ],
    )
    blob = "".join(asyncio.run(_collect(result)))
    events = list(parse_sse_lines(blob.splitlines()))
    types = [(ev.json or {}).get("type") for ev in events if ev.json]
    assert types[0] == "response.created"
    assert "response.output_item.done" in types
    assert types[-1] == "response.completed"
    assert "response.custom_tool_call_input.delta" in types
    search_done = next(ev.json for ev in events if ev.json and ev.json.get("type") == "response.output_item.done" and (ev.json.get("item") or {}).get("type") == "tool_search_call")
    assert search_done["item"]["execution"] == "client"
    assert isinstance(search_done["item"]["arguments"], dict)
    patch_done = next(ev.json for ev in events if ev.json and ev.json.get("type") == "response.output_item.done" and (ev.json.get("item") or {}).get("type") == "custom_tool_call")
    assert patch_done["item"]["input"].startswith("*** Begin Patch")
    msg_ids = [(ev.json or {}).get("item", {}).get("id") for ev in events if (ev.json or {}).get("type") == "response.output_item.done"]
    for item_id in msg_ids:
        if not item_id:
            continue
        assert not str(item_id).startswith("resp_local_"), item_id
    message_ids = [
        (ev.json or {}).get("item", {}).get("id")
        for ev in events
        if (ev.json or {}).get("type") == "response.output_item.done" and (ev.json or {}).get("item", {}).get("type") == "message"
    ]
    assert message_ids and all(str(i).startswith("msg") for i in message_ids)
    assert str(patch_done["item"]["id"]).startswith("ctc")
    assert str(search_done["item"]["id"]).startswith("tsc")


def test_functions_dot_name_is_split_and_namespace_only_is_dropped():
    from app.bridge.parse import parse_emit_value
    from app.ingest.catalog import build_catalog
    from app.protocol.ids import coerce_item_id

    catalog = build_catalog(
        [{"type": "function", "name": "exec", "namespace": "functions", "description": "run a command"}],
        [],
        query="run",
    )
    wrapper = parse_emit_value(
        '{"mode":"tool_call","answer":"","tool_calls":[{"type":"function","name":"functions.exec","arguments":"{\\"cmd\\":\\"ls\\"}"}]}',
        catalog,
    )
    assert wrapper.tool_calls
    assert wrapper.tool_calls[0].name == "exec"
    assert wrapper.tool_calls[0].namespace == "functions"
    dropped = parse_emit_value(
        '{"mode":"tool_call","answer":"","tool_calls":[{"type":"function","name":"functions","arguments":"{}"}]}',
        catalog,
    )
    assert dropped.tool_calls == []
    coerced = coerce_item_id({"id": "resp_local_cc6a52f32f5f4ed4ada6cc34f672057c_msg_0", "type": "message", "role": "assistant"})
    assert coerced["id"].startswith("msg")
    assert not coerced["id"].startswith("resp_local_")
