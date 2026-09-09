from __future__ import annotations

from app.context.pack import LOSS_PREFIX, pack_turn_items
from app.ingest.catalog import build_catalog
from app.protocol.models import Turn


def test_pack_keeps_current_user_and_latest_tool_batch():
    items = []
    for i in range(12):
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"old question {i} " * 40}]})
        items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"old answer {i} " * 40}]})
    items.append({"type": "function_call", "name": "shell_command", "call_id": "c-old", "arguments": "{}"})
    items.append({"type": "function_call_output", "call_id": "c-old", "output": "obsolete " * 200})
    items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "CURRENT TASK: fix the parser"}]})
    items.append({"type": "function_call", "name": "shell_command", "call_id": "c-new", "arguments": "{\"command\":\"rg parser\"}"})
    items.append({"type": "function_call_output", "call_id": "c-new", "output": "parser.rs:12: fn parse()"})
    turn = Turn(
        model="gpt-5.5",
        want_stream=True,
        client_api="responses",
        items=items,
        current_user="CURRENT TASK: fix the parser",
        catalog=build_catalog([], items),
    )
    packed = pack_turn_items(turn)
    blob = str(packed.items)
    assert "CURRENT TASK: fix the parser" in blob
    assert "parser.rs:12" in blob
    assert packed.tokens > 0
    user_positions = [i for i, item in enumerate(packed.items) if "CURRENT TASK" in str(item)]
    tool_positions = [i for i, item in enumerate(packed.items) if "parser.rs:12" in str(item)]
    assert user_positions and tool_positions
    assert max(user_positions + tool_positions) >= min(user_positions)


def test_pack_marks_needs_compact_instead_of_dropping(monkeypatch):
    monkeypatch.setattr("app.context.windows.compact_threshold_tokens", lambda *args, **kwargs: 100)
    items = []
    for i in range(30):
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": ("history " + str(i) + " ") * 80}]})
    items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "latest"}]} )
    turn = Turn(model="tiny-local", want_stream=True, client_api="responses", items=items, current_user="latest", catalog=build_catalog([], items))
    packed = pack_turn_items(turn)
    assert packed.dropped_items == 0
    assert packed.needs_compact is True
    assert "latest" in str(packed.items)
    assert "history 0" in str(packed.items)
    assert any(LOSS_PREFIX in notice for notice in packed.notices)


def test_pack_keeps_encrypted_reasoning():
    items = [
        {"type": "reasoning", "id": "rs_old", "encrypted_content": "enc-old", "summary": []},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
        {"type": "function_call", "name": "js", "call_id": "c1", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "tab locked"},
    ]
    turn = Turn(
        model="gpt-5.5",
        want_stream=True,
        client_api="responses",
        items=items,
        current_user="continue",
        catalog=build_catalog([], items),
    )
    packed = pack_turn_items(turn)
    reasoning = [item for item in packed.items if item.get("type") == "reasoning"]
    assert reasoning
    assert reasoning[0]["encrypted_content"] == "enc-old"


def test_pack_does_not_clip_long_tool_output():
    huge = "parser line " * 8000
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "show parser"}]},
        {"type": "function_call", "name": "shell_command", "call_id": "c1", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": huge},
    ]
    turn = Turn(
        model="gpt-5.5",
        want_stream=True,
        client_api="responses",
        items=items,
        current_user="show parser",
        catalog=build_catalog([], items),
    )
    packed = pack_turn_items(turn)
    blob = str(packed.items)
    assert huge in blob
    assert "_local_clip_notice" not in blob
    assert packed.clipped_outputs == 0
