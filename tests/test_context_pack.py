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


def test_pack_inserts_loss_notice_when_dropping_history(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "context_window_tokens", 16000)
    monkeypatch.setattr(settings, "compact_target_percent", 50)
    monkeypatch.setattr(settings, "max_output_tokens", 8000)
    items = []
    for i in range(30):
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": ("history " + str(i) + " ") * 80}]})
    items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "latest"}]} )
    turn = Turn(model="gpt-5.5", want_stream=True, client_api="responses", items=items, current_user="latest", catalog=build_catalog([], items))
    packed = pack_turn_items(turn)
    assert packed.dropped_items > 0
    assert any(LOSS_PREFIX in str(item) for item in packed.items)


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
