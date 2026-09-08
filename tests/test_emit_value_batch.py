from __future__ import annotations

from app.bridge.parse import parse_emit_value
from app.ingest.catalog import build_catalog
from app.protocol.models import Turn
from app.protocol.sse import parse_sse_lines
from app.upstream.execute import collect_from_events, merge_wrappers


def _catalog():
    return build_catalog(
        [
            {"type": "function", "name": "shell_command"},
            {"type": "custom", "name": "apply_patch"},
            {"type": "tool_search", "name": "tool_search"},
        ],
        [],
        query="edit files",
    )


def test_parse_multiple_tool_calls_in_one_wrapper():
    catalog = _catalog()
    raw = (
        '{"mode":"tool_call","answer":"","tool_calls":['
        '{"type":"function","name":"shell_command","arguments":"{\\"command\\":\\"ls\\"}"},'
        '{"type":"custom","name":"apply_patch","input":"*** Begin Patch\\n*** End Patch","arguments":"{}"}'
        "]}"
    )
    wrapper = parse_emit_value(raw, catalog)
    assert wrapper.mode == "tool_call"
    assert [c.call_type for c in wrapper.tool_calls] == ["function", "custom"]
    assert wrapper.tool_calls[1].input.startswith("*** Begin Patch")


def test_merge_multiple_wrappers_unions_tools():
    catalog = _catalog()
    first = parse_emit_value(
        '{"mode":"tool_call","answer":"checking","tool_calls":[{"type":"function","name":"shell_command","arguments":"{\\"command\\":\\"pwd\\"}"}]}',
        catalog,
    )
    second = parse_emit_value('{"mode":"tool_call","answer":"","tool_calls":[{"type":"custom","name":"apply_patch","input":"patch"}]}', catalog)
    answer, commentary, tools, _ = merge_wrappers([first, second])
    assert "checking" in commentary
    assert [c.name for c in tools] == ["shell_command", "apply_patch"]
    assert answer == ""


def test_collect_from_events_batches_two_emit_values():
    catalog = _catalog()
    turn = Turn(model="gpt-5.5", want_stream=True, client_api="responses", items=[], current_user="hi", catalog=catalog)
    sse = """
event: response.created
data: {"type":"response.created","response":{"id":"resp_up"}}

event: response.output_item.added
data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value","call_id":"c1"}}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","call_id":"c1","arguments":"{\\"mode\\":\\"tool_call\\",\\"answer\\":\\"\\",\\"tool_calls\\":[{\\"type\\":\\"function\\",\\"name\\":\\"shell_command\\",\\"arguments\\":\\"{\\\\\\"command\\\\\\":\\\\\\"ls\\\\\\"}\\"}]}"}

event: response.output_item.added
data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value","call_id":"c2"}}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","call_id":"c2","arguments":"{\\"mode\\":\\"tool_call\\",\\"answer\\":\\"\\",\\"tool_calls\\":[{\\"type\\":\\"custom\\",\\"name\\":\\"apply_patch\\",\\"input\\":\\"*** Begin Patch\\"}]}"}

""".lstrip()
    result = collect_from_events(parse_sse_lines(sse.splitlines()), turn)
    assert result.wrapper_count == 2
    assert result.aborted is True
    assert {c.name for c in result.tool_calls} == {"shell_command", "apply_patch"}
