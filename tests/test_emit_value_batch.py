from __future__ import annotations

from app.bridge.parse import parse_emit_value
from app.bridge.prompt import build_instructions
from app.bridge.schema import build_emit_value_schema
from app.config import settings
from app.ingest.catalog import build_catalog
from app.protocol.models import Turn
from app.protocol.sse import parse_sse_lines
from app.upstream.execute import collect_from_events, merge_reasoning_item, merge_wrappers


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


def test_emit_value_schema_keeps_native_parameter_fields():
    catalog = build_catalog(
        [
            {
                "type": "function",
                "name": "js",
                "namespace": "mcp__cua_repl",
                "description": "run js in computer use",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
                "examples": [{"code": "await browser.user.openTabs()"}],
            }
        ],
        [],
        query="chrome",
    )
    schema = build_emit_value_schema(settings, catalog)
    items = schema["parameters"]["properties"]["tool_calls"]["items"]
    variants = items.get("oneOf") or [items]
    js = next(item for item in variants if item.get("properties", {}).get("name", {}).get("enum") == ["js"])
    args = js["properties"]["arguments"]
    assert args.get("type") == "object"
    assert "code" in args.get("properties", {})
    assert args.get("required") == ["code"]
    assert js["description"] == "run js in computer use"
    assert js["examples"][0]["code"] == "await browser.user.openTabs()"


def test_parse_accepts_native_object_arguments():
    catalog = build_catalog(
        [
            {
                "type": "function",
                "name": "js",
                "namespace": "mcp__cua_repl",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            }
        ],
        [],
        query="chrome",
    )
    wrapper = parse_emit_value(
        '{"mode":"tool_call","answer":"","tool_calls":[{"type":"function","name":"js","namespace":"mcp__cua_repl","arguments":{"code":"1+1"}}]}',
        catalog,
    )
    assert wrapper.tool_calls
    assert "1+1" in wrapper.tool_calls[0].arguments


def test_instructions_keep_full_persona():
    catalog = _catalog()
    developer = "人设段落。" * 2500
    system = "系统规则。" * 1500
    instructions = "客户端说明。" * 1500
    turn = Turn(
        model="gpt-5.5",
        want_stream=True,
        client_api="responses",
        items=[],
        current_user="hi",
        catalog=catalog,
        developer_text=developer,
        system_text=system,
        instructions=instructions,
    )
    blob = build_instructions(turn)
    assert developer in blob
    assert system in blob
    assert instructions in blob
    assert len(developer) > 8000
    assert len(system) > 4000
    assert len(instructions) > 4000


def test_collect_from_events_keeps_encrypted_reasoning():
    catalog = _catalog()
    turn = Turn(model="gpt-5.5", want_stream=True, client_api="responses", items=[], current_user="hi", catalog=catalog)
    sse = """
event: response.output_item.added
data: {"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_mem","encrypted_content":"enc-a","summary":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_mem","encrypted_content":"enc-b","summary":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","item":{"type":"function_call","name":"emit_value","call_id":"c1"}}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","call_id":"c1","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"done\\",\\"tool_calls\\":[]}"}

""".lstrip()
    result = collect_from_events(parse_sse_lines(sse.splitlines()), turn)
    assert result.reasoning_items
    assert result.reasoning_items[0]["id"] == "rs_mem"
    assert result.reasoning_items[0]["encrypted_content"] == "enc-b"


def test_collect_from_events_keeps_reasoning_summary():
    catalog = _catalog()
    turn = Turn(model="gpt-5.5", want_stream=True, client_api="responses", items=[], current_user="hi", catalog=catalog)
    sse = """
event: response.output_item.added
data: {"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_sum","summary":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_sum","summary_index":0,"delta":"先看"}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","item_id":"rs_sum","summary_index":0,"text":"先看标签"}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_sum","encrypted_content":"enc"}}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","call_id":"c1","arguments":"{\\"mode\\":\\"answer\\",\\"answer\\":\\"ok\\",\\"tool_calls\\":[]}"}

""".lstrip()
    result = collect_from_events(parse_sse_lines(sse.splitlines()), turn)
    assert result.reasoning_items[0]["summary"][0]["text"] == "先看标签"
    assert result.reasoning_items[0]["encrypted_content"] == "enc"
    items = []
    merge_reasoning_item(items, {"type": "reasoning", "id": "rs_mem", "encrypted_content": "enc-a"})
    merge_reasoning_item(items, {"type": "reasoning", "id": "rs_mem", "encrypted_content": "enc-b"})
    assert len(items) == 1
    assert items[0]["encrypted_content"] == "enc-b"
