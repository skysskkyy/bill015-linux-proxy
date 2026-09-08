from __future__ import annotations

from app.bridge.exec_source import is_code_mode_exec, normalize_exec_source, unwrap_exec_source
from app.bridge.parse import parse_emit_value
from app.bridge.progress import looks_like_progress, should_continue_progress
from app.ingest.catalog import build_catalog
from app.protocol.models import TurnResult
from app.upstream.execute import merge_wrappers


def test_unwrap_json_input_and_fences():
    assert unwrap_exec_source('{"input":"const x = 1"}') == "const x = 1"
    nested = '{"input":"{\\"code\\":\\"1+1\\"}"}'
    assert unwrap_exec_source(nested) == "1+1"
    fenced = "```javascript\nconst y = 2\n```"
    assert unwrap_exec_source(fenced) == "const y = 2"


def test_normalize_unwraps_json_and_returns_last_expression():
    src = normalize_exec_source(
        '{"input":"const results = await Promise.allSettled([tools.exec_command({cmd:\\"ls\\"})]); results.map((r,i)=>({i,status:r.status}));"}'
    )
    assert "/*__bill015_exec*/" in src
    assert "return results.map" in src
    assert "text(v)" in src
    assert '{"input"' not in src.split("await", 1)[-1]


def test_normalize_makes_top_level_return_legal():
    src = normalize_exec_source("return results.map((r) => r);")
    assert src.startswith("/*__bill015_exec*/") or "/*__bill015_exec*/" in src
    assert "return results.map" in src
    assert "await (async () =>" in src
    again = normalize_exec_source(src)
    assert again.count("/*__bill015_exec*/") == 1


def test_code_mode_exec_identity():
    assert is_code_mode_exec("exec", None)
    assert is_code_mode_exec("exec", "functions")
    assert not is_code_mode_exec("exec", "mcp__python")
    assert not is_code_mode_exec("apply_patch", None)


def test_parse_exec_json_wrapper_becomes_raw_js():
    catalog = build_catalog([{"type": "custom", "name": "exec", "format": {"type": "grammar"}}], [])
    wrapper = parse_emit_value(
        '{"mode":"tool_call","answer":"","tool_calls":[{"type":"custom","name":"exec","arguments":"{\\"input\\":\\"const results = 1; results\\"}"}]}',
        catalog,
    )
    assert wrapper.tool_calls
    call = wrapper.tool_calls[0]
    assert call.call_type == "custom"
    assert call.input.startswith("/*__bill015_exec*/") or "/*__bill015_exec*/" in call.input
    assert "return results" in call.input
    assert call.input.strip()[0] != "{"


def test_progress_note_is_not_final_answer():
    catalog = build_catalog([{"type": "function", "name": "shell_command"}, {"type": "tool_search", "name": "tool_search"}], [])
    text = (
        "\u6211\u5148\u5feb\u901f\u68c0\u67e5\u9879\u76ee\u7ed3\u6784\u3001"
        "\u8bf4\u660e\u6587\u4ef6\u548c\u5f53\u524d\u72b6\u6001\uff0c"
        "\u518d\u7ed9\u4f60\u4e00\u4e2a\u7b80\u660e\u6982\u89c8\u3002"
    )
    assert looks_like_progress(text)
    assert looks_like_progress("I'll inspect the project files, then give you a short overview.")
    wrapper = parse_emit_value(
        '{"mode":"answer","answer":"' + text + '","tool_calls":[]}',
        catalog,
    )
    assert wrapper.mode == "progress"
    answer, commentary, tools, _ = merge_wrappers([wrapper])
    assert answer == ""
    assert text in commentary
    assert tools == []
    result = TurnResult(local_request_id="x", commentary=commentary, answer=answer, tool_calls=tools)
    assert should_continue_progress(result)
    real = parse_emit_value(
        '{"mode":"answer","answer":"\u9879\u76ee\u7528\u9014\u662f\u8ba1\u7b97\u822a\u9053\u3002","tool_calls":[]}',
        catalog,
    )
    assert real.mode == "answer"


def test_answer_with_tools_keeps_tools():
    catalog = build_catalog([{"type": "function", "name": "shell_command"}], [])
    wrapper = parse_emit_value(
        '{"mode":"answer","answer":"checking","tool_calls":[{"type":"function","name":"shell_command","arguments":"{\\"command\\":\\"pwd\\"}"}]}',
        catalog,
    )
    assert wrapper.mode == "tool_call"
    assert wrapper.tool_calls
