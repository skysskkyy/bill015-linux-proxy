from __future__ import annotations

from app.bridge.exec_source import is_code_mode_exec, normalize_exec_source, unwrap_exec_source
from app.bridge.parse import parse_emit_value
from app.ingest.catalog import build_catalog
from app.protocol.models import TurnResult
from app.upstream.execute import finalize_visible_answer, merge_wrappers


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


def test_status_note_rides_with_tools():
    catalog = build_catalog([{"type": "function", "name": "shell_command"}, {"type": "tool_search", "name": "tool_search"}], [])
    wrapper = parse_emit_value(
        '{"mode":"tool_call","answer":"\u6211\u5148\u6253\u5f00 Chrome","tool_calls":[{"type":"function","name":"shell_command","arguments":"{\\"command\\":\\"pwd\\"}"}]}',
        catalog,
    )
    assert wrapper.mode == "tool_call"
    answer, commentary, tools, _ = merge_wrappers([wrapper])
    assert answer == ""
    assert "Chrome" in commentary
    assert tools
    lone = parse_emit_value(
        '{"mode":"answer","answer":"\u6211\u5148\u68c0\u67e5\u9879\u76ee\u7ed3\u6784\u3002","tool_calls":[]}',
        catalog,
    )
    assert lone.mode == "answer"
    assert lone.tool_calls == []
    silent = parse_emit_value(
        '{"mode":"tool_call","tool_calls":[{"type":"function","name":"shell_command","arguments":"{\\"command\\":\\"pwd\\"}"}]}',
        catalog,
    )
    assert silent.mode == "tool_call"
    assert silent.answer == ""
    assert silent.tool_calls


def test_finalize_promotes_commentary_when_answer_empty():
    result = TurnResult(local_request_id="x", answer="", commentary="我先搜索再整理。")
    finalize_visible_answer(result, did_local_work=True)
    assert result.answer == "我先搜索再整理。"
    empty = TurnResult(local_request_id="y", answer="", commentary="")
    finalize_visible_answer(empty, did_local_work=True)
    assert "without a user-facing answer" in empty.answer
    kept = TurnResult(local_request_id="z", answer="葛饰区推荐三家店。", commentary="searching")
    finalize_visible_answer(kept, did_local_work=True)
    assert kept.answer == "葛饰区推荐三家店。"


def test_answer_with_tools_keeps_tools():
    catalog = build_catalog([{"type": "function", "name": "shell_command"}], [])
    wrapper = parse_emit_value(
        '{"mode":"answer","answer":"checking","tool_calls":[{"type":"function","name":"shell_command","arguments":"{\\"command\\":\\"pwd\\"}"}]}',
        catalog,
    )
    assert wrapper.mode == "tool_call"
    assert wrapper.tool_calls
