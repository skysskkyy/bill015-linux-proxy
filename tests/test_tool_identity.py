from __future__ import annotations

import asyncio

from app.bridge.parse import parse_emit_value
from app.config import Settings
from app.ingest.catalog import build_catalog, iter_callable_tools
from app.protocol.models import BridgeToolCall, Turn, TurnResult
from app.protocol.names import (
    catalog_aliases,
    is_namespace_only,
    resolve_catalog_tool,
    split_tool_identity,
)
from app.protocol.sse import parse_sse_lines
from app.replay.events import responses_sse_generator


def _emit(name: str, *, namespace: str | None = None, arguments: str = "{}") -> str:
    ns = f',"namespace":"{namespace}"' if namespace else ""
    return '{"mode":"tool_call","answer":"","tool_calls":[{"type":"function","name":"' + name + '"' + ns + ',"arguments":"' + arguments.replace('"', '\\"') + '"}]}'


def _parse(catalog, raw: str, *, allow_unknown: bool = False):
    cfg = Settings(tool_bridge_allow_unknown_tools=allow_unknown)
    return parse_emit_value(raw, catalog, cfg)


def test_split_does_not_carve_mcp_prefix_off_two_segment_names():
    assert split_tool_identity("mcp__cua_repl") == ("mcp__cua_repl", None)
    assert split_tool_identity("mcp__python") == ("mcp__python", None)
    assert is_namespace_only("mcp__cua_repl")
    assert is_namespace_only("functions")
    assert is_namespace_only("mcp")
    assert not is_namespace_only("exec")
    assert split_tool_identity("functions.exec") == ("exec", "functions")
    assert split_tool_identity("mcp__python.exec") == ("exec", "mcp__python")
    assert split_tool_identity("mcp__codex_apps__calendar.create_event") == (
        "create_event",
        "mcp__codex_apps__calendar",
    )
    assert split_tool_identity("mcp__python__exec") == ("exec", "mcp__python")
    assert split_tool_identity("web.run") == ("run", "web")
    assert split_tool_identity("image_gen.imagegen") == ("imagegen", "image_gen")
    assert split_tool_identity("skills.list") == ("list", "skills")
    assert split_tool_identity("memories.search") == ("search", "memories")
    assert split_tool_identity("clock.curr_time") == ("curr_time", "clock")
    assert split_tool_identity("collaboration.spawn_agent") == ("spawn_agent", "collaboration")
    assert split_tool_identity("extension/echo") == ("echo", "extension")


def test_namespace_objects_unpack_and_keep_full_mcp_namespace():
    tools = [
        {"type": "function", "name": "exec", "description": "code-mode exec"},
        {"type": "custom", "name": "apply_patch"},
        {"type": "tool_search", "name": "tool_search"},
        {
            "type": "namespace",
            "name": "mcp__cua_repl",
            "tools": [{"type": "function", "name": "cua_repl", "description": "computer use repl"}],
        },
        {
            "type": "namespace",
            "name": "mcp__python",
            "tools": [{"type": "function", "name": "exec", "description": "python mcp exec"}],
        },
        {
            "type": "namespace",
            "name": "mcp__codex_apps__calendar",
            "tools": [{"type": "function", "name": "create_event"}],
        },
        {
            "type": "namespace",
            "name": "web",
            "tools": [{"type": "function", "name": "run"}],
        },
        {
            "type": "namespace",
            "name": "image_gen",
            "tools": [{"type": "function", "name": "imagegen"}],
        },
        {
            "type": "namespace",
            "name": "skills",
            "tools": [{"type": "function", "name": "list"}, {"type": "function", "name": "read"}],
        },
        {
            "type": "namespace",
            "name": "memories",
            "tools": [{"type": "function", "name": "search"}],
        },
        {
            "type": "namespace",
            "name": "clock",
            "tools": [{"type": "function", "name": "curr_time"}, {"type": "function", "name": "sleep"}],
        },
        {
            "type": "namespace",
            "name": "collaboration",
            "tools": [{"type": "function", "name": "spawn_agent"}],
        },
        {"type": "web_search"},
    ]
    flat = iter_callable_tools(tools)
    assert all(item.get("type") != "namespace" for item in flat)
    assert all(item.get("type") != "web_search" for item in flat)
    catalog = build_catalog(tools, [], query="run")
    cua = resolve_catalog_tool(catalog, "cua_repl")
    assert cua is not None
    assert cua.name == "cua_repl"
    assert cua.namespace == "mcp__cua_repl"
    python_exec = resolve_catalog_tool(catalog, "mcp__python.exec")
    assert python_exec is not None
    assert python_exec.name == "exec"
    assert python_exec.namespace == "mcp__python"
    core_exec = resolve_catalog_tool(catalog, "exec")
    assert core_exec is not None
    assert core_exec.namespace is None
    calendar = resolve_catalog_tool(catalog, "mcp__codex_apps__calendar.create_event")
    assert calendar is not None
    assert calendar.name == "create_event"
    assert calendar.namespace == "mcp__codex_apps__calendar"


def test_emit_replays_catalog_identity_not_display_concat():
    catalog = build_catalog(
        [
            {"type": "function", "name": "exec"},
            {
                "type": "namespace",
                "name": "mcp__cua_repl",
                "tools": [{"type": "function", "name": "cua_repl"}],
            },
            {
                "type": "namespace",
                "name": "mcp__python",
                "tools": [{"type": "function", "name": "exec"}],
            },
            {"type": "function", "name": "exec", "namespace": "functions"},
        ],
        [],
    )
    dropped = _parse(catalog, _emit("functions"))
    assert dropped.tool_calls == []
    dropped_mcp = _parse(catalog, _emit("mcp"))
    assert dropped_mcp.tool_calls == []
    cua = _parse(catalog, _emit("mcp__cua_repl"))
    assert cua.tool_calls
    assert cua.tool_calls[0].name == "cua_repl"
    assert cua.tool_calls[0].namespace == "mcp__cua_repl"
    wrong_split = _parse(catalog, _emit("cua_repl", namespace="mcp"))
    assert wrong_split.tool_calls
    assert wrong_split.tool_calls[0].name == "cua_repl"
    assert wrong_split.tool_calls[0].namespace == "mcp__cua_repl"
    joined = _parse(catalog, _emit("mcp__python__exec"))
    assert joined.tool_calls[0].name == "exec"
    assert joined.tool_calls[0].namespace == "mcp__python"
    functions_dot = _parse(catalog, _emit("functions.exec"))
    assert functions_dot.tool_calls[0].name == "exec"
    assert functions_dot.tool_calls[0].namespace in {None, "functions"}


def test_unknown_mcp_flat_name_is_not_split_to_mcp_plus_tool():
    catalog = build_catalog([{"type": "function", "name": "exec"}], [])
    dropped = _parse(catalog, _emit("mcp__cua_repl"))
    assert dropped.tool_calls == []
    unknown = _parse(catalog, _emit("mcp__cua_repl"), allow_unknown=True)
    assert unknown.tool_calls
    assert unknown.tool_calls[0].name == "mcp__cua_repl"
    assert unknown.tool_calls[0].namespace is None


def test_plain_exec_wins_functions_dot_when_catalog_has_no_functions_namespace():
    catalog = build_catalog([{"type": "function", "name": "exec", "description": "run js"}], [])
    wrapper = _parse(catalog, _emit("functions.exec"))
    assert wrapper.tool_calls
    assert wrapper.tool_calls[0].name == "exec"
    assert wrapper.tool_calls[0].namespace is None


def test_additional_tools_namespace_objects_are_cataloged():
    items = [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {"type": "function", "name": "shell_command"},
                {
                    "type": "namespace",
                    "name": "mcp__codex_apps__calendar",
                    "tools": [{"type": "function", "name": "create_event"}],
                },
            ],
        }
    ]
    catalog = build_catalog([], items)
    spec = resolve_catalog_tool(catalog, "create_event", "mcp__codex_apps__calendar")
    assert spec is not None
    assert spec.name == "create_event"
    assert spec.namespace == "mcp__codex_apps__calendar"
    assert "mcp__codex_apps__calendar.create_event" in catalog_aliases(spec.name, spec.namespace)


def test_replay_sse_keeps_full_mcp_namespace():
    turn = Turn(
        model="gpt-5.5",
        want_stream=True,
        client_api="responses",
        items=[],
        current_user="run",
        catalog=build_catalog([], []),
    )
    result = TurnResult(
        local_request_id="resp_local_ns",
        commentary="",
        tool_calls=[
            BridgeToolCall(
                id="call_cua",
                name="cua_repl",
                arguments="{}",
                call_type="function",
                namespace="mcp__cua_repl",
            )
        ],
    )

    async def _ready(value: TurnResult) -> TurnResult:
        return value

    async def _collect() -> list[str]:
        chunks: list[str] = []
        async for raw in responses_sse_generator(_ready(result), turn, result.local_request_id):
            chunks.append(raw.decode("utf-8"))
        return chunks

    events = list(parse_sse_lines("".join(asyncio.run(_collect())).splitlines()))
    done = next(
        ev.json
        for ev in events
        if ev.json
        and ev.json.get("type") == "response.output_item.done"
        and (ev.json.get("item") or {}).get("type") == "function_call"
    )
    item = done["item"]
    assert item["name"] == "cua_repl"
    assert item["namespace"] == "mcp__cua_repl"
    assert item["namespace"] + item["name"] != "mcpcua_repl"
