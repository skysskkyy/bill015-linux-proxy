from __future__ import annotations

import json
from typing import Any

from ..config import Settings, settings
from ..protocol.models import TOOL_OUTPUT_TYPES, Catalog, ToolSpec
from ..protocol.names import catalog_aliases, is_namespace_only, mcp_join, split_tool_identity

CORE_NAMES = {
    "exec",
    "wait",
    "exec_command",
    "write_stdin",
    "shell",
    "shell_command",
    "apply_patch",
    "tool_search",
    "browser",
    "chrome",
    "playwright",
    "node_repl",
    "jshook",
    "update_plan",
    "grep_files",
    "view_image",
    "web_search",
    "web_extract",
}

HOSTED_TOOL_TYPES = {"web_search", "computer", "computer_use"}
NAME_KEYS = ("name", "tool_name", "callable_name", "id")
NAMESPACE_KEYS = ("namespace", "tool_namespace", "callable_namespace")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _first_str(tool: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        raw = tool.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _tool_name(tool: dict[str, Any], idx: int) -> str:
    name = _first_str(tool, NAME_KEYS)
    if name:
        return name
    nested = tool.get("function")
    if isinstance(nested, dict):
        name = _first_str(nested, NAME_KEYS)
        if name:
            return name
    typ = str(tool.get("type") or "tool")
    return f"{typ}_{idx}"


def _tool_namespace(tool: dict[str, Any]) -> str | None:
    return _first_str(tool, NAMESPACE_KEYS)


def iter_callable_tools(tools: Any) -> list[dict[str, Any]]:
    """Unpack `type=namespace` containers; skip hosted tools and namespace-only rows."""
    found: list[dict[str, Any]] = []
    for tool in _as_list(tools):
        if not isinstance(tool, dict):
            continue
        typ = str(tool.get("type") or tool.get("raw_type") or "").strip().lower()
        nested = tool.get("tools")
        if typ == "namespace" or (typ == "" and isinstance(nested, list) and _first_str(tool, NAME_KEYS)):
            ns = _tool_namespace(tool) or _first_str(tool, NAME_KEYS)
            for child in _as_list(nested):
                if not isinstance(child, dict):
                    continue
                cloned = dict(child)
                if ns and not _tool_namespace(cloned):
                    cloned["namespace"] = ns
                found.extend(iter_callable_tools([cloned]))
            continue
        if typ in HOSTED_TOOL_TYPES:
            continue
        found.append(tool)
    return found


def _call_type(tool: dict[str, Any], name: str, namespace: str | None = None) -> tuple[str, str]:
    raw = str(tool.get("type") or tool.get("raw_type") or "function").strip().lower()
    ns = namespace or _tool_namespace(tool) or ""
    if name == "exec" and not str(ns).startswith("mcp__"):
        return "custom", raw or "custom"
    if raw in {"custom", "freeform"} or name in {"apply_patch"} or tool.get("format"):
        return "custom", raw or "custom"
    if raw == "tool_search" or name == "tool_search":
        return "tool_search", raw or "tool_search"
    return "function", raw or "function"


def _is_core(name: str, namespace: str | None, description: str) -> bool:
    blob = f"{name} {namespace or ''} {description}".lower()
    return any(token in blob for token in CORE_NAMES) or name.lower() in CORE_NAMES


def _nested_function(tool: dict[str, Any]) -> dict[str, Any]:
    nested = tool.get("function")
    return nested if isinstance(nested, dict) else {}


def _tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    nested = _nested_function(tool)
    for source in (tool, nested):
        for key in ("parameters", "input_schema", "json_schema"):
            value = source.get(key)
            if isinstance(value, dict) and value:
                return value
    return {}


def _tool_examples(tool: dict[str, Any]) -> Any:
    nested = _nested_function(tool)
    for source in (tool, nested):
        for key in ("examples", "example"):
            value = source.get(key)
            if value not in (None, "", []):
                return value
    return None


def _tool_format(tool: dict[str, Any]) -> dict[str, Any] | None:
    value = tool.get("format")
    return value if isinstance(value, dict) else None


def spec_from_tool(tool: dict[str, Any], idx: int) -> ToolSpec | None:
    if not isinstance(tool, dict):
        return None
    raw_name = _tool_name(tool, idx)
    namespace_s = _tool_namespace(tool)
    name, namespace_s = split_tool_identity(raw_name, namespace_s)
    if is_namespace_only(name, namespace_s) or not name:
        return None
    call_type, raw_type = _call_type(tool, name, namespace_s)
    nested = _nested_function(tool)
    description = str(tool.get("description") or nested.get("description") or "")
    parameters = _tool_parameters(tool)
    alias = name if not namespace_s else mcp_join(name, namespace_s)
    return ToolSpec(
        alias=alias,
        name=name,
        namespace=namespace_s,
        call_type=call_type,  # type: ignore[arg-type]
        raw_type=raw_type,
        description=description,
        parameters=parameters,
        examples=_tool_examples(tool),
        format=_tool_format(tool),
        core=_is_core(name, namespace_s, description),
    )


def _collect_from_tools_array(tools: Any, specs: dict[str, ToolSpec]) -> None:
    for idx, tool in enumerate(iter_callable_tools(tools)):
        spec = spec_from_tool(tool, idx)
        if not spec:
            continue
        for alias in catalog_aliases(spec.name, spec.namespace):
            specs.setdefault(alias, spec)


def collect_additional_tools(items: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for item in _as_list(items):
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") == "additional_tools":
            found.extend(tool for tool in _as_list(item.get("tools")) if isinstance(tool, dict))
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and str(part.get("type") or "") == "additional_tools":
                    found.extend(tool for tool in _as_list(part.get("tools")) if isinstance(tool, dict))
    return found


def collect_search_output_tools(items: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for item in _as_list(items):
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") in TOOL_OUTPUT_TYPES | {"tool_search_output"}:
            tools = item.get("tools")
            found.extend(tool for tool in _as_list(tools) if isinstance(tool, dict))
    return found


def _score(spec: ToolSpec, query: str) -> int:
    if spec.core:
        return 1000
    blob = f"{spec.alias} {spec.name} {spec.description}".lower()
    score = 0
    for token in query.lower().split():
        if len(token) < 3:
            continue
        if token in blob:
            score += 8
    return score


def build_catalog(
    tools: Any,
    items: Any,
    *,
    query: str = "",
    cfg: Settings = settings,
) -> Catalog:
    specs: dict[str, ToolSpec] = {}
    _collect_from_tools_array(tools, specs)
    _collect_from_tools_array(collect_additional_tools(items), specs)
    _collect_from_tools_array(collect_search_output_tools(items), specs)

    unique: dict[str, ToolSpec] = {}
    for spec in specs.values():
        unique.setdefault(spec.alias, spec)
    ranked = sorted(unique.values(), key=lambda spec: (-_score(spec, query), spec.alias))
    selected: list[str] = []
    deferred: list[str] = []
    limit = max(8, cfg.tool_bridge_selection_max_tools)
    for spec in ranked:
        if len(selected) < limit:
            selected.append(spec.alias)
        else:
            deferred.append(spec.alias)
    dropped = 0
    if len(unique) > cfg.tool_bridge_schema_max_tools:
        overflow = ranked[cfg.tool_bridge_schema_max_tools :]
        dropped = len(overflow)
        for spec in overflow:
            unique.pop(spec.alias, None)
            if spec.alias in selected:
                selected.remove(spec.alias)
            if spec.alias in deferred:
                deferred.remove(spec.alias)
    catalog = Catalog(specs=unique, selected=selected, deferred=deferred, dropped=dropped)
    if cfg.web_enabled:
        from ..tools.web import inject_proxy_web_tools

        catalog = inject_proxy_web_tools(catalog, cfg)
    return catalog


def catalog_json(catalog: Catalog, cfg: Settings = settings) -> str:
    _ = cfg
    rows = []
    for alias in catalog.selected:
        spec = catalog.specs.get(alias)
        if not spec:
            continue
        row: dict[str, Any] = {
            "name": spec.name,
            "namespace": spec.namespace,
            "call_type": spec.call_type,
            "description": spec.description or "",
        }
        if spec.parameters:
            row["parameters"] = spec.parameters
            required = spec.parameters.get("required")
            if required:
                row["required"] = required
        if spec.examples not in (None, "", []):
            row["examples"] = spec.examples
        if spec.format:
            row["format"] = spec.format
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False)
