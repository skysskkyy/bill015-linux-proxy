from __future__ import annotations

import json
from typing import Any

from ..config import Settings, settings
from ..protocol.models import TOOL_OUTPUT_TYPES, Catalog, ToolSpec

CORE_NAMES = {
    "exec",
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
}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _tool_name(tool: dict[str, Any], idx: int) -> str:
    for key in ("name", "tool_name", "id"):
        raw = tool.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    namespace = str(tool.get("namespace") or "").strip()
    name = str(tool.get("name") or "").strip()
    if namespace and name:
        return f"{namespace}__{name}"
    typ = str(tool.get("type") or "tool")
    return f"{typ}_{idx}"


def _call_type(tool: dict[str, Any], name: str) -> tuple[str, str]:
    raw = str(tool.get("type") or tool.get("raw_type") or "function").strip().lower()
    if raw in {"custom", "freeform"} or name in {"apply_patch"} or tool.get("format"):
        return "custom", raw or "custom"
    if raw in {"tool_search", "web_search"} or name == "tool_search":
        return "tool_search", raw or "tool_search"
    return "function", raw or "function"


def _is_core(name: str, namespace: str | None, description: str) -> bool:
    blob = f"{name} {namespace or ''} {description}".lower()
    return any(token in blob for token in CORE_NAMES) or name.lower() in CORE_NAMES


def spec_from_tool(tool: dict[str, Any], idx: int) -> ToolSpec | None:
    if not isinstance(tool, dict):
        return None
    name = _tool_name(tool, idx)
    namespace = tool.get("namespace")
    namespace_s = str(namespace).strip() if isinstance(namespace, str) and namespace.strip() else None
    call_type, raw_type = _call_type(tool, name)
    description = str(tool.get("description") or "")
    parameters = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    alias = name if not namespace_s else f"{namespace_s}__{name}"
    return ToolSpec(
        alias=alias,
        name=name,
        namespace=namespace_s,
        call_type=call_type,  # type: ignore[arg-type]
        raw_type=raw_type,
        description=description,
        parameters=parameters,
        core=_is_core(name, namespace_s, description),
    )


def _collect_from_tools_array(tools: Any, specs: dict[str, ToolSpec]) -> None:
    for idx, tool in enumerate(_as_list(tools)):
        spec = spec_from_tool(tool, idx)
        if spec:
            specs.setdefault(spec.alias, spec)
            specs.setdefault(spec.name, spec)


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
    return Catalog(specs=unique, selected=selected, deferred=deferred, dropped=dropped)


def catalog_json(catalog: Catalog, cfg: Settings = settings) -> str:
    rows = []
    for alias in catalog.selected:
        spec = catalog.specs.get(alias)
        if not spec:
            continue
        rows.append(
            {
                "name": spec.name,
                "namespace": spec.namespace,
                "call_type": spec.call_type,
                "description": spec.description[:400],
            }
        )
    blob = json.dumps(rows, ensure_ascii=False)
    if len(blob) > cfg.tool_bridge_catalog_max_chars:
        blob = blob[: cfg.tool_bridge_catalog_max_chars] + "…"
    return blob
