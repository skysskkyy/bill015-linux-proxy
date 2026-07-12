from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from typing import Any

from .config import Settings, settings
from .models import BridgeToolCall


def _compact_tool_doc(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _compact_tool_doc(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_compact_tool_doc(v) for v in value]
    return value


def _register_tool_alias(registry: dict[str, dict[str, Any]], alias: str, spec: dict[str, Any], *, replace: bool = False) -> None:
    alias = str(alias or "").strip()
    if not alias:
        return
    key = alias.lower()
    if replace or key not in registry:
        registry[key] = spec
        return
    current = registry.get(key) or {}
    if (
        current.get("namespace") != spec.get("namespace")
        or current.get("output_name") != spec.get("output_name")
    ):
        # Bare subtool names can collide across multiple MCP namespaces. Keep
        # the first mapping for backward compatibility, but mark it so the
        # system prompt and audit trail can diagnose ambiguous unqualified
        # calls. Qualified names (namespace.tool / namespace__tool / explicit
        # namespace field) always resolve losslessly.
        current["ambiguous_alias"] = True
        current.setdefault("ambiguous_with", [])
        current["ambiguous_with"].append({"namespace": spec.get("namespace"), "name": spec.get("output_name")})


def build_client_tool_catalog(tools: Any, max_chars: int = 120_000) -> tuple[str, dict[str, dict[str, Any]]]:
    registry: dict[str, dict[str, Any]] = {}
    if not isinstance(tools, list) or not tools:
        return "", registry

    catalog: list[dict[str, Any]] = []
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        typ = str(tool.get("type") or "function")
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        desc = tool.get("description") or (tool.get("function") or {}).get("description") or ""

        if typ == "namespace":
            catalog.append(_namespace_tool_doc(idx, name, desc, tool, registry))
            continue

        if not name:
            name = _implicit_tool_name(idx, typ)

        call_type = "custom" if typ == "custom" else "function"
        if typ in {"tool_search", "web_search"}:
            # These are native Responses item types, not ordinary function_call
            # items. Codex Desktop records them as tool_search_call and
            # web_search_call in subsequent input history.
            call_type = typ

        doc = _single_tool_doc(str(name), typ, desc, tool)
        catalog.append(_compact_tool_doc(doc))
        _register_tool_alias(
            registry,
            str(name),
            {"call_type": call_type, "output_name": str(name), "raw_type": typ, "schema": tool},
            replace=True,
        )

    text = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[local proxy truncated tool catalog; ask for tool_search if required tool is missing]"
    return text, registry


def _namespace_tool_doc(
    idx: int,
    name: Any,
    desc: str,
    tool: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ns_name = str(name or f"namespace_{idx}")
    ns_entry: dict[str, Any] = {"type": "namespace", "name": ns_name, "description": desc, "tools": []}
    for sub_idx, sub in enumerate(tool.get("tools") or []):
        if not isinstance(sub, dict):
            continue
        sub_name = str(sub.get("name") or f"tool_{sub_idx}")
        full_name = f"{ns_name}.{sub_name}"
        sub_typ = str(sub.get("type") or "function")
        call_type = "custom" if sub_typ == "custom" else "function"
        sub_doc = {
            "name": full_name,
            "namespace": ns_name,
            "native_name": sub_name,
            "aliases": [sub_name, f"{ns_name}__{sub_name}"],
            "native_call": {"namespace": ns_name, "name": sub_name},
            "type": sub_typ,
            "description": sub.get("description") or "",
            "strict": sub.get("strict", False),
            "parameters": sub.get("parameters") or {},
        }
        if "format" in sub:
            sub_doc["format"] = sub.get("format")
        ns_entry["tools"].append(_compact_tool_doc(sub_doc))
        # Native Codex function_call items for namespace tools use
        # {"namespace": ns_name, "name": sub_name}; the dotted name is only a
        # catalog/LLM hint. Register every alias but always emit the native
        # split form to Codex.
        spec = {"call_type": call_type, "output_name": sub_name, "raw_type": sub_typ, "namespace": ns_name, "schema": sub}
        _register_tool_alias(registry, full_name, spec, replace=True)
        _register_tool_alias(registry, f"{ns_name}__{sub_name}", spec)
        _register_tool_alias(registry, sub_name, spec)
    return _compact_tool_doc(ns_entry)


def _implicit_tool_name(idx: int, typ: str) -> str:
    if typ == "tool_search":
        return "tool_search"
    if typ == "web_search":
        return "web_search"
    return f"tool_{idx}"


def _single_tool_doc(name: str, typ: str, desc: str, tool: dict[str, Any]) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "name": name,
        "type": typ,
        "description": desc,
        "strict": tool.get("strict", False),
    }
    if "parameters" in tool or "function" in tool:
        doc["parameters"] = tool.get("parameters") or (tool.get("function") or {}).get("parameters") or {}
    if "format" in tool:
        doc["format"] = tool.get("format")
        doc["input_field"] = "input"
    if typ == "tool_search":
        doc["parameters"] = tool.get("parameters") or {}
    if typ == "web_search":
        doc["note"] = "Do not call upstream/server-side web_search directly. This local proxy rewrites web_search requests to tool_search so Codex can expose local browser/chrome/node_repl/shell tools for web fetching."
        doc["local_proxy_rewrite"] = "web_search -> tool_search"
    return doc


def summarize_client_tools(tools: Any, max_tools: int = 30) -> str:
    catalog, _ = build_client_tool_catalog(tools, max_chars=30_000)
    return catalog


def build_typed_tool_call_schema(
    tool_registry: dict[str, dict[str, Any]] | None,
    *,
    allow_generic_fallback: bool = False,
    max_tools: int = 96,
) -> dict[str, Any]:
    """Build an upstream-compatible item schema for emit_value.tool_calls.

    We previously embedded a per-tool ``oneOf`` schema here.  That improved
    local guidance, but Packy/New API rejects function schemas containing
    ``oneOf`` under array items for several models:

        Invalid schema ... ('properties', 'tool_calls', 'items'), 'oneOf' is not permitted

    Tool calls therefore became impossible whenever the native Codex registry
    was non-empty.  Keep the schema one-level and upstream-compatible, but still
    mirror Codex's structured-tool idea by constraining scalar fields with
    enums when a registry is available.  This is weaker than a per-tool oneOf,
    but much better than asking the model to infer names from prose.
    """
    return _generic_tool_call_schema(tool_registry or {}, allow_generic_fallback=allow_generic_fallback, max_tools=max_tools)


def _typed_tool_call_variants(registry: dict[str, dict[str, Any]], *, max_tools: int) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    specs = sorted(
        (spec for spec in registry.values() if isinstance(spec, dict)),
        key=lambda spec: (str(spec.get("namespace") or ""), str(spec.get("output_name") or ""), str(spec.get("raw_type") or "")),
    )
    for spec in specs:
        namespace = str(spec.get("namespace") or "")
        output_name = str(spec.get("output_name") or "").strip()
        raw_type = str(spec.get("raw_type") or spec.get("call_type") or "function")
        if not output_name:
            continue
        key = (namespace, output_name, raw_type)
        if key in seen:
            continue
        seen.add(key)
        variants.append(_typed_tool_call_variant(spec, namespace=namespace, output_name=output_name, raw_type=raw_type))
        if len(variants) >= max_tools:
            break
    return variants


def _typed_tool_call_variant(spec: dict[str, Any], *, namespace: str, output_name: str, raw_type: str) -> dict[str, Any]:
    call_type = str(spec.get("call_type") or "function")
    if raw_type == "custom" or output_name == "apply_patch":
        call_type = "custom"
    elif raw_type in {"tool_search", "web_search"}:
        call_type = raw_type

    properties: dict[str, Any] = {
        "type": {"type": "string", "enum": [call_type]},
        "namespace": {"type": "string", "enum": [namespace]},
        "name": {"type": "string", "enum": [output_name]},
        "arguments": _arguments_schema_for_spec(spec, call_type=call_type, raw_type=raw_type),
        "input": _input_schema_for_spec(call_type),
    }
    return {
        "type": "object",
        "properties": properties,
        "required": ["type", "namespace", "name", "arguments", "input"],
        "additionalProperties": False,
    }


def _arguments_schema_for_spec(spec: dict[str, Any], *, call_type: str, raw_type: str) -> dict[str, Any]:
    if call_type == "custom":
        return _empty_object_schema()
    raw_schema = spec.get("schema") if isinstance(spec.get("schema"), dict) else {}
    parameters = raw_schema.get("parameters") if isinstance(raw_schema, dict) else {}
    if raw_type == "web_search":
        # Bridge web_search intent to local tool_search semantics.  The parser
        # accepts this shape and rewrites it into a client-side tool_search call.
        parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search/fetch request to satisfy locally via Codex tools."},
                "limit": {"type": "integer", "description": "Maximum local discovery results."},
            },
            "required": ["query"],
            "additionalProperties": False,
        }
    if not isinstance(parameters, dict) or not parameters:
        return _empty_object_schema()
    return _stricten_json_schema(parameters)


def _input_schema_for_spec(call_type: str) -> dict[str, Any]:
    if call_type == "custom":
        return {"type": "string", "description": "Raw FREEFORM/custom tool input. For apply_patch this is the full patch text."}
    return {"type": "string", "enum": [""], "description": "Must be empty for non-custom/function tools."}


def _empty_object_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def _generic_tool_call_schema(
    tool_registry: dict[str, dict[str, Any]] | None = None,
    *,
    allow_generic_fallback: bool = False,
    max_tools: int = 96,
) -> dict[str, Any]:
    registry = tool_registry or {}
    type_schema: dict[str, Any] = {"type": "string", "enum": ["auto", "function", "custom", "tool_search", "web_search"]}
    namespace_schema: dict[str, Any] = {"type": "string"}
    name_schema: dict[str, Any] = {"type": "string"}
    if registry and not allow_generic_fallback:
        names, namespaces, call_types = _tool_schema_enums(registry, max_tools=max_tools)
        if names:
            name_schema = {
                "type": "string",
                "enum": names,
                "description": "Choose one exact local Codex tool name from this enum. For namespace/MCP tools, set namespace separately when applicable.",
            }
        if namespaces:
            namespace_schema = {
                "type": "string",
                "enum": namespaces,
                "description": "Use the exact namespace for namespace/MCP tools, otherwise empty string.",
            }
        if call_types:
            type_schema = {
                "type": "string",
                "enum": call_types,
                "description": "Native Codex output item kind for this local tool call.",
            }
    return {
        "type": "object",
        "properties": {
            "type": type_schema,
            "namespace": namespace_schema,
            "name": name_schema,
            "arguments": {
                "type": "string",
                "description": "Legacy fallback: JSON string arguments matching the requested tool schema.",
            },
            "input": {"type": "string"},
        },
        "required": ["type", "namespace", "name", "arguments", "input"],
        "additionalProperties": False,
    }


def _tool_schema_enums(registry: dict[str, dict[str, Any]], *, max_tools: int) -> tuple[list[str], list[str], list[str]]:
    names: list[str] = []
    namespaces: list[str] = [""]
    call_types: list[str] = []
    seen_specs: set[tuple[str, str, str]] = set()
    for alias, spec in sorted(registry.items()):
        if not isinstance(spec, dict):
            continue
        output_name = str(spec.get("output_name") or alias or "").strip()
        namespace = str(spec.get("namespace") or "").strip()
        raw_type = str(spec.get("raw_type") or spec.get("call_type") or "function").strip()
        call_type = str(spec.get("call_type") or "function").strip()
        if raw_type == "custom" or output_name == "apply_patch":
            call_type = "custom"
        elif raw_type in {"tool_search", "web_search"}:
            call_type = raw_type
        key = (namespace, output_name, call_type)
        if not output_name or key in seen_specs:
            continue
        seen_specs.add(key)
        _append_unique(names, output_name)
        if namespace:
            _append_unique(names, f"{namespace}.{output_name}")
            _append_unique(names, f"{namespace}__{output_name}")
            _append_unique(names, output_name)
            _append_unique(namespaces, namespace)
        _append_unique(call_types, call_type)
        if len(seen_specs) >= max_tools:
            break
    # Keep compatibility with discovery/rewrite paths when those are present.
    if "tool_search" in registry:
        _append_unique(names, "tool_search")
        _append_unique(call_types, "tool_search")
    if "web_search" in registry:
        _append_unique(names, "web_search")
        _append_unique(call_types, "web_search")
    if not call_types:
        call_types = ["auto", "function", "custom", "tool_search", "web_search"]
    return names[: max_tools * 3], namespaces[: max_tools + 1], call_types


def _append_unique(values: list[str], value: str) -> None:
    value = str(value or "")
    if value not in values:
        values.append(value)


def _stricten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make nested tool parameter schemas safer for strict function outputs."""
    out = deepcopy(schema)
    if not isinstance(out, dict):
        return _empty_object_schema()
    _stricten_schema_node(out)
    if out.get("type") != "object":
        out = {"type": "object", "properties": {"value": out}, "required": ["value"], "additionalProperties": False}
    out.setdefault("properties", {})
    out.setdefault("required", list((out.get("properties") or {}).keys()))
    out.setdefault("additionalProperties", False)
    return out


def _stricten_schema_node(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _stricten_schema_node(item)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or isinstance(node.get("properties"), dict):
        properties = node.setdefault("properties", {})
        if isinstance(properties, dict):
            original_required = {str(k) for k in node.get("required", [])}
            for prop_name, prop_schema in properties.items():
                if str(prop_name) not in original_required:
                    _make_nullable_schema(prop_schema)
            node["required"] = list(dict.fromkeys([str(k) for k in node.get("required", [])] + [str(k) for k in properties.keys()]))
        node.setdefault("additionalProperties", False)
    for key in ("properties", "$defs", "definitions"):
        child = node.get(key)
        if isinstance(child, dict):
            for value in child.values():
                _stricten_schema_node(value)
    for key in ("items", "additionalProperties"):
        _stricten_schema_node(node.get(key))
    for key in ("anyOf", "oneOf", "allOf"):
        _stricten_schema_node(node.get(key))


def _make_nullable_schema(node: Any) -> None:
    if not isinstance(node, dict):
        return
    typ = node.get("type")
    if isinstance(typ, str):
        if typ != "null":
            node["type"] = [typ, "null"]
        return
    if isinstance(typ, list):
        if "null" not in typ:
            node["type"] = [*typ, "null"]
        return
    enum = node.get("enum")
    if isinstance(enum, list) and None not in enum:
        node["enum"] = [*enum, None]
        return
    if not any(key in node for key in ("anyOf", "oneOf", "allOf")):
        node["anyOf"] = [{"type": "null"}, deepcopy(node)]


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            obj = json.loads(value)
            return obj if isinstance(obj, dict) else {"input": obj}
        except Exception:
            return {"query": value}
    return {}


def _local_tool_search_arguments(call: dict[str, Any]) -> str:
    """Convert any attempted web_search into a local tool_search request.

    The goal is not to use the upstream Responses built-in web_search. Instead,
    ask Codex to expose local browser/chrome/node_repl/shell/curl tools, then the
    model can fetch/search locally on the following turn.
    """
    obj = _json_object(call.get("arguments") if call.get("arguments") is not None else call.get("input"))
    action = _json_object(obj.get("action")) if "action" in obj else obj
    query = ""
    if isinstance(action.get("query"), str):
        query = action["query"]
    elif isinstance(action.get("queries"), list):
        query = " ".join(str(x) for x in action["queries"][:3])
    elif isinstance(action.get("url"), str):
        query = "open/fetch url " + action["url"]
    elif isinstance(obj.get("query"), str):
        query = obj["query"]
    else:
        query = json.dumps(obj, ensure_ascii=False) if obj else "web search fetch current information"
    query = query.strip() or "web search fetch current information"
    local_query = (
        "local web search/fetch using browser chrome node_repl playwright curl PowerShell: "
        + query
    )
    return json.dumps({"query": local_query, "limit": 8}, ensure_ascii=False, separators=(",", ":"))


def resolve_bridge_tool_call(
    call: dict[str, Any],
    tool_registry: dict[str, dict[str, Any]] | None = None,
    cfg: Settings = settings,
) -> BridgeToolCall | None:
    raw_name = _call_name(call)
    raw_namespace = _call_namespace(call)
    if not raw_name:
        return None

    requested_type = str(call.get("type") or call.get("call_type") or "auto").lower()
    registry = tool_registry or {}
    spec = _resolve_tool_spec(raw_name, raw_namespace, registry)
    discovery_request = raw_name == "tool_search" or requested_type == "tool_search"
    if not spec and not cfg.tool_bridge_allow_unknown_tools:
        if not discovery_request:
            return None
        spec = {"call_type": "tool_search", "output_name": "tool_search", "raw_type": "tool_search"}
    fallback_namespace, fallback_name = _split_namespace_name(raw_name, raw_namespace)
    if raw_namespace:
        fallback_namespace = raw_namespace
    call_type = requested_type if requested_type in {"function", "custom", "tool_search", "web_search"} else str(spec.get("call_type") or "function")
    output_name = str(spec.get("output_name") or fallback_name or raw_name)
    namespace = str(spec.get("namespace") or fallback_namespace or "") or None

    if spec.get("raw_type") == "custom" or output_name == "apply_patch" or raw_name == "apply_patch":
        call_type = "custom"
    if spec.get("raw_type") == "tool_search":
        call_type = "tool_search"
    if spec.get("raw_type") == "web_search" or call_type == "web_search" or raw_name == "web_search":
        # Never route to upstream/server-side web_search. Convert it to native
        # Codex tool_search so local browser/chrome/node_repl/shell tools can be
        # exposed and used without normal web_search billing semantics.
        call_type = "tool_search"
        output_name = "tool_search"
        arguments = _local_tool_search_arguments(call)
    else:
        arguments = _custom_input(call) if call_type == "custom" else _function_arguments(call)
    return BridgeToolCall(
        id="call_" + uuid.uuid4().hex[:24],
        name=output_name,
        arguments=arguments,
        call_type=call_type,
        requested_name=raw_name,
        namespace=namespace,
    )


def _resolve_tool_spec(raw_name: str, raw_namespace: str, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve catalog aliases for ordinary and namespace/MCP tools.

    Codex native namespace calls use split fields:
    {"namespace":"mcp__node_repl","name":"js"}. The upstream bridge model may
    instead emit mcp__node_repl.js or mcp__node_repl__js because those are the
    safest unambiguous hints in our text catalog. Accept all three forms.
    """
    candidates: list[str] = []
    if raw_namespace:
        candidates.extend([f"{raw_namespace}.{raw_name}", f"{raw_namespace}__{raw_name}"])
    candidates.append(raw_name)

    split_namespace, split_name = _split_namespace_name(raw_name, raw_namespace)
    if split_namespace and split_name:
        candidates.extend([f"{split_namespace}.{split_name}", f"{split_namespace}__{split_name}", split_name])

    for candidate in candidates:
        spec = registry.get(candidate.lower())
        if spec:
            return spec
    fuzzy = _resolve_tool_spec_fuzzy(raw_name, raw_namespace, registry)
    if fuzzy:
        return fuzzy
    return {}


def _call_name(call: dict[str, Any]) -> str:
    """Accept common non-native tool-call spellings from less reliable models.

    gpt-5.6-sol sometimes follows the semantic contract but emits a display-ish
    key such as ``tool_name``/``function_name`` or nests the native call under
    ``tool``/``native_call``.  Native Codex itself is tolerant at the item layer;
    the bridge should not turn those recoverable shapes into a user-visible
    "[local proxy] ... no valid tool_calls" answer.
    """
    for key in ("name", "tool_name", "function_name", "tool", "function"):
        value = call.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("name") or value.get("tool_name") or value.get("function_name")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    native_call = call.get("native_call")
    if isinstance(native_call, dict):
        nested = native_call.get("name")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    requested_type = str(call.get("type") or call.get("call_type") or "").lower()
    if requested_type in {"tool_search", "web_search"}:
        return requested_type
    return ""


def _call_namespace(call: dict[str, Any]) -> str:
    for key in ("namespace", "tool_namespace"):
        value = call.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("tool", "function", "native_call"):
        value = call.get(key)
        if isinstance(value, dict):
            nested = value.get("namespace")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _resolve_tool_spec_fuzzy(raw_name: str, raw_namespace: str, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    needle = _tool_name_key(raw_name)
    if not needle:
        return {}
    ns_key = _tool_name_key(raw_namespace)
    matches: list[dict[str, Any]] = []
    for alias, spec in registry.items():
        candidates = {
            alias,
            str(spec.get("output_name") or ""),
            str(spec.get("namespace") or ""),
        }
        if spec.get("namespace") and spec.get("output_name"):
            candidates.add(f"{spec.get('namespace')}.{spec.get('output_name')}")
            candidates.add(f"{spec.get('namespace')} {spec.get('output_name')}")
        normalized = {_tool_name_key(c) for c in candidates if c}
        if needle in normalized or any(needle and (needle in c or c in needle) for c in normalized):
            if ns_key and ns_key not in {_tool_name_key(str(spec.get("namespace") or "")), _tool_name_key(alias)}:
                continue
            matches.append(spec)
    if len(matches) == 1:
        return matches[0]
    return {}


def _tool_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _split_namespace_name(raw_name: str, raw_namespace: str = "") -> tuple[str | None, str | None]:
    """Best-effort namespace split used when a tool was not in the registry.

    This keeps future MCP servers working even before their metadata is seen:
    - mcp__playwright.browser_tabs -> (mcp__playwright, browser_tabs)
    - mcp__node_repl__js          -> (mcp__node_repl, js)
    - codex_app.read_thread_terminal -> (codex_app, read_thread_terminal)
    """
    raw_name = (raw_name or "").strip()
    raw_namespace = (raw_namespace or "").strip()
    if raw_namespace:
        return raw_namespace, raw_name
    if "." in raw_name:
        ns, name = raw_name.rsplit(".", 1)
        if ns and name and _looks_like_namespace(ns):
            return ns, name
    # MCP server names themselves start with mcp__ and the namespace/tool
    # separator is the final double-underscore.
    if raw_name.startswith("mcp__") and "__" in raw_name.removeprefix("mcp__"):
        ns, name = raw_name.rsplit("__", 1)
        if ns and name:
            return ns, name
    return None, raw_name


def _looks_like_namespace(value: str) -> bool:
    if value.startswith("mcp__"):
        return True
    if value in {"codex_app", "multi_tool_use", "functions", "tool_search"}:
        return True
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value or ""))


def _custom_input(call: dict[str, Any]) -> str:
    value = call.get("input")
    if value is None:
        value = call.get("arguments", "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _function_arguments(call: dict[str, Any]) -> str:
    value = call.get("arguments")
    if value is None:
        value = call.get("input", {})
    if not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    stripped = value.strip()
    if not stripped:
        return "{}"
    if stripped.startswith("{") or stripped.startswith("["):
        return value
    return json.dumps({"input": value}, ensure_ascii=False)



def _tool_search_query(call: BridgeToolCall) -> str:
    try:
        obj = json.loads(call.arguments or "{}")
        if isinstance(obj, dict):
            return str(obj.get("query") or obj.get("input") or "")
    except Exception:
        pass
    return str(call.arguments or "")


def _make_tool_search_call(query: str, limit: int = 20) -> BridgeToolCall:
    return BridgeToolCall(
        id="call_" + uuid.uuid4().hex[:24],
        name="tool_search",
        arguments=json.dumps({"query": query, "limit": limit}, ensure_ascii=False, separators=(",", ":")),
        call_type="tool_search",
        requested_name="tool_search:auto_expand",
    )


def expand_deferred_tool_searches(
    calls: list[BridgeToolCall],
    tool_registry: dict[str, dict[str, Any]] | None = None,
    cfg: Settings = settings,
) -> list[BridgeToolCall]:
    """Ask native Codex tool_search to expose richer deferred MCP tools.

    Native Codex only executes the tool_search calls the model explicitly asks
    for. Keep that behavior by default; the optional expansion flag is retained
    for targeted experiments but should stay disabled for native-like operation.
    """
    if not cfg.tool_bridge_auto_expand_search:
        return calls
    if not calls:
        return calls
    registry = tool_registry or {}
    if registry and "tool_search" not in registry:
        return calls

    existing_searches = [c for c in calls if c.call_type == "tool_search" or c.name == "tool_search"]
    if not existing_searches:
        return calls

    combined = " ".join(_tool_search_query(c).lower() for c in existing_searches)
    trigger_words = (
        "browser", "playwright", "chrome", "jshook", "mcp", "node", "node_repl",
        "cookie", "localstorage", "sessionstorage", "dom", "javascript", "network",
        "tab", "authenticated", "login", "current page", "web", "hook", "memory",
        "浏览器", "页面", "登录", "工具", "cookie",
    )
    if combined.strip() and not any(word in combined for word in trigger_words):
        return calls

    broad_queries = [
        "playwright browser navigate evaluate tabs network requests cookies localStorage sessionStorage DOM JavaScript current page",
        "chrome browser current tab cookies localStorage sessionStorage evaluate JavaScript network requests authenticated session",
        "node_repl js playwright chrome browser fetch HTTP DOM localStorage cookie automation",
        "jshook call_tool route_tool activate_tools browser hook network intercept memory coverage runtime JavaScript",
        "computer use screenshot click type browser window desktop automation",
    ]
    seen = {_tool_search_query(c).strip().lower() for c in existing_searches}
    out = list(calls)
    for query in broad_queries:
        if query.lower() in seen:
            continue
        out.append(_make_tool_search_call(query))
        break
    return out

def parse_function_arguments(
    raw: str,
    cfg: Settings = settings,
    tool_registry: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, bool, bool, str, list[BridgeToolCall]]:
    obj, malformed, repaired = _load_arguments_object(raw)
    mode = obj.get("mode") or ("answer" if cfg.answer_field in obj else "answer")
    answer = obj.get(cfg.answer_field, "")
    tool_calls: list[BridgeToolCall] = []

    if mode == "tool_call":
        raw_calls = _iter_call_objects(obj.get("tool_calls"))
        for call in raw_calls:
            resolved = resolve_bridge_tool_call(call, tool_registry)
            if resolved:
                tool_calls.append(resolved)
        tool_calls = expand_deferred_tool_searches(tool_calls, tool_registry, cfg)
        if not tool_calls:
            fallback = _fallback_tool_search_for_invalid_calls(raw_calls, answer)
            resolved = resolve_bridge_tool_call(fallback, tool_registry)
            if resolved:
                tool_calls = [resolved]
                # Keep recovery invisible in the UI. Codex native tool discovery
                # appears as a tool event, not repeated assistant prose.
                answer = answer if isinstance(answer, str) else ""
            else:
                mode = "answer"
                answer = answer or "需要继续操作，但代理没有拿到可执行的本地工具调用。请重试一次。"

    if not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)
    if len(answer) > cfg.max_answer_chars:
        answer = answer[: cfg.max_answer_chars] + "\n[local proxy truncated answer at max_answer_chars]"
    return answer, malformed, repaired, str(mode), tool_calls


def _load_arguments_object(raw: str) -> tuple[dict[str, Any], bool, bool]:
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("function arguments JSON is not an object") from None
        return obj, False, False
    except Exception:
        repaired_raw = raw.strip()
        last = max(repaired_raw.rfind("}"), repaired_raw.rfind("]"))
        if last < 0:
            raise
        obj = json.loads(repaired_raw[: last + 1])
        if not isinstance(obj, dict):
            raise ValueError("function arguments JSON is not an object") from None
        return obj, True, True


def _iter_call_objects(calls: Any) -> list[dict[str, Any]]:
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list):
        return []
    out: list[dict[str, Any]] = []
    for call in calls[:8]:
        if not isinstance(call, dict):
            continue
        out.append(_normalize_call_object(call))
    return out


def _normalize_call_object(call: dict[str, Any]) -> dict[str, Any]:
    out = dict(call)
    # Some models put the actual function arguments under parameters/input_json
    # even though the bridge schema asks for `arguments`.
    if out.get("arguments") in (None, ""):
        for key in ("parameters", "args", "input_json"):
            if key in out:
                out["arguments"] = out.get(key)
                break
    # Some models nest the whole call under a `tool`/`function` object.
    for key in ("tool", "function", "native_call"):
        nested = out.get(key)
        if not isinstance(nested, dict):
            continue
        out.setdefault("namespace", nested.get("namespace"))
        out.setdefault("name", nested.get("name") or nested.get("tool_name") or nested.get("function_name"))
        if out.get("arguments") in (None, ""):
            out["arguments"] = nested.get("arguments") or nested.get("parameters") or nested.get("args")
    return out


def _fallback_tool_search_for_invalid_calls(raw_calls: list[dict[str, Any]], answer: Any) -> dict[str, Any]:
    labels: list[str] = []
    for call in raw_calls:
        name = _call_name(call)
        namespace = _call_namespace(call)
        requested_type = str(call.get("type") or call.get("call_type") or "").strip()
        bits = [bit for bit in (namespace, name, requested_type) if bit]
        arg_preview = _argument_preview(call.get("arguments") if call.get("arguments") is not None else call.get("input"))
        if arg_preview:
            bits.append(arg_preview)
        if bits:
            labels.append(" ".join(bits))
    if not labels and isinstance(answer, str) and answer.strip():
        labels.append(answer.strip()[:500])
    query = "Resolve the correct local Codex tool for this requested action: "
    query += "; ".join(labels[:4]) if labels else "browser chrome local tools terminal file editing"
    query += " chrome browser current tab DOM page text extract settings local integration"
    return {
        "type": "tool_search",
        "namespace": "",
        "name": "tool_search",
        "arguments": json.dumps({"query": query, "limit": 8}, ensure_ascii=False, separators=(",", ":")),
        "input": "",
    }


def _argument_preview(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value[:500]
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:500]
    except Exception:
        return str(value)[:500]
