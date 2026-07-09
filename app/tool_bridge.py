from __future__ import annotations

import json
import re
import uuid
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


def resolve_bridge_tool_call(call: dict[str, Any], tool_registry: dict[str, dict[str, Any]] | None = None) -> BridgeToolCall | None:
    raw_name = str(call.get("name") or "").strip()
    raw_namespace = str(call.get("namespace") or "").strip()
    if not raw_name:
        return None

    requested_type = str(call.get("type") or call.get("call_type") or "auto").lower()
    registry = tool_registry or {}
    spec = _resolve_tool_spec(raw_name, raw_namespace, registry)
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
    return {}


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
) -> list[BridgeToolCall]:
    """Ask native Codex tool_search to expose richer deferred MCP tools.

    In exploit/emit_value mode the upstream model does not participate in
    Codex's native deferred-tool search loop directly. A narrow query such as
    "Playwright" often exposes only tabs/network tools. When a turn is clearly
    about browser/MCP/jshook/node tooling, append several broad native
    tool_search calls so Codex reveals the same families of tools it would make
    discoverable in a normal session.
    """
    if not calls:
        return calls
    registry = tool_registry or {}
    if registry and "tool_search" not in registry:
        return calls

    existing_searches = [c for c in calls if c.call_type == "tool_search" or c.name == "tool_search"]
    namespace_calls = [c for c in calls if (c.namespace or "").startswith("mcp__") or c.namespace == "codex_app"]
    if not existing_searches and not namespace_calls:
        return calls

    combined = " ".join(_tool_search_query(c).lower() for c in existing_searches)
    combined += " " + " ".join(((c.namespace or "") + " " + c.name).lower() for c in namespace_calls)
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
        seen.add(query.lower())
        # Cap added discovery calls. One original + five broad searches is enough
        # to reveal Browser/Chrome/Node/jshook families without flooding Codex.
        if sum(1 for c in out if c.call_type == "tool_search" or c.name == "tool_search") >= 6:
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
        for call in _iter_call_objects(obj.get("tool_calls")):
            resolved = resolve_bridge_tool_call(call, tool_registry)
            if resolved:
                tool_calls.append(resolved)
        tool_calls = expand_deferred_tool_searches(tool_calls, tool_registry)
        if not tool_calls:
            mode = "answer"
            answer = "[local proxy] tool_call mode requested but no valid tool_calls were provided."

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
        return obj, False, True


def _iter_call_objects(calls: Any) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        return []
    return [call for call in calls[:8] if isinstance(call, dict)]
