from __future__ import annotations

from typing import Any, Iterable

# Reserved Responses/extension namespaces. MCP namespaces are `mcp__` + server
# (and `mcp__codex_apps__` + connector). Lookup is Eq on (namespace, name);
# Display concatenates them with no extra separator.
RESERVED_NAMESPACES = {
    "functions",
    "web",
    "image_gen",
    "skills",
    "memories",
    "memory",
    "clock",
    "collaboration",
    "multi_agent_v1",
    "extension",
}

MCP_PREFIX = "mcp__"
KNOWN_NAMESPACES = RESERVED_NAMESPACES | {"mcp", "codex_app"}


def _clean(value: str | None) -> str:
    return (value or "").strip()


def flat_tool_name(name: str, namespace: str | None) -> str:
    """Codex ToolName Display / flat_tool_name: `{namespace}{name}`."""
    tool = _clean(name)
    ns = _clean(namespace)
    return f"{ns}{tool}" if ns else tool


def mcp_join(name: str, namespace: str | None) -> str:
    """MCP hook join: trimmed namespace + `__` + trimmed name."""
    tool = _clean(name).lstrip("_")
    ns = _clean(namespace).rstrip("_")
    if not ns:
        return tool
    if not tool:
        return ns
    return f"{ns}__{tool}"


def _is_reserved_namespace(token: str) -> bool:
    return _clean(token).lower().rstrip("._/") in RESERVED_NAMESPACES


def _is_mcp_namespace(token: str) -> bool:
    raw = _clean(token)
    return raw.lower() == "mcp" or raw.startswith(MCP_PREFIX)


def looks_like_namespace(token: str) -> bool:
    raw = _clean(token)
    if not raw:
        return False
    if _is_reserved_namespace(raw) or _is_mcp_namespace(raw):
        return True
    return raw.endswith(".") or raw.endswith("/") or raw.endswith("__")


def _unjoin_mcp_flat(raw: str) -> tuple[str, str] | None:
    """Split `mcp__server__tool` but never `mcp__tool` (two segments only)."""
    if not raw.startswith(MCP_PREFIX):
        return None
    rest = raw[len(MCP_PREFIX) :]
    if not rest or "__" not in rest:
        return None
    server, tool = rest.rsplit("__", 1)
    server, tool = server.strip(), tool.strip()
    if not server or not tool:
        return None
    return tool, MCP_PREFIX + server


def split_tool_identity(name: str, namespace: str | None = None) -> tuple[str, str | None]:
    """Recover (callable_name, namespace) from a model-emitted string.

    Splits `functions.exec` / `web.run` / `mcp__python.exec` on `.` or `/`.
    Does not treat `__` as a generic separator: `mcp__cua_repl` stays intact.
    Flattened MCP joins with 3+ segments (`mcp__server__tool`) unjoin.
    """
    raw = _clean(name)
    ns = _clean(namespace) or None
    if not raw:
        return "", ns

    for sep in (".", "/"):
        if sep not in raw:
            continue
        left, right = raw.split(sep, 1)
        left, right = left.strip(), right.strip()
        if not left or not right:
            continue
        if ns and ns not in {left, right} and not ns.rstrip("._/") == left.rstrip("._/"):
            continue
        if looks_like_namespace(left) or (ns and ns.rstrip("._/") == left.rstrip("._/")):
            return right, left

    if ns is None:
        unjoined = _unjoin_mcp_flat(raw)
        if unjoined:
            return unjoined
    return raw, ns


def is_namespace_only(name: str, namespace: str | None = None) -> bool:
    tool, ns = split_tool_identity(name, namespace)
    if not tool:
        return True
    if ns and tool.lower() == ns.lower():
        return True
    if ns is None and looks_like_namespace(tool):
        if tool.startswith(MCP_PREFIX) and "__" in tool[len(MCP_PREFIX) :]:
            return False
        return True
    return False


def catalog_aliases(name: str, namespace: str | None) -> list[str]:
    tool = _clean(name)
    ns = _clean(namespace) or None
    aliases: list[str] = []
    if tool:
        aliases.append(tool)
    if ns and tool:
        aliases.append(flat_tool_name(tool, ns))
        if not ns.endswith((".", "/", "_")):
            aliases.extend([f"{ns}.{tool}", f"{ns}/{tool}", f"{ns}__{tool}"])
        aliases.append(mcp_join(tool, ns))
        stripped = ns.rstrip("_")
        if stripped and stripped != ns:
            aliases.append(mcp_join(tool, stripped))
    return list(dict.fromkeys(a for a in aliases if a))


def recover_tool_name(obj: dict[str, Any]) -> tuple[str, str | None]:
    name = str(obj.get("name") or obj.get("tool") or obj.get("function") or obj.get("tool_name") or obj.get("callable_name") or "")
    namespace = str(
        obj.get("namespace")
        or obj.get("tool_namespace")
        or obj.get("callable_namespace")
        or ""
    )
    if isinstance(obj.get("function"), dict):
        nested = obj["function"]
        name = str(nested.get("name") or nested.get("tool_name") or name)
        namespace = str(nested.get("namespace") or nested.get("tool_namespace") or namespace)
    tool, ns = split_tool_identity(name, namespace or None)
    if is_namespace_only(tool, ns):
        for key in ("subtool", "tool_name", "method", "action", "callable_name"):
            extra = obj.get(key)
            if not isinstance(extra, str):
                continue
            extra = extra.strip()
            if extra and not looks_like_namespace(extra):
                holder = tool if looks_like_namespace(tool) else ns
                return split_tool_identity(extra, holder)
        if looks_like_namespace(tool) and not ns:
            return tool, None
        return "", ns
    return tool, ns


def _unique_specs(specs: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[int] = set()
    for spec in specs:
        marker = id(spec)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(spec)
    return out


def resolve_catalog_tool(catalog: Any, name: str, namespace: str | None = None) -> Any | None:
    """Map a model-emitted identity onto the catalog spec. Replay spec.name/namespace as-is."""
    specs = _unique_specs(getattr(catalog, "specs", {}).values())
    if not specs:
        return None
    index: dict[str, list[Any]] = {}
    for spec in specs:
        for alias in catalog_aliases(spec.name, spec.namespace):
            index.setdefault(alias.lower(), []).append(spec)

    tool, ns = split_tool_identity(name, namespace)
    keys = list(dict.fromkeys([_clean(name), *catalog_aliases(tool, ns), *catalog_aliases(_clean(name), _clean(namespace) or None)]))

    def _hits_for(keys_iter: list[str]) -> list[Any]:
        found: list[Any] = []
        seen: set[int] = set()
        for key in keys_iter:
            for spec in index.get(key.lower(), []):
                marker = id(spec)
                if marker in seen:
                    continue
                seen.add(marker)
                found.append(spec)
        return found

    hits = _hits_for([key for key in keys if key])
    chosen = _disambiguate(hits, tool, ns)
    if chosen is not None:
        return chosen

    if tool:
        shorts = [spec for spec in specs if str(spec.name).lower() == tool.lower()]
        chosen = _disambiguate(shorts, tool, ns)
        if chosen is not None:
            return chosen

    holder = _clean(name) if looks_like_namespace(_clean(name)) else (tool if looks_like_namespace(tool) else "")
    if holder.startswith(MCP_PREFIX) and holder != MCP_PREFIX:
        children = [spec for spec in specs if (spec.namespace or "") == holder]
        if len(children) == 1:
            return children[0]
        tail = holder.split("__")[-1].split(".")[-1].split("/")[-1]
        named = [spec for spec in children if str(spec.name).lower() == tail.lower()]
        if len(named) == 1:
            return named[0]
    return None


def _disambiguate(hits: list[Any], tool: str, ns: str | None) -> Any | None:
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    if ns:
        matched = [spec for spec in hits if (spec.namespace or "") == ns]
        if len(matched) == 1:
            return matched[0]
        matched_ci = [spec for spec in hits if (spec.namespace or "").lower() == ns.lower()]
        if len(matched_ci) == 1:
            return matched_ci[0]
        stripped = [spec for spec in hits if (spec.namespace or "").rstrip("._/") == ns.rstrip("._/")]
        if len(stripped) == 1:
            return stripped[0]
    if ns and _is_reserved_namespace(ns):
        plain = [spec for spec in hits if not spec.namespace]
        if len(plain) == 1:
            return plain[0]
        named = [spec for spec in hits if str(spec.name).lower() == tool.lower() and not spec.namespace]
        if len(named) == 1:
            return named[0]
    if tool:
        named = [spec for spec in hits if str(spec.name).lower() == tool.lower()]
        if ns:
            ns_named = [spec for spec in named if (spec.namespace or "").lower() == ns.lower()]
            if len(ns_named) == 1:
                return ns_named[0]
        plain = [spec for spec in named if not spec.namespace]
        if len(plain) == 1:
            return plain[0]
        if len(named) == 1:
            return named[0]
    return None
