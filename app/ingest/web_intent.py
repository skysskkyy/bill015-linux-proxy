from __future__ import annotations

import re
from typing import Any

from ..protocol.models import BridgeToolCall, Catalog, new_call_id

LIVE_WEB_RE = re.compile(
    r"(上网查|联网搜索|打开\s*https?://|browse\s+https?://|search the web|look up (the )?(latest|current)|"
    r"what's the (latest|current)|open (the )?url|fetch https?://)",
    re.I,
)
PROJECT_SEARCH_RE = re.compile(r"(修复|fix).{0,24}(网络搜索|web search|search)", re.I)
BROWSER_HINTS = ("browser", "chrome", "playwright", "jshook", "node_repl", "fetch", "http")


def detect_web_intent(text: str) -> bool:
    blob = text or ""
    if PROJECT_SEARCH_RE.search(blob):
        return False
    return bool(LIVE_WEB_RE.search(blob))


def _has_tool_search(catalog: Catalog) -> bool:
    return any(spec.call_type == "tool_search" or spec.name == "tool_search" for spec in catalog.specs.values())


def _has_browser_tool(catalog: Catalog) -> bool:
    for spec in catalog.specs.values():
        blob = f"{spec.alias} {spec.name} {spec.description}".lower()
        if any(hint in blob for hint in BROWSER_HINTS):
            if spec.call_type != "tool_search":
                return True
    return False


def local_web_preflight(text: str, catalog: Catalog, history: list[dict[str, Any]] | None = None) -> BridgeToolCall | None:
    if not detect_web_intent(text):
        return None
    if _has_browser_tool(catalog):
        return None
    if not _has_tool_search(catalog):
        return None
    for item in history or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") in {"tool_search_call", "tool_search_output"}:
            return None
    return BridgeToolCall(
        id=new_call_id(),
        name="tool_search",
        arguments="",
        call_type="tool_search",
        execution="client",
        search_arguments={"query": "browser chrome playwright http fetch", "limit": 12},
    )
