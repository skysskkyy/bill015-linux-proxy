from __future__ import annotations

import json
import re
import uuid
from typing import Any

from .config import Settings, settings
from .models import BridgeToolCall, NormalizedRequest

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()\"']+")

ONLINE_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"上网\s*(?:查|搜|搜索|看看|了解|查询)|"
    r"联网\s*(?:查|搜|搜索|看看|查询)|"
    r"网上\s*(?:查|搜|搜索|看看|查询)|"
    r"(?:网页|网站)\s*(?:搜索|查询|检索)|"
    r"(?:浏览器|chrome|playwright)\s*(?:查|搜|搜索|打开|访问|浏览)|"
    r"(?:打开|访问|读取|抓取|浏览)\s*(?:网页|网站|链接|网址|url)|"
    r"(?:帮我|请|麻烦|需要|用|通过|进行).{0,12}网络搜索(?:一下)?|"
    r"search\s+(?:the\s+)?web|"
    r"web\s+search|"
    r"internet\s+search|"
    r"browse\s+(?:the\s+)?(?:web|internet)|"
    r"use\s+(?:a\s+)?(?:browser|chrome|playwright).{0,30}search|"
    r"(?:open|visit|fetch)\s+(?:https?://|www\.)|"
    r"look\s+up.{0,80}(?:online|on\s+the\s+web)"
    r")"
)

POLITE_SEARCH_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?:帮我|请|麻烦).{0,16}(?:搜|搜索|查|查询)(?:一下)?|"
    r"(?:搜|搜索|查|查询)(?:一下)?(?:今天|今日|现在|当前|实时|最新)|"
    r"(?:look\s+up|find\s+out|search\s+for|google|bing)\b"
    r")"
)

FRESHNESS_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"今天|今日|昨天|明天|现在|当前|实时|最新|最近|新闻|价格|汇率|天气|赛程|比分|发布|公告|版本|"
    r"\blatest\b|\bcurrent\b|\btoday\b|\bnews\b|\bprice\b|\brate\b|\bweather\b|\bschedule\b|\bscore\b|\brelease\b"
    r")"
)

SEARCH_VERB_RE = re.compile(r"(?ix)(?:查|查询|搜|搜索|检索|了解|看看|\bsearch\b|\blook\s+up\b|\bfind\b|\bwhat\s+is\b|\bwho\s+is\b)")

LOCAL_CODE_SEARCH_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"项目|仓库|代码|源码|目录|文件|本地|函数|类名|变量|实现|代理|配置|测试|日志|"
    r"\brg\b|\bripgrep\b|\bgrep\b|\brepository\b|\brepo\b|\bworkspace\b|\bcodebase\b|\bsource\b|\bfile\b|\bdirectory\b"
    r")"
)

META_SEARCH_FEATURE_RE = re.compile(r"(?i)(?:网络搜索|web\s+search|browser\s+search).{0,12}(?:功能|机制|实现|对齐|修复|优化|feature|implementation)")

BROWSER_TOOL_HINTS = (
    "browser",
    "chrome",
    "playwright",
    "puppeteer",
    "selenium",
    "jshook",
    "node_repl",
    "node-repl",
    "网页",
    "浏览器",
    "current page",
    "current_page",
    "navigate",
    "navigation",
    "tab",
    "dom",
    "localstorage",
    "sessionstorage",
    "cookie",
    "network",
    "intercept",
)

HTTP_FALLBACK_HINTS = (
    "shell_command",
    "powershell",
    "curl",
    "wget",
    "invoke-webrequest",
    "http",
    "https",
    "fetch",
    "requests",
    "node_repl",
    "node-repl",
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "using",
    "local",
    "web",
    "search",
    "fetch",
    "browser",
    "chrome",
    "playwright",
    "tool",
    "tools",
    "一下",
    "搜索",
    "查询",
    "最新",
    "今天",
}


def detect_local_web_research_intent(text: str) -> bool:
    """Return True when the latest user request needs live/local web research.

    This intentionally stays narrower than "contains the word search".  Codex
    tasks often ask to search the repository or to implement the proxy's own
    "network search feature"; those should remain normal coding/file tasks.
    """
    value = _compact_text(text)
    if not value:
        return False
    if URL_RE.search(value):
        return True
    if META_SEARCH_FEATURE_RE.search(value):
        return False

    has_online = bool(ONLINE_RE.search(value))
    local_codeish = bool(LOCAL_CODE_SEARCH_RE.search(value))
    if has_online:
        return True
    if local_codeish and SEARCH_VERB_RE.search(value):
        return False
    if POLITE_SEARCH_RE.search(value) and not local_codeish:
        return True
    return bool(FRESHNESS_RE.search(value) and SEARCH_VERB_RE.search(value) and not local_codeish)


def local_web_research_query(text: str, *, max_chars: int = 1200) -> str:
    query = _compact_text(text)
    if len(query) > max_chars:
        return query[: max_chars - 1].rstrip() + "…"
    return query or "web search/fetch current information"


def build_local_web_discovery_arguments(query: str, *, limit: int = 20) -> str:
    discovery_query = (
        "local web research browser chrome playwright node_repl jshook HTTP fetch search engine open URL "
        "current page network tools. User task: "
        + local_web_research_query(query)
    )
    return json.dumps({"query": discovery_query, "limit": limit}, ensure_ascii=False, separators=(",", ":"))


def has_tool_search(registry: dict[str, dict[str, Any]] | None) -> bool:
    for alias, spec in (registry or {}).items():
        if str(alias).lower() == "tool_search":
            return True
        if str(spec.get("raw_type") or spec.get("call_type") or "").lower() == "tool_search":
            return True
        if str(spec.get("output_name") or "").lower() == "tool_search":
            return True
    return False


def browser_tool_candidates(registry: dict[str, dict[str, Any]] | None, *, max_tools: int = 16) -> list[str]:
    return _tool_candidates(registry, hints=BROWSER_TOOL_HINTS, max_tools=max_tools, include_shell=False)


def http_tool_candidates(registry: dict[str, dict[str, Any]] | None, *, max_tools: int = 16) -> list[str]:
    return _tool_candidates(registry, hints=(*BROWSER_TOOL_HINTS, *HTTP_FALLBACK_HINTS), max_tools=max_tools, include_shell=True)


def build_local_web_research_guidance(current_request: str, registry: dict[str, dict[str, Any]] | None) -> str:
    if not detect_local_web_research_intent(current_request):
        return ""
    query = local_web_research_query(current_request)
    browser_candidates = browser_tool_candidates(registry)
    fallback_candidates = [name for name in http_tool_candidates(registry) if name not in browser_candidates]
    search_available = has_tool_search(registry)
    lines = [
        "LOCAL WEB RESEARCH POLICY (triggered by the current user request):",
        "- The user appears to need live web information or URL/page access. Do not answer from memory when freshness or page contents matter.",
        "- Never use or request upstream/server-side web_search. All web lookup must be done through local Codex tools so strict-zero usage stays on the BILL-015 bridge.",
    ]
    if browser_candidates:
        lines.extend(
            [
                "- Concrete local browser/web-capable tools are already visible. Call one of them now; do not call tool_search first.",
                "- Prefer these exact local tool names/namespaces: " + ", ".join(browser_candidates),
            ]
        )
    elif search_available:
        lines.extend(
            [
                "- No concrete browser/Chrome/Playwright/node_repl/jshook tool is visible yet, but tool_search is available.",
                "- Call tool_search now with this discovery query, then after tool_search_output immediately use the exposed concrete tool(s) to search/fetch; do not stop after discovery.",
                "- Discovery query: " + json.loads(build_local_web_discovery_arguments(query)).get("query", ""),
            ]
        )
    elif fallback_candidates:
        lines.extend(
            [
                "- No dedicated browser discovery tool is visible. Use an available local HTTP/shell fallback to fetch/search if possible.",
                "- Candidate fallback tools: " + ", ".join(fallback_candidates[:12]),
            ]
        )
    else:
        lines.append("- No local web-capable tool is visible. State that blocker concisely instead of claiming the model itself browsed.")
    lines.append("- Original web task: " + query)
    return "\n".join(lines)


def local_web_research_preflight(n: NormalizedRequest, cfg: Settings = settings) -> BridgeToolCall | None:
    if not getattr(cfg, "tool_bridge_local_web_research_preflight", True):
        return None
    current = n.current_user_request or n.user_input or ""
    if not detect_local_web_research_intent(current):
        return None
    registry = n.tool_registry or {}
    if browser_tool_candidates(registry):
        return None
    if not has_tool_search(registry):
        return None
    query = local_web_research_query(current)
    if _recent_discovery_for_query(n.tool_history, query):
        return None
    return BridgeToolCall(
        id="call_" + uuid.uuid4().hex[:24],
        name="tool_search",
        arguments=build_local_web_discovery_arguments(query),
        call_type="tool_search",
        requested_name="tool_search:local_web_research_preflight",
        namespace=None,
    )


def _tool_candidates(
    registry: dict[str, dict[str, Any]] | None,
    *,
    hints: tuple[str, ...],
    max_tools: int,
    include_shell: bool,
) -> list[str]:
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for alias, spec in (registry or {}).items():
        if not isinstance(spec, dict):
            continue
        raw_type = str(spec.get("raw_type") or spec.get("call_type") or "").lower()
        output_name = str(spec.get("output_name") or alias or "").strip()
        namespace = str(spec.get("namespace") or "").strip()
        if raw_type in {"tool_search", "web_search"} or output_name in {"tool_search", "web_search"}:
            continue
        label = f"{namespace}.{output_name}" if namespace else output_name
        if not output_name or not label:
            continue
        key = (namespace.lower(), output_name.lower())
        if key in seen:
            continue
        haystack = _tool_haystack(alias, spec)
        if not include_shell and "shell_command" in haystack:
            continue
        if any(hint.lower() in haystack for hint in hints):
            seen.add(key)
            out.append(label)
            if len(out) >= max_tools:
                break
    return out


def _tool_haystack(alias: str, spec: dict[str, Any]) -> str:
    parts = [
        alias,
        str(spec.get("namespace") or ""),
        str(spec.get("output_name") or ""),
        str(spec.get("raw_type") or ""),
        str(spec.get("call_type") or ""),
    ]
    schema = spec.get("schema")
    if isinstance(schema, dict):
        for key in ("name", "description", "title"):
            value = schema.get(key)
            if isinstance(value, str):
                parts.append(value)
        function = schema.get("function")
        if isinstance(function, dict):
            parts.extend(str(function.get(key) or "") for key in ("name", "description"))
    return " ".join(parts).lower()


def _recent_discovery_for_query(history: Any, query: str) -> bool:
    if not history:
        return False
    for call in getattr(history, "pending_calls", []) or []:
        if str(getattr(call, "name", "") or "").lower() == "tool_search":
            return True
    needle = _normalized_query_key(query)
    for call in (getattr(history, "calls", []) or [])[-8:]:
        if str(getattr(call, "name", "") or "").lower() != "tool_search":
            continue
        args = str(getattr(call, "arguments", "") or "")
        lowered = args.lower()
        if "local web research" not in lowered and "local web search/fetch" not in lowered:
            continue
        if _query_matches_arguments(needle, query, lowered):
            return True
    return False


def _query_matches_arguments(needle: str, query: str, arguments_lower: str) -> bool:
    if needle and needle in _normalized_query_key(arguments_lower):
        return True
    compact_query = re.sub(r"\s+", "", query.lower())
    if len(compact_query) >= 12 and compact_query[:80] in re.sub(r"\s+", "", arguments_lower):
        return True
    tokens = [t for t in re.findall(r"[\w-]{3,}", query.lower()) if t not in STOPWORDS]
    if not tokens:
        return False
    hits = sum(1 for token in dict.fromkeys(tokens[:12]) if token in arguments_lower)
    return hits >= max(2, min(4, len(set(tokens)) // 2))


def _normalized_query_key(text: str) -> str:
    tokens = [t for t in re.findall(r"[\w-]{3,}", str(text or "").lower()) if t not in STOPWORDS]
    return " ".join(dict.fromkeys(tokens[:16]))


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
