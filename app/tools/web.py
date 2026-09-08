from __future__ import annotations

import json
from typing import Any

from ..config import Settings, settings
from ..protocol.models import BridgeToolCall, Catalog, ToolSpec
from .firecrawl import firecrawl_scrape, firecrawl_search, is_safe_extract_url

PROXY_TOOL_NAMES = {"web_search", "web_extract"}

WEB_SEARCH_SPEC = ToolSpec(
    alias="web_search",
    name="web_search",
    namespace=None,
    call_type="function",
    raw_type="function",
    description=(
        "Search the live web via local Firecrawl. Returns titles, URLs, and descriptions. "
        "Operators such as site:domain, filetype:pdf, intitle:word, -term, and quoted phrases "
        "may work. Use web_extract to read a specific URL."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
        },
        "required": ["query"],
    },
    core=True,
    proxy=True,
)

WEB_EXTRACT_SPEC = ToolSpec(
    alias="web_extract",
    name="web_extract",
    namespace=None,
    call_type="function",
    raw_type="function",
    description=(
        "Extract clean markdown from page or PDF URLs via local Firecrawl. "
        "Pass up to 5 URLs. Pages over the character budget are head+tail truncated."
    ),
    parameters={
        "type": "object",
        "properties": {
            "urls": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "char_limit": {"type": "integer", "minimum": 2000},
            "format": {"type": "string", "enum": ["markdown", "html"]},
        },
        "required": ["urls"],
    },
    core=True,
    proxy=True,
)


def proxy_web_specs() -> list[ToolSpec]:
    return [WEB_SEARCH_SPEC, WEB_EXTRACT_SPEC]


def inject_proxy_web_tools(catalog: Catalog, cfg: Settings = settings) -> Catalog:
    if not cfg.web_enabled:
        return catalog
    for spec in proxy_web_specs():
        catalog.specs[spec.alias] = spec
        catalog.specs[spec.name] = spec
        if spec.alias not in catalog.selected:
            catalog.selected.insert(0, spec.alias)
        if spec.alias in catalog.deferred:
            catalog.deferred.remove(spec.alias)
    return catalog


def is_proxy_tool(name: str) -> bool:
    return (name or "").strip().lower() in PROXY_TOOL_NAMES


def _args(call: BridgeToolCall) -> dict[str, Any]:
    raw = call.arguments or call.input or "{}"
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {"query": raw} if call.name == "web_search" else {}


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(500, limit // 2)
    omitted = len(text) - 2 * keep
    return text[:keep] + f"\n\n[truncated {omitted} chars; raise char_limit or extract a smaller page]\n\n" + text[-keep:]


async def run_web_search(call: BridgeToolCall, cfg: Settings = settings) -> str:
    args = _args(call)
    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"success": False, "error": "query is required"}, ensure_ascii=False)
    try:
        limit = int(args.get("limit") or cfg.web_search_limit_default)
    except Exception:
        limit = cfg.web_search_limit_default
    limit = min(max(limit, 1), 100)
    try:
        result = await firecrawl_search(query, limit, cfg)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Firecrawl search failed: {exc}"}, ensure_ascii=False)


async def run_web_extract(call: BridgeToolCall, cfg: Settings = settings) -> str:
    args = _args(call)
    urls = args.get("urls")
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list):
        return json.dumps({"success": False, "error": "urls must be a list"}, ensure_ascii=False)
    urls = [str(url).strip() for url in urls if str(url).strip()][:5]
    if not urls:
        return json.dumps({"success": False, "error": "urls is required"}, ensure_ascii=False)
    try:
        char_limit = int(args.get("char_limit") or cfg.web_extract_char_limit)
    except Exception:
        char_limit = cfg.web_extract_char_limit
    char_limit = max(2000, char_limit)
    fmt = str(args.get("format") or "markdown").strip().lower()
    formats = ["html"] if fmt == "html" else ["markdown"]
    pages: list[dict[str, Any]] = []
    for url in urls:
        if not is_safe_extract_url(url):
            pages.append({"url": url, "title": "", "content": "", "error": "Blocked: URL is not a public http(s) address"})
            continue
        try:
            payload = await firecrawl_scrape(url, formats, cfg)
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            title = str(metadata.get("title") or "")
            final_url = str(metadata.get("sourceURL") or metadata.get("url") or url)
            if not is_safe_extract_url(final_url):
                pages.append({"url": final_url, "title": title, "content": "", "error": "Blocked: redirected URL is not public"})
                continue
            content = str(payload.get("markdown") or payload.get("html") or payload.get("content") or "")
            pages.append(
                {
                    "url": final_url,
                    "title": title,
                    "content": _clip(content, char_limit),
                    "metadata": {"statusCode": metadata.get("statusCode"), "contentType": metadata.get("contentType")},
                }
            )
        except Exception as exc:
            pages.append({"url": url, "title": "", "content": "", "error": str(exc)})
    return json.dumps({"success": True, "data": pages}, ensure_ascii=False)


async def run_proxy_tool(call: BridgeToolCall, cfg: Settings = settings) -> str:
    name = (call.name or "").strip().lower()
    if name == "web_search":
        return await run_web_search(call, cfg)
    if name == "web_extract":
        return await run_web_extract(call, cfg)
    return json.dumps({"success": False, "error": f"unknown proxy tool {call.name}"}, ensure_ascii=False)
