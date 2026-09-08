from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import Settings, settings

_PRIVATE_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def is_safe_extract_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host or host in _PRIVATE_HOSTS:
        return False
    if host.endswith(".local") or host.endswith(".internal"):
        return False
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        a, b = int(parts[0]), int(parts[1])
        if a == 10 or a == 127 or (a == 192 and b == 168) or (a == 172 and 16 <= b <= 31):
            return False
    return True


def _headers(cfg: Settings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.web_api_key:
        headers["Authorization"] = f"Bearer {cfg.web_api_key}"
    return headers


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list, str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    return value


def extract_search_results(response: Any) -> list[dict[str, Any]]:
    response_plain = _plain(response)
    if not isinstance(response_plain, dict):
        return []
    data = response_plain.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("web", "results"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]
    for key in ("web", "results"):
        rows = response_plain.get(key)
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    return []


def extract_scrape_payload(response: Any) -> dict[str, Any]:
    plain = _plain(response)
    if not isinstance(plain, dict):
        return {}
    nested = plain.get("data")
    return nested if isinstance(nested, dict) else plain


async def firecrawl_search(query: str, limit: int, cfg: Settings = settings) -> dict[str, Any]:
    url = cfg.web_api_url.rstrip("/") + "/v2/search"
    payload = {"query": query, "limit": limit}
    async with httpx.AsyncClient(timeout=cfg.web_timeout_seconds) as client:
        resp = await client.post(url, json=payload, headers=_headers(cfg))
        resp.raise_for_status()
        body = resp.json()
    results = extract_search_results(body)
    web = []
    for index, item in enumerate(results, start=1):
        web.append(
            {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or item.get("link") or ""),
                "description": str(item.get("description") or item.get("snippet") or item.get("markdown") or ""),
                "position": index,
            }
        )
    return {"success": True, "data": {"web": web}}


async def firecrawl_scrape(url: str, formats: list[str], cfg: Settings = settings) -> dict[str, Any]:
    endpoint = cfg.web_api_url.rstrip("/") + "/v2/scrape"
    payload = {"url": url, "formats": formats}
    async with httpx.AsyncClient(timeout=cfg.web_timeout_seconds) as client:
        resp = await client.post(endpoint, json=payload, headers=_headers(cfg))
        resp.raise_for_status()
        return extract_scrape_payload(resp.json())
