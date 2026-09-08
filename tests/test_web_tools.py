from __future__ import annotations

import asyncio
import json

from app.ingest.catalog import build_catalog
from app.protocol.models import BridgeToolCall
from app.tools.firecrawl import extract_search_results, is_safe_extract_url
from app.tools.loop import split_proxy_calls
from app.tools.web import is_proxy_tool, run_web_extract, run_web_search


def test_catalog_injects_web_search_and_extract():
    catalog = build_catalog([], [], query="latest news")
    assert "web_search" in catalog.specs
    assert "web_extract" in catalog.specs
    assert catalog.specs["web_search"].proxy is True
    assert catalog.selected[0] in {"web_search", "web_extract"}


def test_split_proxy_calls_keeps_codex_tools():
    proxy, client = split_proxy_calls(
        [
            BridgeToolCall(id="1", name="web_search", arguments='{"query":"x"}'),
            BridgeToolCall(id="2", name="shell_command", arguments='{"command":"ls"}'),
            BridgeToolCall(id="3", name="web_extract", arguments='{"urls":["https://example.com"]}'),
        ]
    )
    assert [c.name for c in proxy] == ["web_search", "web_extract"]
    assert [c.name for c in client] == ["shell_command"]
    assert is_proxy_tool("web_search")
    assert not is_proxy_tool("shell_command")


def test_ssrf_blocks_private_extract_urls():
    assert is_safe_extract_url("https://example.com/page")
    assert not is_safe_extract_url("http://127.0.0.1:3002")
    assert not is_safe_extract_url("http://192.168.1.1/")
    assert not is_safe_extract_url("file:///etc/passwd")


def test_search_result_normalizer():
    rows = extract_search_results({"success": True, "data": {"web": [{"title": "A", "url": "https://a.test", "description": "d"}]}})
    assert rows[0]["url"] == "https://a.test"


def test_web_search_against_local_firecrawl():
    call = BridgeToolCall(id="c1", name="web_search", arguments=json.dumps({"query": "example domain", "limit": 2}))
    raw = asyncio.run(run_web_search(call))
    body = json.loads(raw)
    assert body.get("success") is True
    assert body["data"]["web"]


def test_web_extract_against_local_firecrawl():
    call = BridgeToolCall(id="c2", name="web_extract", arguments=json.dumps({"urls": ["https://example.com"]}))
    raw = asyncio.run(run_web_extract(call))
    body = json.loads(raw)
    assert body.get("success") is True
    assert "Example Domain" in (body["data"][0].get("content") or "")
