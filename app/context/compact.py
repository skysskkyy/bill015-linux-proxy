from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings, settings
from ..ingest.text import rewrite_encrypted_content_parts
from ..protocol.ids import make_item_id
from ..protocol.models import Turn
from .pack import _latest_tool_batch, _latest_user_index


def parse_compact_output(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    output = body.get("output")
    if isinstance(output, list):
        return [item for item in output if isinstance(item, dict)]
    response = body.get("response")
    if isinstance(response, dict) and isinstance(response.get("output"), list):
        return [item for item in response["output"] if isinstance(item, dict)]
    return []


def _has_compaction_item(items: list[dict[str, Any]]) -> bool:
    return any(item.get("type") == "compaction" for item in items)


def _pinned_tail(source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tail: list[dict[str, Any]] = []
    user_index = _latest_user_index(source)
    batch = _latest_tool_batch(source)
    seen: set[int] = set()
    if user_index is not None:
        tail.append(dict(source[user_index]))
        seen.add(user_index)
    if batch is not None:
        for index in range(batch[0], batch[1]):
            if index in seen:
                continue
            tail.append(dict(source[index]))
            seen.add(index)
    return tail


def merge_compact_output(source: list[dict[str, Any]], compacted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [dict(item) for item in compacted]
    if not _has_compaction_item(merged):
        summary_bits = []
        for item in compacted:
            text = ""
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text += part["text"]
            elif isinstance(item.get("encrypted_content"), str):
                text = ""
            if text.strip():
                summary_bits.append(text.strip())
        blob = "\n".join(summary_bits).strip() or "Earlier turns were compacted."
        merged.append(
            {
                "id": make_item_id("compaction"),
                "type": "compaction",
                "summary": [{"type": "summary_text", "text": blob}],
            }
        )
    tail = _pinned_tail(source)
    for item in tail:
        marker = (item.get("type"), item.get("id"), item.get("call_id"), str(item.get("content"))[:80])
        if any((other.get("type"), other.get("id"), other.get("call_id"), str(other.get("content"))[:80]) == marker for other in merged):
            continue
        merged.append(item)
    return merged


async def compact_history(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    turn: Turn,
    items: list[dict[str, Any]],
    cfg: Settings = settings,
) -> list[dict[str, Any]] | None:
    url = cfg.upstream_base_url.rstrip("/") + "/v1/responses/compact"
    payload = {
        "model": turn.model,
        "input": rewrite_encrypted_content_parts(items),
        "store": False,
        "stream": False,
    }
    compact_headers = dict(headers)
    compact_headers["accept"] = "application/json"
    try:
        resp = await client.post(url, headers=compact_headers, json=payload)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except Exception:
        return None
    compacted = parse_compact_output(body)
    if not compacted:
        return None
    return merge_compact_output(items, compacted)
