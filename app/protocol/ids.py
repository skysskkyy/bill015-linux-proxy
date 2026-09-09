from __future__ import annotations

import uuid
from typing import Any

from .models import BridgeToolCall

ITEM_ID_PREFIX = {
    "message": "msg",
    "function_call": "fc",
    "function_call_output": "fco",
    "custom_tool_call": "ctc",
    "custom_tool_call_output": "ctco",
    "tool_search_call": "tsc",
    "tool_search_output": "tso",
    "reasoning": "rs",
    "web_search_call": "ws",
    "compaction": "cmp",
}

_KNOWN_PREFIXES = tuple(sorted({f"{p}_" for p in ITEM_ID_PREFIX.values()} | {"resp_local_", "call_"}, key=len, reverse=True))


def _token(seed: str = "") -> str:
    raw = (seed or uuid.uuid4().hex).replace("-", "")
    for prefix in _KNOWN_PREFIXES:
        while raw.startswith(prefix):
            raw = raw[len(prefix) :]
    return (raw or uuid.uuid4().hex)[:24]


def make_item_id(kind: str, seed: str = "") -> str:
    prefix = ITEM_ID_PREFIX.get(kind, "msg")
    return f"{prefix}_{_token(seed)}"


def message_item_id(seed: str = "", index: int = 0) -> str:
    return make_item_id("message", f"{seed}{index}")


def tool_item_id(call: BridgeToolCall) -> str:
    kind = {"custom": "custom_tool_call", "tool_search": "tool_search_call"}.get(call.call_type, "function_call")
    return make_item_id(kind, call.id)


def ensure_call_id(call_id: str) -> str:
    token = _token(call_id)
    return f"call_{token}"


def coerce_item_id(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return item
    typ = str(item.get("type") or ("message" if item.get("role") else ""))
    prefix = ITEM_ID_PREFIX.get(typ)
    if not prefix:
        return item
    current = str(item.get("id") or "")
    if current.startswith(prefix):
        return item
    out = dict(item)
    out["id"] = make_item_id(typ, current)
    return out
