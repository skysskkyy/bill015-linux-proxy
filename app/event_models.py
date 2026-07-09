from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .models import BridgeToolCall, NormalizedRequest
from .usage_estimator import build_response_usage


@dataclass
class EventContext:
    response_id: str
    model: str
    created_at: int = field(default_factory=lambda: int(time.time()))
    sequence_number: int = 0
    output_index: int = 0

    def next_sequence(self) -> int:
        current = self.sequence_number
        self.sequence_number += 1
        return current


@dataclass
class OutputItemRef:
    id: str
    output_index: int
    item_type: str
    call_id: str | None = None
    name: str | None = None


@dataclass
class MessagePart:
    type: str = "output_text"
    text: str = ""
    annotations: list[dict[str, Any]] = field(default_factory=list)
    refusal: str = ""


def response_event(event_type: str, sequence_number: int, **payload: Any) -> dict[str, Any]:
    event = {"type": event_type, **payload}
    event["sequence_number"] = sequence_number
    return event


def make_response_object(
    rid: str,
    n: NormalizedRequest,
    status: str,
    *,
    output: list[dict[str, Any]] | None = None,
    answer: str = "",
    calls: list[BridgeToolCall] | None = None,
    created_at: int | None = None,
    error: dict[str, Any] | None = None,
    incomplete_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_config = n.text_config or {"format": {"type": "text"}}
    if "format" not in text_config:
        text_config = {"format": {"type": "text"}, **text_config}
    return {
        "id": rid,
        "object": "response",
        "created_at": created_at or int(time.time()),
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "model": n.model,
        "output": output or [],
        "parallel_tool_calls": bool(n.parallel_tool_calls),
        "tool_choice": n.tool_choice,
        "tools": [],
        "usage": build_response_usage(n, answer=answer, calls=calls),
        "reasoning": n.reasoning,
        "text": text_config,
        "metadata": n.metadata or {},
    }


def message_item_id(rid: str, output_index: int = 0) -> str:
    return "msg_" + rid.removeprefix("resp_local_")[:18] + f"_{output_index}"


def reasoning_item_id(rid: str, output_index: int = 0) -> str:
    return "rs_" + rid.removeprefix("resp_local_")[:18] + f"_{output_index}"


def tool_call_item_id(call: BridgeToolCall) -> str:
    if call.call_type == "custom":
        prefix = "ctc_"
    elif call.call_type == "tool_search":
        prefix = "tsc_"
    elif call.call_type == "web_search":
        prefix = "wsc_"
    else:
        prefix = "item_"
    return prefix + call.id.removeprefix("call_")[:24]


def make_message_part(part: MessagePart | dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "output_text", "text": part, "annotations": []}
    if isinstance(part, dict):
        if part.get("type") == "refusal":
            return {"type": "refusal", "refusal": str(part.get("refusal") or part.get("text") or "")}
        out = {
            "type": str(part.get("type") or "output_text"),
            "text": str(part.get("text") or ""),
        }
        if "annotations" in part:
            out["annotations"] = part.get("annotations") if isinstance(part.get("annotations"), list) else []
        return out
    if part.type == "refusal":
        return {"type": "refusal", "refusal": part.refusal or part.text}
    return {"type": part.type, "text": part.text, "annotations": part.annotations}


def make_message_item(item_id: str, status: str, parts: list[MessagePart | dict[str, Any] | str] | None = None) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [make_message_part(part) for part in (parts or [])],
    }


def _arguments_object(raw: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    default = default or {}
    if not raw:
        return dict(default)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"input": obj}
    except Exception:
        return {"query": raw} if isinstance(raw, str) else dict(default)


def make_function_call_item(call: BridgeToolCall, item_id: str, status: str = "completed") -> dict[str, Any]:
    item = {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments if status == "completed" else "",
    }
    if call.namespace:
        item["namespace"] = call.namespace
    return item


def make_custom_tool_call_item(call: BridgeToolCall, item_id: str, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "custom_tool_call",
        "status": status,
        "call_id": call.id,
        "name": call.name,
        "input": call.arguments if status == "completed" else "",
    }


def make_tool_search_call_item(call: BridgeToolCall, item_id: str, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "tool_search_call",
        "status": status,
        "call_id": call.id,
        "execution": "client",
        "arguments": _arguments_object(call.arguments, {"query": "", "limit": 8}) if status == "completed" else {},
    }


def make_web_search_call_item(call: BridgeToolCall, item_id: str, status: str = "completed") -> dict[str, Any]:
    action = _arguments_object(call.arguments, {"type": "search", "query": ""}) if status == "completed" else {}
    if "type" not in action and "query" in action:
        action = {"type": "search", **action}
    return {
        "id": item_id,
        "type": "web_search_call",
        "status": status,
        "action": action,
    }


def make_tool_call_item(call: BridgeToolCall, item_id: str, status: str = "completed", *, allow_web_search: bool = False) -> dict[str, Any]:
    if call.call_type == "custom":
        return make_custom_tool_call_item(call, item_id, status)
    if call.call_type == "tool_search" or (call.call_type == "web_search" and not allow_web_search):
        return make_tool_search_call_item(call, item_id, status)
    if call.call_type == "web_search":
        return make_web_search_call_item(call, item_id, status)
    return make_function_call_item(call, item_id, status)


def make_reasoning_item(item_id: str, status: str = "completed", summary: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": summary or [],
    }
