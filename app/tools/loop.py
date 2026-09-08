from __future__ import annotations

import json

from ..config import Settings, settings
from ..protocol.ids import ensure_call_id, make_item_id
from ..protocol.models import BridgeToolCall, Turn
from .web import is_proxy_tool, run_proxy_tool


def split_proxy_calls(calls: list[BridgeToolCall]) -> tuple[list[BridgeToolCall], list[BridgeToolCall]]:
    proxy: list[BridgeToolCall] = []
    client: list[BridgeToolCall] = []
    for call in calls:
        (proxy if is_proxy_tool(call.name) else client).append(call)
    return proxy, client


def _call_item(call: BridgeToolCall) -> dict:
    return {
        "id": make_item_id("function_call", call.id),
        "type": "function_call",
        "call_id": ensure_call_id(call.id),
        "name": call.name,
        "arguments": call.arguments or json.dumps({"query": call.input}, ensure_ascii=False),
    }


def _output_item(call: BridgeToolCall, output: str) -> dict:
    return {
        "id": make_item_id("function_call_output", call.id),
        "type": "function_call_output",
        "call_id": ensure_call_id(call.id),
        "output": output,
    }


async def apply_proxy_tools(turn: Turn, calls: list[BridgeToolCall], cfg: Settings = settings) -> list[str]:
    notices: list[str] = []
    for call in calls:
        output = await run_proxy_tool(call, cfg)
        turn.items.append(_call_item(call))
        turn.items.append(_output_item(call, output))
        notices.append(call.name)
    return notices
