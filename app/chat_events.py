from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from fastapi import HTTPException

from .models import Bill015Result, BridgeToolCall, NormalizedRequest
from .sse import encode_sse, split_text
from .usage_estimator import build_chat_usage


async def chat_sse_generator(result_coro, n: NormalizedRequest) -> AsyncIterator[bytes]:
    try:
        result: Bill015Result = await result_coro
        finish_reason = "tool_calls" if result.bridge_mode == "tool_call" and result.tool_calls else "stop"
        initial_delta: dict[str, Any] = {"role": "assistant"}
        if finish_reason == "tool_calls":
            initial_delta["tool_calls"] = [_chat_tool_call_delta(call, index) for index, call in enumerate(result.tool_calls)]
        yield encode_sse({"id": result.local_request_id, "object": "chat.completion.chunk", "model": n.model, "choices": [{"index": 0, "delta": initial_delta}]})
        if finish_reason != "tool_calls":
            for chunk in split_text(result.answer, 512):
                yield encode_sse({"id": result.local_request_id, "object": "chat.completion.chunk", "model": n.model, "choices": [{"index": 0, "delta": {"content": chunk}}]})
        yield encode_sse({"id": result.local_request_id, "object": "chat.completion.chunk", "model": n.model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": build_chat_usage(n, answer=result.answer)})
        yield b"data: [DONE]\n\n"
    except HTTPException as e:
        yield encode_sse({"error": {"message": str(e.detail), "code": e.status_code, "type": "local_proxy_error"}})
        yield b"data: [DONE]\n\n"
    except Exception as e:
        yield encode_sse({"error": {"message": f"{type(e).__name__}: {e}", "code": 502, "type": "local_proxy_error"}})
        yield b"data: [DONE]\n\n"


def chat_json(result: Bill015Result, n: NormalizedRequest) -> dict[str, Any]:
    finish_reason = "tool_calls" if result.bridge_mode == "tool_call" and result.tool_calls else "stop"
    message: dict[str, Any] = {"role": "assistant", "content": result.answer}
    if finish_reason == "tool_calls":
        message["content"] = None
        message["tool_calls"] = [_chat_tool_call(call, index) for index, call in enumerate(result.tool_calls)]
    return {
        "id": "chatcmpl-local-" + result.local_request_id.removeprefix("resp_local_"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": n.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": build_chat_usage(n, answer=result.answer),
    }


def _chat_tool_call(call: BridgeToolCall, index: int) -> dict[str, Any]:
    # Chat Completions only has the function tool-call shape. Preserve custom
    # and namespace metadata inside arguments so downstream clients can still
    # route the native Codex call.
    arguments = call.arguments
    name = call.name
    if call.call_type != "function" or call.namespace:
        arguments = _chat_tool_call_arguments(call)
        name = call.name
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _chat_tool_call_delta(call: BridgeToolCall, index: int) -> dict[str, Any]:
    item = _chat_tool_call(call, index)
    return {"index": index, **item}


def _chat_tool_call_arguments(call: BridgeToolCall) -> str:
    return json.dumps(
        {
            "type": call.call_type,
            "namespace": call.namespace or "",
            "name": call.name,
            "arguments": call.arguments,
            "input": call.arguments if call.call_type == "custom" else "",
            "requested_name": call.requested_name or call.name,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
