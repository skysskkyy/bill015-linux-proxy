from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from fastapi import HTTPException

from .models import Bill015Result, BridgeToolCall, NormalizedRequest, local_response_id
from .token_usage import chat_usage, response_usage
from .sse import encode_sse, split_text


def response_json(result: Bill015Result, n: NormalizedRequest) -> dict[str, Any]:
    if result.bridge_mode == "tool_call" and result.tool_calls:
        return response_object_with_tool_calls(result.local_request_id, n, result.tool_calls, "completed")
    return response_object(result.local_request_id, n, "completed", result.answer)


def response_object(rid: str, n: NormalizedRequest, status: str, answer: str = "") -> dict[str, Any]:
    item_id = "msg_" + rid.removeprefix("resp_local_")[:24]
    output = []
    if answer or status == "completed":
        output = [{
            "id": item_id,
            "type": "message",
            "status": "completed" if status == "completed" else "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": answer, "annotations": []}],
        }]
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": n.model,
        "output": output,
        "parallel_tool_calls": bool(n.parallel_tool_calls),
        "tool_choice": n.tool_choice,
        "tools": [],
        "usage": response_usage(n, answer=answer),
    }


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


def _arguments_object(raw: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    default = default or {}
    if not raw:
        return dict(default)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"input": obj}
    except Exception:
        return {"query": raw} if isinstance(raw, str) else dict(default)


def tool_call_item(call: BridgeToolCall, item_id: str, status: str = "completed") -> dict[str, Any]:
    if call.call_type == "custom":
        return {
            "id": item_id,
            "type": "custom_tool_call",
            "status": status,
            "call_id": call.id,
            "name": call.name,
            "input": call.arguments if status == "completed" else "",
        }
    if call.call_type == "tool_search":
        return {
            "id": item_id,
            "type": "tool_search_call",
            "status": status,
            "call_id": call.id,
            "execution": "client",
            "arguments": _arguments_object(call.arguments, {"query": "", "limit": 8}) if status == "completed" else {},
        }
    if call.call_type == "web_search":
        action = _arguments_object(call.arguments, {"type": "search", "query": ""}) if status == "completed" else {}
        if "type" not in action and "query" in action:
            action = {"type": "search", **action}
        return {
            "id": item_id,
            "type": "web_search_call",
            "status": status,
            "action": action,
        }
    return {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments if status == "completed" else "",
    }


def response_object_with_tool_calls(
    rid: str,
    n: NormalizedRequest,
    calls: list[BridgeToolCall],
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": n.model,
        "output": [tool_call_item(call, tool_call_item_id(call), status) for call in calls],
        "parallel_tool_calls": bool(n.parallel_tool_calls),
        "tool_choice": n.tool_choice,
        "tools": [],
        "usage": response_usage(n, calls=calls),
    }


class ResponsesEventStream:
    def __init__(self, rid: str, n: NormalizedRequest) -> None:
        self.rid = rid
        self.n = n
        self.sequence_number = 0

    def encode(self, payload: dict[str, Any], event_name: str) -> bytes:
        payload.setdefault("sequence_number", self.sequence_number)
        self.sequence_number += 1
        return encode_sse(payload, event_name)

    def created_events(self) -> list[bytes]:
        created = response_object(self.rid, self.n, "in_progress")
        return [
            self.encode({"type": "response.created", "response": created}, "response.created"),
            self.encode({"type": "response.in_progress", "response": created}, "response.in_progress"),
        ]

    def tool_call_events(self, call: BridgeToolCall, output_index: int) -> list[bytes]:
        item_id = tool_call_item_id(call)
        events = [self.encode({"type": "response.output_item.added", "output_index": output_index, "item": tool_call_item(call, item_id, "in_progress")}, "response.output_item.added")]
        if call.call_type == "custom":
            events.extend(self._custom_tool_input_events(call, item_id, output_index))
        elif call.call_type in {"tool_search", "web_search"}:
            # Native Codex records these as built-in Responses output items,
            # not function_call_arguments delta/done streams.
            pass
        else:
            events.extend(self._function_call_argument_events(call, item_id, output_index))
        events.append(self.encode({"type": "response.output_item.done", "output_index": output_index, "item": tool_call_item(call, item_id, "completed")}, "response.output_item.done"))
        return events

    def message_events(self, answer: str) -> list[bytes]:
        item_id = "msg_" + self.rid.removeprefix("resp_local_")[:24]
        events = [
            self.encode({"type": "response.output_item.added", "output_index": 0, "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}}, "response.output_item.added"),
            self.encode({"type": "response.content_part.added", "item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}, "response.content_part.added"),
        ]
        for chunk in split_text(answer, 256):
            events.append(self.encode({"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": chunk, "logprobs": []}, "response.output_text.delta"))
        final_part = {"type": "output_text", "text": answer, "annotations": []}
        events.extend([
            self.encode({"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "content_index": 0, "text": answer, "logprobs": []}, "response.output_text.done"),
            self.encode({"type": "response.content_part.done", "item_id": item_id, "output_index": 0, "content_index": 0, "part": final_part}, "response.content_part.done"),
            self.encode({"type": "response.output_item.done", "output_index": 0, "item": {"id": item_id, "type": "message", "status": "completed", "role": "assistant", "content": [final_part]}}, "response.output_item.done"),
        ])
        return events

    def completed_tool_response(self, calls: list[BridgeToolCall]) -> bytes:
        return self.encode({"type": "response.completed", "response": response_object_with_tool_calls(self.rid, self.n, calls, "completed")}, "response.completed")

    def completed_message_response(self, answer: str) -> bytes:
        return self.encode({"type": "response.completed", "response": response_object(self.rid, self.n, "completed", answer)}, "response.completed")

    def failed_events(self, error: HTTPException) -> list[bytes]:
        failed = response_object(self.rid, self.n, "failed")
        failed["error"] = {"message": str(error.detail), "code": error.status_code}
        return [
            self.encode({"type": "response.failed", "response": failed}, "response.failed"),
            self.encode({"type": "error", "error": {"message": str(error.detail), "code": error.status_code}}, "error"),
        ]

    def _custom_tool_input_events(self, call: BridgeToolCall, item_id: str, output_index: int) -> list[bytes]:
        events = []
        for chunk in split_text(call.arguments, 256):
            events.append(self.encode({"type": "response.custom_tool_call_input.delta", "item_id": item_id, "output_index": output_index, "delta": chunk}, "response.custom_tool_call_input.delta"))
        events.append(self.encode({"type": "response.custom_tool_call_input.done", "item_id": item_id, "output_index": output_index, "input": call.arguments}, "response.custom_tool_call_input.done"))
        return events

    def _function_call_argument_events(self, call: BridgeToolCall, item_id: str, output_index: int) -> list[bytes]:
        events = []
        for chunk in split_text(call.arguments, 256):
            events.append(self.encode({"type": "response.function_call_arguments.delta", "item_id": item_id, "output_index": output_index, "call_id": call.id, "name": call.name, "delta": chunk}, "response.function_call_arguments.delta"))
        events.append(self.encode({"type": "response.function_call_arguments.done", "item_id": item_id, "output_index": output_index, "call_id": call.id, "name": call.name, "arguments": call.arguments}, "response.function_call_arguments.done"))
        return events


async def responses_sse_generator(result_coro, n: NormalizedRequest, rid: str | None = None) -> AsyncIterator[bytes]:
    rid = rid or local_response_id()
    stream = ResponsesEventStream(rid, n)
    try:
        for chunk in stream.created_events():
            yield chunk

        result: Bill015Result = await result_coro
        result.local_request_id = rid
        if result.bridge_mode == "tool_call" and result.tool_calls:
            for output_index, call in enumerate(result.tool_calls):
                for chunk in stream.tool_call_events(call, output_index):
                    yield chunk
            yield stream.completed_tool_response(result.tool_calls)
            yield b"data: [DONE]\n\n"
            return

        for chunk in stream.message_events(result.answer):
            yield chunk
        yield stream.completed_message_response(result.answer)
        yield b"data: [DONE]\n\n"
    except HTTPException as e:
        for chunk in stream.failed_events(e):
            yield chunk
        yield b"data: [DONE]\n\n"


async def chat_sse_generator(result_coro, n: NormalizedRequest) -> AsyncIterator[bytes]:
    try:
        result: Bill015Result = await result_coro
        yield encode_sse({"id": result.local_request_id, "object": "chat.completion.chunk", "model": n.model, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        for chunk in split_text(result.answer, 512):
            yield encode_sse({"id": result.local_request_id, "object": "chat.completion.chunk", "model": n.model, "choices": [{"index": 0, "delta": {"content": chunk}}]})
        yield encode_sse({"id": result.local_request_id, "object": "chat.completion.chunk", "model": n.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        yield b"data: [DONE]\n\n"
    except HTTPException as e:
        yield encode_sse({"error": {"message": str(e.detail), "code": e.status_code}})
        yield b"data: [DONE]\n\n"


def chat_json(result: Bill015Result, n: NormalizedRequest) -> dict[str, Any]:
    return {
        "id": "chatcmpl-local-" + result.local_request_id.removeprefix("resp_local_"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": n.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.answer}, "finish_reason": "stop"}],
        "usage": chat_usage(n, answer=result.answer),
    }
