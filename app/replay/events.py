from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Any, AsyncIterator

from fastapi import HTTPException

from ..config import Settings, settings
from ..protocol.ids import message_item_id, tool_item_id
from ..protocol.models import BridgeToolCall, Turn, TurnResult, local_response_id
from ..protocol.sse import encode_sse, split_text
from ..upstream.errors import error_message_from_detail


def _seq_event(event_type: str, sequence: int, **payload: Any) -> bytes:
    body = {"type": event_type, "sequence_number": sequence, **payload}
    return encode_sse(body, event_type)


def _message_item(item_id: str, text: str, status: str, *, phase: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "logprobs": [], "annotations": []}] if status == "completed" else [],
    }
    if phase:
        item["phase"] = phase
    return item


def _function_item(call: BridgeToolCall, item_id: str, status: str) -> dict[str, Any]:
    item: dict[str, Any] = {
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


def _custom_item(call: BridgeToolCall, item_id: str, status: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "type": "custom_tool_call",
        "status": status,
        "call_id": call.id,
        "name": call.name,
        "input": call.input or call.arguments if status == "completed" else "",
    }
    if call.namespace:
        item["namespace"] = call.namespace
    return item


def _search_item(call: BridgeToolCall, item_id: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "tool_search_call",
        "status": status,
        "call_id": call.id,
        "name": "tool_search",
        "execution": call.execution or "client",
        "arguments": call.search_arguments or {"query": call.name, "limit": 12},
    }


def _tool_item(call: BridgeToolCall, item_id: str, status: str) -> dict[str, Any]:
    if call.call_type == "custom":
        return _custom_item(call, item_id, status)
    if call.call_type == "tool_search":
        return _search_item(call, item_id, status)
    return _function_item(call, item_id, status)


def make_response_object(rid: str, turn: Turn, status: str, *, output: list[dict[str, Any]] | None = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "error": error,
        "incomplete_details": None,
        "model": turn.model,
        "output": output or [],
        "parallel_tool_calls": True,
        "store": False,
        "usage": {"input_tokens": turn.estimated_input_tokens, "output_tokens": 0, "total_tokens": turn.estimated_input_tokens},
        "metadata": turn.metadata or {},
    }


def compact_output(result: TurnResult) -> dict[str, Any]:
    item_id = message_item_id(result.local_request_id, 0)
    return {
        "output": [
            {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.answer or ""}],
            }
        ]
    }


def response_json(result: TurnResult, turn: Turn) -> dict[str, Any]:
    rid = result.local_request_id
    output: list[dict[str, Any]] = []
    text = result.commentary or result.answer
    if text and result.tool_calls:
        output.append(_message_item(message_item_id(rid, 0), result.commentary or result.answer, "completed", phase="commentary"))
    for call in result.tool_calls:
        output.append(_tool_item(call, tool_item_id(call), "completed"))
    if not result.tool_calls:
        output.append(_message_item(message_item_id(rid, 0), result.answer, "completed"))
    return make_response_object(rid, turn, "completed", output=output)


def chat_json(result: TurnResult, turn: Turn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.answer or result.commentary or None}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments or call.input},
            }
            for call in result.tool_calls
        ]
    return {
        "id": result.local_request_id.replace("resp_local_", "chatcmpl_"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": turn.model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if result.tool_calls else "stop"}],
    }


class ReplayStream:
    def __init__(self, rid: str, turn: Turn, cfg: Settings = settings) -> None:
        self.rid = rid
        self.turn = turn
        self.cfg = cfg
        self.seq = 0

    def event(self, event_type: str, **payload: Any) -> bytes:
        chunk = _seq_event(event_type, self.seq, **payload)
        self.seq += 1
        return chunk

    def created(self) -> list[bytes]:
        created = make_response_object(self.rid, self.turn, "in_progress")
        return [
            self.event("response.created", response=created),
            self.event("response.in_progress", response=created),
        ]

    def keepalive(self) -> bytes:
        return self.event("response.in_progress", response=make_response_object(self.rid, self.turn, "in_progress"))

    def message(self, text: str, output_index: int, *, phase: str | None = None) -> list[bytes]:
        item_id = message_item_id(self.rid, output_index)
        events = [self.event("response.output_item.added", output_index=output_index, item=_message_item(item_id, "", "in_progress", phase=phase))]
        for chunk in split_text(text, self.cfg.responses_chunk_size):
            events.append(self.event("response.output_text.delta", item_id=item_id, output_index=output_index, content_index=0, delta=chunk, logprobs=[]))
        events.append(
            self.event(
                "response.output_item.done",
                output_index=output_index,
                item_id=item_id,
                item=_message_item(item_id, text, "completed", phase=phase),
            )
        )
        return events

    def tool(self, call: BridgeToolCall, output_index: int) -> list[bytes]:
        item_id = tool_item_id(call)
        events = [self.event("response.output_item.added", output_index=output_index, item=_tool_item(call, item_id, "in_progress"))]
        if call.call_type == "custom":
            payload = call.input or call.arguments
            for chunk in split_text(payload, self.cfg.responses_chunk_size):
                events.append(
                    self.event(
                        "response.custom_tool_call_input.delta",
                        item_id=item_id,
                        output_index=output_index,
                        call_id=call.id,
                        delta=chunk,
                    )
                )
        events.append(self.event("response.output_item.done", output_index=output_index, item_id=item_id, item=_tool_item(call, item_id, "completed")))
        return events

    def completed(self, result: TurnResult) -> bytes:
        return self.event("response.completed", response=response_json(result, self.turn))

    def failed(self, error: HTTPException) -> list[bytes]:
        err = {"message": error_message_from_detail(error.detail), "type": "local_proxy_error"}
        failed = make_response_object(self.rid, self.turn, "failed", error=err)
        return [self.event("response.failed", response=failed), self.event("error", error=err)]


async def responses_sse_generator(result_coro, turn: Turn, rid: str | None = None) -> AsyncIterator[bytes]:
    rid = rid or local_response_id()
    stream = ReplayStream(rid, turn)
    task: asyncio.Task | None = None
    try:
        for chunk in stream.created():
            yield chunk
        task = asyncio.ensure_future(result_coro)
        heartbeat = max(0.05, settings.client_heartbeat_interval_ms / 1000)
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=heartbeat)
            if done:
                break
            yield stream.keepalive()
        result: TurnResult = task.result()
        result.local_request_id = rid
        index = 0
        text = result.commentary if result.tool_calls else result.answer
        phase = "commentary" if result.tool_calls and result.commentary else None
        if text:
            for chunk in stream.message(text, index, phase=phase):
                yield chunk
            index += 1
        for call in result.tool_calls:
            for chunk in stream.tool(call, index):
                yield chunk
            index += 1
        if not result.tool_calls and not text:
            for chunk in stream.message(result.answer or "", 0):
                yield chunk
        yield stream.completed(result)
    except HTTPException as e:
        for chunk in stream.failed(e):
            yield chunk
    except asyncio.CancelledError:
        if task:
            task.cancel()
            with suppress(Exception):
                await task
        raise
    finally:
        if task and not task.done():
            task.cancel()
            with suppress(Exception):
                await task


async def chat_sse_generator(result_coro, turn: Turn) -> AsyncIterator[bytes]:
    result: TurnResult = await result_coro
    payload = chat_json(result, turn)
    yield encode_sse(payload)
    yield encode_sse("[DONE]")
