from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Any, AsyncIterator

from fastapi import HTTPException

from ..config import Settings, settings
from ..protocol.ids import coerce_item_id, make_item_id, message_item_id, tool_item_id
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


def _summary_texts(item: dict[str, Any]) -> list[str]:
    summary = item.get("summary") if isinstance(item.get("summary"), list) else []
    texts: list[str] = []
    for part in summary:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        elif isinstance(part, str) and part:
            texts.append(part)
    return texts


def _reasoning_item(item: dict[str, Any], status: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": item.get("id") or make_item_id("reasoning"),
        "type": "reasoning",
        "status": status,
        "summary": item.get("summary") if isinstance(item.get("summary"), list) else [],
        "content": item.get("content") if isinstance(item.get("content"), list) else [],
    }
    encrypted = item.get("encrypted_content")
    if isinstance(encrypted, str) and encrypted:
        payload["encrypted_content"] = encrypted
    return coerce_item_id(payload)


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


def visible_assistant_text(result: TurnResult) -> str:
    if result.tool_calls:
        return (result.commentary or "").strip()
    return ((result.answer or "").strip() or (result.commentary or "").strip())


def compact_output(result: TurnResult, turn: Turn | None = None) -> dict[str, Any]:
    if result.compacted_output:
        return {
            "id": result.local_request_id,
            "object": "response.compaction",
            "model": turn.model if turn is not None else "",
            "output": result.compacted_output,
        }
    item_id = message_item_id(result.local_request_id, 0)
    return {
        "id": result.local_request_id,
        "object": "response.compaction",
        "model": turn.model if turn is not None else "",
        "output": [
            {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": visible_assistant_text(result)}],
            }
        ],
    }


def response_json(result: TurnResult, turn: Turn) -> dict[str, Any]:
    rid = result.local_request_id
    output: list[dict[str, Any]] = [_reasoning_item(item, "completed") for item in result.reasoning_items]
    text = visible_assistant_text(result)
    if text and result.tool_calls:
        output.append(_message_item(message_item_id(rid, 0), text, "completed", phase="commentary"))
    for call in result.tool_calls:
        output.append(_tool_item(call, tool_item_id(call), "completed"))
    if not result.tool_calls:
        output.append(_message_item(message_item_id(rid, 0), text, "completed"))
    return make_response_object(rid, turn, "completed", output=output)


def chat_json(result: TurnResult, turn: Turn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": visible_assistant_text(result) or None}
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
        self._opened_summaries: set[tuple[str, int]] = set()

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

    def reasoning(self, item: dict[str, Any], output_index: int) -> list[bytes]:
        added = _reasoning_item({**item, "summary": []}, "in_progress")
        done = _reasoning_item(item, "completed")
        item_id = str(done.get("id") or added.get("id"))
        events = [self.event("response.output_item.added", output_index=output_index, item=added)]
        for index, text in enumerate(_summary_texts(item)):
            events.extend(self.reasoning_summary(item_id, output_index, index, text))
        events.append(self.event("response.output_item.done", output_index=output_index, item_id=item_id, item=done))
        return events

    def reasoning_summary(self, item_id: str, output_index: int, summary_index: int, text: str) -> list[bytes]:
        events = [
            self.event(
                "response.reasoning_summary_part.added",
                item_id=item_id,
                output_index=output_index,
                summary_index=summary_index,
                part={"type": "summary_text", "text": ""},
            )
        ]
        for chunk in split_text(text, self.cfg.responses_chunk_size):
            events.append(
                self.event(
                    "response.reasoning_summary_text.delta",
                    item_id=item_id,
                    output_index=output_index,
                    summary_index=summary_index,
                    delta=chunk,
                )
            )
        events.append(
            self.event(
                "response.reasoning_summary_text.done",
                item_id=item_id,
                output_index=output_index,
                summary_index=summary_index,
                text=text,
            )
        )
        events.append(
            self.event(
                "response.reasoning_summary_part.done",
                item_id=item_id,
                output_index=output_index,
                summary_index=summary_index,
                part={"type": "summary_text", "text": text},
            )
        )
        return events

    def live(self, event: dict[str, Any]) -> list[bytes]:
        kind = str(event.get("kind") or "")
        if kind == "reasoning_added":
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            payload = _reasoning_item({**item, "summary": []}, "in_progress")
            return [self.event("response.output_item.added", output_index=0, item=payload)]
        if kind == "reasoning_summary_delta":
            item_id = str(event.get("item_id") or "")
            summary_index = int(event.get("summary_index") or 0)
            events: list[bytes] = []
            marker = (item_id, summary_index)
            if marker not in self._opened_summaries:
                self._opened_summaries.add(marker)
                events.append(
                    self.event(
                        "response.reasoning_summary_part.added",
                        item_id=item_id,
                        output_index=0,
                        summary_index=summary_index,
                        part={"type": "summary_text", "text": ""},
                    )
                )
            events.append(
                self.event(
                    "response.reasoning_summary_text.delta",
                    item_id=item_id,
                    output_index=0,
                    summary_index=summary_index,
                    delta=str(event.get("delta") or ""),
                )
            )
            return events
        if kind == "reasoning_summary_done":
            text = str(event.get("text") or "")
            return [
                self.event(
                    "response.reasoning_summary_text.done",
                    item_id=str(event.get("item_id") or ""),
                    output_index=0,
                    summary_index=int(event.get("summary_index") or 0),
                    text=text,
                )
            ]
        if kind == "reasoning_done":
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            payload = _reasoning_item(item, "completed")
            return [self.event("response.output_item.done", output_index=0, item_id=payload.get("id"), item=payload)]
        return []

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
        turn.event_sink = asyncio.Queue()
        task = asyncio.ensure_future(result_coro)
        heartbeat = max(0.05, settings.client_heartbeat_interval_ms / 1000)
        while not task.done():
            try:
                live_event = await asyncio.wait_for(turn.event_sink.get(), timeout=heartbeat)
            except TimeoutError:
                yield stream.keepalive()
                continue
            for chunk in stream.live(live_event):
                yield chunk
        while turn.event_sink and not turn.event_sink.empty():
            live_event = turn.event_sink.get_nowait()
            for chunk in stream.live(live_event):
                yield chunk
        result: TurnResult = task.result()
        result.local_request_id = rid
        index = 0
        if not result.reasoning_live:
            for item in result.reasoning_items:
                for chunk in stream.reasoning(item, index):
                    yield chunk
                index += 1
        elif result.reasoning_items:
            index = len(result.reasoning_items)
        text = visible_assistant_text(result)
        phase = "commentary" if result.tool_calls and text else None
        if text:
            for chunk in stream.message(text, index, phase=phase):
                yield chunk
            index += 1
        for call in result.tool_calls:
            for chunk in stream.tool(call, index):
                yield chunk
            index += 1
        if not result.tool_calls and not text:
            for chunk in stream.message("", index):
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
