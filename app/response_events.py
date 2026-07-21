from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, AsyncIterator

from fastapi import HTTPException

from .chat_events import chat_json as chat_json
from .chat_events import chat_sse_generator as chat_sse_generator
from .config import settings
from .event_models import (
    EventContext,
    make_message_item,
    make_reasoning_item,
    make_response_object,
    make_tool_call_item,
    message_item_id,
    reasoning_item_id,
    response_event,
    tool_call_item_id,
)
from .models import Bill015Result, BridgeToolCall, NormalizedRequest, local_response_id
from .sse import encode_sse, split_text
from .upstream_errors import error_message_from_detail

TRUNCATION_MARKERS = (
    "[local proxy truncated answer at max_answer_chars]",
    "[local proxy truncated",
)


def response_json(result: Bill015Result, n: NormalizedRequest) -> dict[str, Any]:
    status, incomplete_details = _status_for_answer(result.answer)
    if result.bridge_mode == "tool_call" and result.tool_calls:
        return response_object_with_tool_calls(result.local_request_id, n, result.tool_calls, "completed", answer=result.answer)
    return response_object(result.local_request_id, n, status, result.answer, incomplete_details=incomplete_details)


def response_object(
    rid: str,
    n: NormalizedRequest,
    status: str,
    answer: str = "",
    *,
    incomplete_details: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if answer or status in {"completed", "incomplete"}:
        item_id = message_item_id(rid, 0)
        output = [make_message_item(item_id, "completed" if status in {"completed", "incomplete"} else "in_progress", [_message_part(answer)])]
    return make_response_object(
        rid,
        n,
        status,
        output=output,
        answer=answer,
        error=error,
        incomplete_details=incomplete_details,
    )


def tool_call_item(call: BridgeToolCall, item_id: str, status: str = "completed") -> dict[str, Any]:
    return make_tool_call_item(
        call,
        item_id,
        status,
        allow_web_search=settings.responses_allow_web_search_call_event,
    )


def response_object_with_tool_calls(
    rid: str,
    n: NormalizedRequest,
    calls: list[BridgeToolCall],
    status: str = "completed",
    *,
    answer: str = "",
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if answer.strip():
        output.append(make_message_item(message_item_id(rid, 0), "completed", [_message_part(answer)], phase="commentary"))
    output.extend(tool_call_item(call, tool_call_item_id(call), status) for call in calls)
    return make_response_object(rid, n, status, output=output, answer=answer, calls=calls)


def _message_part(answer: str) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "output_text", "text": answer, "logprobs": []}
    if settings.responses_emit_annotations:
        part["annotations"] = []
    return part


class ResponsesEventStream:
    def __init__(self, rid: str, n: NormalizedRequest) -> None:
        self.rid = rid
        self.n = n
        self.ctx = EventContext(response_id=rid, model=n.model)

    def encode(self, payload: dict[str, Any], event_name: str) -> bytes:
        payload["sequence_number"] = self.ctx.next_sequence()
        return encode_sse(payload, event_name)

    def event(self, event_type: str, **payload: Any) -> bytes:
        return self.encode(response_event(event_type, 0, **payload), event_type)

    def created_events(self) -> list[bytes]:
        created = make_response_object(self.rid, self.n, "in_progress", created_at=self.ctx.created_at)
        return [
            self.event("response.created", response=created),
            self.event("response.in_progress", response=created),
        ]

    def metadata_events(self) -> list[bytes]:
        # Native Codex treats response.metadata as server-owned moderation,
        # verification, and turn-state metadata. Do not synthesize it from
        # client metadata; preserving it in the final response object is enough
        # and avoids showing non-native stream artifacts.
        return []

    def keepalive_event(self) -> bytes:
        return self.event("response.in_progress", response=make_response_object(self.rid, self.n, "in_progress", created_at=self.ctx.created_at))

    def message_events(self, answer: str, *, output_index: int = 0, phase: str | None = None) -> list[bytes]:
        item_id = message_item_id(self.rid, output_index)
        final_part = {"type": "output_text", "text": answer, "logprobs": []}
        if settings.responses_emit_annotations:
            final_part["annotations"] = []
        events = [self.event("response.output_item.added", output_index=output_index, item=make_message_item(item_id, "in_progress", [], phase=phase))]
        for chunk in split_text(answer, settings.responses_chunk_size):
            events.append(self.event("response.output_text.delta", item_id=item_id, output_index=output_index, content_index=0, delta=chunk, logprobs=[]))
        events.extend(
            [
                self.event("response.output_item.done", output_index=output_index, item_id=item_id, item=make_message_item(item_id, "completed", [final_part], phase=phase)),
            ]
        )
        return events

    def reasoning_summary_events(self, summary: str, *, output_index: int = 0) -> list[bytes]:
        if not summary or not settings.responses_emit_reasoning_summary:
            return []
        item_id = reasoning_item_id(self.rid, output_index)
        part = {"type": "summary_text", "text": summary}
        events = [
            self.event("response.output_item.added", output_index=output_index, item=make_reasoning_item(item_id, "in_progress", [])),
            self.event("response.reasoning_summary_part.added", item_id=item_id, output_index=output_index, summary_index=0, part={"type": "summary_text", "text": ""}),
        ]
        for chunk in split_text(summary, settings.responses_chunk_size):
            events.append(self.event("response.reasoning_summary_text.delta", item_id=item_id, output_index=output_index, summary_index=0, delta=chunk))
        events.extend(
            [
                self.event("response.reasoning_summary_text.done", item_id=item_id, output_index=output_index, summary_index=0, text=summary),
                self.event("response.reasoning_summary_part.done", item_id=item_id, output_index=output_index, summary_index=0, part=part),
                self.event("response.output_item.done", output_index=output_index, item_id=item_id, item=make_reasoning_item(item_id, "completed", [part])),
            ]
        )
        return events

    def tool_call_events(self, call: BridgeToolCall, output_index: int) -> list[bytes]:
        item_id = tool_call_item_id(call)
        events = [self.event("response.output_item.added", output_index=output_index, item=tool_call_item(call, item_id, "in_progress"))]
        if call.call_type == "custom":
            events.extend(self._custom_tool_input_events(call, item_id, output_index))
        elif call.call_type not in {"tool_search", "web_search"}:
            events.extend(self._function_call_argument_events(call, item_id, output_index))
        events.append(self.event("response.output_item.done", output_index=output_index, item_id=item_id, item=tool_call_item(call, item_id, "completed")))
        return events

    def completed_tool_response(self, calls: list[BridgeToolCall], *, answer: str = "") -> bytes:
        return self.event("response.completed", response=response_object_with_tool_calls(self.rid, self.n, calls, "completed", answer=answer))

    def completed_message_response(self, answer: str) -> bytes:
        status, incomplete_details = _status_for_answer(answer)
        event_type = "response.incomplete" if status == "incomplete" else "response.completed"
        return self.event(event_type, response=response_object(self.rid, self.n, status, answer, incomplete_details=incomplete_details))

    def failed_events(self, error: HTTPException) -> list[bytes]:
        error_obj = _error_object(error)
        failed = response_object(self.rid, self.n, "failed", error=error_obj)
        return [
            self.event("response.failed", response=failed),
            self.event("error", error=error_obj),
        ]

    def cancelled_event(self) -> bytes:
        cancelled = response_object(self.rid, self.n, "cancelled")
        return self.event("response.cancelled", response=cancelled)

    def _custom_tool_input_events(self, call: BridgeToolCall, item_id: str, output_index: int) -> list[bytes]:
        events = []
        for chunk in split_text(call.arguments, settings.responses_chunk_size):
            events.append(self.event("response.custom_tool_call_input.delta", item_id=item_id, output_index=output_index, call_id=call.id, delta=chunk))
        return events

    def _function_call_argument_events(self, call: BridgeToolCall, item_id: str, output_index: int) -> list[bytes]:
        events = []
        base = {"item_id": item_id, "output_index": output_index, "call_id": call.id, "name": call.name}
        if call.namespace:
            base["namespace"] = call.namespace
        for chunk in split_text(call.arguments, settings.responses_chunk_size):
            events.append(self.event("response.function_call_arguments.delta", **base, delta=chunk))
        return events


async def responses_sse_generator(result_coro, n: NormalizedRequest, rid: str | None = None) -> AsyncIterator[bytes]:
    rid = rid or local_response_id()
    stream = ResponsesEventStream(rid, n)
    task: asyncio.Task | None = None
    try:
        for chunk in stream.created_events():
            yield chunk
        for chunk in stream.metadata_events():
            yield chunk

        task = asyncio.ensure_future(result_coro)
        heartbeat_seconds = max(0.05, settings.client_heartbeat_interval_ms / 1000)
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
            if done:
                break
            # Keep the Codex client and any intermediate proxy from declaring
            # the stream dead while the upstream model is still reasoning or
            # while the local bridge is waiting for emit_value arguments.
            # Codex Desktop's idle detector expects parsed SSE events; comments
            # can still leave the UI reconnecting during long gpt-5.6-sol gaps.
            yield stream.keepalive_event()
        result: Bill015Result = task.result()
        result.local_request_id = rid
        if result.bridge_mode == "tool_call" and result.tool_calls:
            tool_start_index = 0
            if result.answer.strip():
                for chunk in stream.message_events(result.answer, output_index=0, phase="commentary"):
                    yield chunk
                tool_start_index = 1
            for output_index, call in enumerate(result.tool_calls, start=tool_start_index):
                for chunk in stream.tool_call_events(call, output_index):
                    yield chunk
            yield stream.completed_tool_response(result.tool_calls, answer=result.answer)
            yield b"data: [DONE]\n\n"
            return

        for chunk in stream.reasoning_summary_events(getattr(result, "reasoning_summary", "") or "", output_index=0):
            yield chunk
        for chunk in stream.message_events(result.answer):
            yield chunk
        yield stream.completed_message_response(result.answer)
        yield b"data: [DONE]\n\n"
    except HTTPException as e:
        for chunk in stream.failed_events(e):
            yield chunk
        yield b"data: [DONE]\n\n"
    except asyncio.CancelledError:
        if task and not task.done():
            task.cancel()
        try:
            yield stream.cancelled_event()
            yield b"data: [DONE]\n\n"
        finally:
            raise
    except Exception as e:
        err = HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
        for chunk in stream.failed_events(err):
            yield chunk
        yield b"data: [DONE]\n\n"
    finally:
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _stream_metadata(n: NormalizedRequest) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source in (n.metadata, n.client_metadata):
        if isinstance(source, dict):
            for key, value in source.items():
                if key in {"safety_identifier", "user", "rate_limits", "credits", "openai_verification_recommendation"}:
                    continue
                if value in (None, "", [], {}):
                    continue
                metadata[str(key)] = value
    return metadata


def _status_for_answer(answer: str) -> tuple[str, dict[str, Any] | None]:
    if settings.responses_emit_incomplete_on_truncation and any(marker in (answer or "") for marker in TRUNCATION_MARKERS):
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def _error_object(error: HTTPException) -> dict[str, Any]:
    code = error.status_code
    err: dict[str, Any] = {
        "code": "local_proxy_error",
        "message": error_message_from_detail(error.detail),
        "type": "invalid_request_error" if 400 <= code < 500 else "server_error",
    }
    if isinstance(error.detail, dict) and isinstance(error.detail.get("code"), str):
        err["code"] = error.detail["code"]
    if isinstance(error.detail, dict) and "upstream_status" in error.detail:
        err["code"] = "upstream_error"
        err["upstream_status"] = error.detail.get("upstream_status")
    return err
