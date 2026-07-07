from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Iterator, Optional


@dataclass
class SSEEvent:
    event: str | None
    data: str

    @property
    def json(self) -> dict[str, Any] | None:
        if self.data == "[DONE]":
            return None
        try:
            obj = json.loads(self.data)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def encode_sse(data: Any, event: str | None = None) -> bytes:
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SSEEvent]:
    event: str | None = None
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data_parts:
                yield SSEEvent(event=event, data="\n".join(data_parts))
            event = None
            data_parts = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if data_parts:
        yield SSEEvent(event=event, data="\n".join(data_parts))


async def parse_async_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    event: str | None = None
    data_parts: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data_parts:
                yield SSEEvent(event=event, data="\n".join(data_parts))
            event = None
            data_parts = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if data_parts:
        yield SSEEvent(event=event, data="\n".join(data_parts))


def split_text(text: str, chunk_size: int = 512) -> Iterator[str]:
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
