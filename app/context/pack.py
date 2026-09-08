from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings, settings
from ..ingest.text import item_text, item_type
from ..protocol.models import TOOL_CALL_TYPES, TOOL_OUTPUT_TYPES, Turn
from .tokens import count_text_tokens, item_tokens, items_tokens, pack_budget, threshold_tokens

LOSS_PREFIX = "[LOCAL CONTEXT LOSS]"


def _is_user(item: dict[str, Any]) -> bool:
    return str(item.get("role") or "") == "user" and item_type(item) in {"", "message"}


def _latest_user_index(items: list[dict[str, Any]]) -> int | None:
    for index in range(len(items) - 1, -1, -1):
        if _is_user(items[index]) and item_text(items[index]).strip():
            return index
    return None


def _latest_tool_batch(items: list[dict[str, Any]]) -> tuple[int, int] | None:
    last_output = next((i for i in range(len(items) - 1, -1, -1) if item_type(items[i]) in TOOL_OUTPUT_TYPES), None)
    if last_output is None:
        last_call = next((i for i in range(len(items) - 1, -1, -1) if item_type(items[i]) in TOOL_CALL_TYPES), None)
        if last_call is None:
            return None
        start = last_call
        while start > 0 and item_type(items[start - 1]) in TOOL_CALL_TYPES:
            start -= 1
        return start, last_call + 1
    start = last_output
    while start > 0 and item_type(items[start - 1]) in TOOL_CALL_TYPES | TOOL_OUTPUT_TYPES:
        start -= 1
    return start, last_output + 1


def _head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(64, max_chars // 2)
    return text[:keep] + "\n…\n" + text[-keep:]


def _clip_output_item(item: dict[str, Any], max_chars: int) -> tuple[dict[str, Any], bool]:
    cloned = dict(item)
    text = item_text(item)
    if len(text) <= max_chars:
        return cloned, False
    clipped = _head_tail(text, max_chars)
    notice = f"{LOSS_PREFIX} clipped tool output {len(text)} → {len(clipped)} chars."
    if "output" in cloned:
        cloned["output"] = clipped
    elif "content" in cloned:
        cloned["content"] = [{"type": "output_text", "text": clipped}]
    else:
        cloned["text"] = clipped
    cloned["_local_clip_notice"] = notice
    return cloned, True


def _loss_item(message: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": message}],
    }


def _reasoning_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item_type(item) == "reasoning" and item.get("encrypted_content")]


@dataclass
class PackResult:
    items: list[dict[str, Any]]
    notices: list[str]
    clipped_outputs: int
    dropped_items: int
    tokens: int


def pack_turn_items(turn: Turn, cfg: Settings = settings, *, aggressive: bool = False) -> PackResult:
    items = [item for item in turn.items if isinstance(item, dict)]
    budget = pack_budget(cfg, turn.max_output_tokens)
    if aggressive:
        budget = min(budget, threshold_tokens(cfg.context_window_tokens, max(40, cfg.compact_target_percent - 15)))
    max_output_chars = max(800, cfg.latest_tool_output_max_chars // (2 if aggressive else 1))

    user_index = _latest_user_index(items)
    batch_span = _latest_tool_batch(items)
    pinned_indexes: set[int] = set()
    if user_index is not None:
        pinned_indexes.add(user_index)
    if batch_span is not None:
        pinned_indexes.update(range(batch_span[0], batch_span[1]))

    reasoning = _reasoning_items(items)
    pinned: list[dict[str, Any]] = []
    notices: list[str] = []
    clipped = 0

    if user_index is not None:
        user_item = dict(items[user_index])
        text = item_text(user_item)
        if count_text_tokens(text) > budget // 2:
            clipped_text = _head_tail(text, max(2000, max_output_chars * 2))
            notices.append(f"{LOSS_PREFIX} clipped current user request {len(text)} → {len(clipped_text)} chars.")
            user_item["content"] = [{"type": "input_text", "text": clipped_text}]
        pinned.append(user_item)

    if batch_span is not None:
        for index in range(batch_span[0], batch_span[1]):
            item = items[index]
            if item_type(item) in TOOL_OUTPUT_TYPES:
                clipped_item, did = _clip_output_item(item, max_output_chars)
                pinned.append(clipped_item)
                if did:
                    clipped += 1
                    notices.append(str(clipped_item.get("_local_clip_notice") or ""))
            else:
                pinned.append(dict(item))

    for item in reasoning:
        pinned.append(dict(item))

    used = items_tokens(pinned)
    history: list[dict[str, Any]] = []
    dropped = 0
    for index in range(len(items) - 1, -1, -1):
        if index in pinned_indexes:
            continue
        item = items[index]
        if item_type(item) == "reasoning":
            continue
        candidate = dict(item)
        if item_type(item) in TOOL_OUTPUT_TYPES:
            candidate, did = _clip_output_item(candidate, max_output_chars)
            if did:
                clipped += 1
        cost = item_tokens(candidate)
        if used + cost > budget:
            dropped += 1
            continue
        history.append(candidate)
        used += cost

    history.reverse()
    packed = list(history)
    # Current user and latest tool batch stay last so the model sees them at the tail.
    packed.extend(pinned)
    if notices or dropped:
        notice = (
            f"{LOSS_PREFIX} kept current user + latest tool batch; "
            f"dropped {dropped} older items; clipped {clipped} tool outputs. "
            "Re-read files before exact claims."
        )
        notices = [n for n in notices if n] + [notice]
        packed.insert(0, _loss_item(notice))
        used = items_tokens(packed)

    turn.loss_notices = notices
    return PackResult(items=packed, notices=notices, clipped_outputs=clipped, dropped_items=dropped, tokens=used)
