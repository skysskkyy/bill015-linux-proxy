from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings, settings
from ..ingest.text import item_text, item_type
from ..protocol.ids import coerce_item_id
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

    if user_index is not None:
        user_item = dict(items[user_index])
        text = item_text(user_item)
        if count_text_tokens(text) > budget // 2:
            clipped_text = _head_tail(text, max(2000, max_output_chars * 2))
            notices.append(f"{LOSS_PREFIX} clipped current user request {len(text)} → {len(clipped_text)} chars.")
            user_item["content"] = [{"type": "input_text", "text": clipped_text}]
        pinned.append(coerce_item_id(user_item))

    if batch_span is not None:
        for index in range(batch_span[0], batch_span[1]):
            pinned.append(coerce_item_id(dict(items[index])))

    for item in reasoning:
        pinned.append(coerce_item_id(dict(item)))

    used = items_tokens(pinned)
    history: list[dict[str, Any]] = []
    dropped = 0
    for index in range(len(items) - 1, -1, -1):
        if index in pinned_indexes:
            continue
        item = items[index]
        if item_type(item) == "reasoning":
            continue
        candidate = coerce_item_id(dict(item))
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
            f"dropped {dropped} older items. "
            "Re-read files before exact claims."
        )
        notices = [n for n in notices if n] + [notice]
        packed.insert(0, _loss_item(notice))
        used = items_tokens(packed)

    packed = [coerce_item_id(item) if isinstance(item, dict) else item for item in packed]
    turn.loss_notices = notices
    return PackResult(items=packed, notices=notices, clipped_outputs=0, dropped_items=dropped, tokens=used)
