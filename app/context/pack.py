from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings, settings
from ..ingest.text import item_text, item_type
from ..protocol.ids import coerce_item_id
from ..protocol.models import TOOL_CALL_TYPES, TOOL_OUTPUT_TYPES, Turn
from .tokens import items_tokens, threshold_tokens

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


@dataclass
class PackResult:
    items: list[dict[str, Any]]
    notices: list[str]
    clipped_outputs: int
    dropped_items: int
    tokens: int
    needs_compact: bool = False


def _last_compaction_index(items: list[dict[str, Any]]) -> int | None:
    for index in range(len(items) - 1, -1, -1):
        if item_type(items[index]) == "compaction":
            return index
    return None


def pack_turn_items(turn: Turn, cfg: Settings = settings, *, aggressive: bool = False) -> PackResult:
    from .windows import compact_threshold_tokens, model_context_window

    items = [item for item in turn.items if isinstance(item, dict)]
    last_cmp = _last_compaction_index(items)
    kept = items[last_cmp:] if last_cmp is not None else list(items)
    packed = [coerce_item_id(dict(item)) for item in kept]
    used = items_tokens(packed)
    threshold = compact_threshold_tokens(turn.model, cfg)
    if aggressive:
        threshold = min(threshold, threshold_tokens(model_context_window(turn.model, cfg), max(40, cfg.auto_compact_percent - 15)))
    needs_compact = used > threshold
    notices: list[str] = []
    if needs_compact:
        notices.append(
            f"{LOSS_PREFIX} context is over the compact threshold; "
            "older turns will be folded into a compaction item instead of dropped."
        )
    turn.loss_notices = notices
    return PackResult(
        items=packed,
        notices=notices,
        clipped_outputs=0,
        dropped_items=0,
        tokens=used,
        needs_compact=needs_compact,
    )
