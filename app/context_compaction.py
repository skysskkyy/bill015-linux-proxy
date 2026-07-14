from __future__ import annotations

import copy
from typing import Any

from .usage_estimator import estimate_request_input_tokens

COMPACTION_SUMMARY_PREFIX = "CONTEXT CHECKPOINT COMPACTION"
TOOL_CALL_TYPES = {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "computer_call"}
TOOL_OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output", "tool_search_output", "computer_call_output", "tool_result"}


def payload_input_tokens(payload: dict[str, Any]) -> int:
    return estimate_request_input_tokens(payload)


def context_threshold_tokens(context_window_tokens: int, percent: int) -> int:
    return max(8_000, int(max(1, context_window_tokens) * max(1, min(percent, 99)) / 100))


def _item_type(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("type") or ("message" if item.get("role") else ""))


def _item_text(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    value = item.get("content", item.get("text", item.get("output", "")))
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("output_text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(value or "")


def _is_compaction_summary(item: Any) -> bool:
    return COMPACTION_SUMMARY_PREFIX in _item_text(item)


def _latest_user_index(items: list[Any]) -> int | None:
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if (
            isinstance(item, dict)
            and str(item.get("role") or "") == "user"
            and _item_text(item).strip()
            and not _is_compaction_summary(item)
        ):
            return index
    return None


def _latest_tool_batch(items: list[Any], user_index: int | None) -> tuple[int, int] | None:
    last_output = next((index for index in range(len(items) - 1, -1, -1) if _item_type(items[index]) in TOOL_OUTPUT_TYPES), None)
    if last_output is None:
        return None
    start = last_output
    while start > 0 and _item_type(items[start - 1]) in TOOL_CALL_TYPES | TOOL_OUTPUT_TYPES:
        start -= 1
    return start, last_output + 1

def _clip_string(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    head = max_chars // 3
    tail = max_chars - head
    return value[:head] + f"\n...[local compaction omitted {len(value) - max_chars} chars]...\n" + value[-tail:]


def _clip_large_fields(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return _clip_string(value, max_chars)
    if isinstance(value, list):
        return [_clip_large_fields(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: (_clip_large_fields(item, max_chars) if key in {"output", "content", "text", "summary"} else item) for key, item in value.items()}
    return value


def _summary_item(removed: list[Any]) -> dict[str, Any]:
    lines = [COMPACTION_SUMMARY_PREFIX, "Older conversation history was compacted locally before retry."]
    for item in removed[-12:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or _item_type(item) or "item")
        text = " ".join(_item_text(item).split())
        if text:
            lines.append(f"[{role}] {text[:600]}")
    # Match Codex compaction history shape: summaries are user-message-like
    # input items prefixed with CONTEXT CHECKPOINT COMPACTION.  The proxy inserts
    # this before the latest real user request, which is equivalent to native
    # pre-turn compaction where the summary becomes the last old history item
    # and the new user message is appended afterwards.
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "\n".join(lines)}]}


def _remove_related_tool_items(items: list[Any], index: int) -> list[Any]:
    item = items[index]
    call_id = str(item.get("call_id") or "") if isinstance(item, dict) else ""
    if not call_id or _item_type(item) not in TOOL_CALL_TYPES | TOOL_OUTPUT_TYPES:
        return [items.pop(index)]
    removed = [candidate for candidate in items if isinstance(candidate, dict) and str(candidate.get("call_id") or "") == call_id]
    items[:] = [candidate for candidate in items if not (isinstance(candidate, dict) and str(candidate.get("call_id") or "") == call_id)]
    return removed


def compact_payload_history(payload: dict[str, Any], target_tokens: int, *, aggressive: bool = False, latest_tool_output_max_chars: int = 12_000) -> tuple[dict[str, Any], int, int]:
    """Keep initial context and the latest turn; trim oldest history first like Codex."""
    compacted = copy.deepcopy(payload)
    items = compacted.get("input")
    if not isinstance(items, list) or not items:
        return compacted, 0, 0
    user_index = _latest_user_index(items)
    tool_batch = _latest_tool_batch(items, user_index)
    clipped = 0
    for index, item in enumerate(list(items)):
        if _item_type(item) not in TOOL_OUTPUT_TYPES:
            continue
        in_latest = tool_batch is not None and tool_batch[0] <= index < tool_batch[1]
        limit = latest_tool_output_max_chars if in_latest else max(2_000, latest_tool_output_max_chars // 3)
        if aggressive:
            limit = max(2_000, limit // 2)
        replacement = _clip_large_fields(item, limit)
        if replacement != item:
            items[index] = replacement
            clipped += 1
    removed: list[Any] = []
    while payload_input_tokens(compacted) > target_tokens and len(items) > 1:
        user_index = _latest_user_index(items)
        tool_batch = _latest_tool_batch(items, user_index)
        protected: set[int] = set()
        if user_index is not None:
            protected.add(user_index)
        if tool_batch is not None:
            protected.update(range(tool_batch[0], tool_batch[1]))
        for index, item in enumerate(items):
            if isinstance(item, dict) and str(item.get("role") or "") in {"system", "developer"}:
                protected.add(index)
        removable = next((index for index in range(len(items)) if index not in protected and not _is_compaction_summary(items[index])), None)
        if removable is None:
            break
        removed.extend(_remove_related_tool_items(items, removable))
    if removed:
        user_index = _latest_user_index(items)
        tool_batch = _latest_tool_batch(items, user_index)
        insert_at = tool_batch[0] if tool_batch is not None else (user_index if user_index is not None else len(items))
        items.insert(insert_at, _summary_item(removed))
    if aggressive and payload_input_tokens(compacted) > target_tokens:
        for index, item in enumerate(list(items)):
            if _item_type(item) in TOOL_OUTPUT_TYPES:
                replacement = _clip_large_fields(item, 2_000)
                if replacement != item:
                    items[index] = replacement
                    clipped += 1
    if aggressive and payload_input_tokens(compacted) > target_tokens:
        # Last-resort native-aligned behavior: Codex keeps recent user messages,
        # but build_compacted_history truncates them to a token budget when
        # necessary.  If the latest user request by itself is too large for the
        # upstream window, preserve its head/tail with an explicit marker rather
        # than repeatedly failing with context_length_exceeded.
        user_index = _latest_user_index(items)
        if user_index is not None:
            limit = max(4_000, min(60_000, target_tokens * 3))
            while payload_input_tokens(compacted) > target_tokens and limit >= 4_000:
                replacement = _clip_large_fields(items[user_index], limit)
                if replacement == items[user_index]:
                    limit //= 2
                    continue
                items[user_index] = replacement
                clipped += 1
                limit //= 2
    return compacted, len(removed), clipped


def reinforce_final_action(payload: dict[str, Any]) -> dict[str, Any]:
    retried = copy.deepcopy(payload)
    retried["instructions"] = str(retried.get("instructions") or "") + (
        "\n\nThe previous upstream attempt ended after reasoning without a final tool call. "
        "Do not emit more reasoning-only output; immediately call the required final/action tool exactly once."
    )
    return retried
