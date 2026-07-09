from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from .config import settings
from .models import BridgeToolCall

MESSAGE_OVERHEAD = 8
CONTENT_PART_OVERHEAD = 2
FUNCTION_CALL_OVERHEAD = 12
CUSTOM_TOOL_CALL_OVERHEAD = 10
TOOL_HISTORY_OUTPUT_MAX_CHARS = 12_000
OLDER_TOOL_HISTORY_OUTPUT_MAX_CHARS = 4_000
TOKENIZER_NAME = "o200k_base"

_TIKTOKEN_ENCODER: Any | None = None
_TIKTOKEN_CHECKED = False
_SHA_TEXT_CACHE: dict[str, int] = {}


@dataclass
class UsageEstimate:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    text_tokens: int = 0
    tool_schema_tokens: int = 0
    tool_history_tokens: int = 0
    image_tokens: int = 0
    reasoning_tokens: int = 0
    metadata_tokens: int = 0
    native_context_tokens: int = 0
    upstream_payload_tokens: int = 0
    estimate_method: str = "heuristic"
    confidence: str = "medium"

    def finalize(self) -> "UsageEstimate":
        self.input_tokens = max(
            0,
            self.text_tokens
            + self.tool_schema_tokens
            + self.tool_history_tokens
            + self.image_tokens
            + self.reasoning_tokens
            + self.metadata_tokens,
        )
        self.total_tokens = self.input_tokens + max(0, self.output_tokens)
        if not self.native_context_tokens:
            self.native_context_tokens = self.input_tokens
        if not self.upstream_payload_tokens:
            self.upstream_payload_tokens = self.input_tokens
        if self.cached_tokens > self.input_tokens:
            self.cached_tokens = self.input_tokens
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_responses_usage_from_body(body: dict[str, Any] | None) -> UsageEstimate:
    estimate = UsageEstimate()
    if not isinstance(body, dict):
        return estimate.finalize()

    estimate.text_tokens += estimate_instructions_tokens(body.get("instructions"))
    text_tokens, tool_history_tokens, image_tokens, reasoning_tokens = estimate_input_items_tokens(body.get("input"))
    estimate.text_tokens += text_tokens
    estimate.tool_history_tokens += tool_history_tokens
    estimate.image_tokens += image_tokens if settings.usage_include_images else 0
    estimate.reasoning_tokens += reasoning_tokens
    if settings.usage_include_tools_schema:
        estimate.tool_schema_tokens += estimate_tools_schema_tokens(body.get("tools"))
    estimate.metadata_tokens += estimate_metadata_tokens(body.get("metadata"), body.get("client_metadata"))
    estimate.metadata_tokens += estimate_json_tokens(body.get("text")) if isinstance(body.get("text"), dict) else 0
    estimate.finalize()
    estimate.cached_tokens = estimate_cached_tokens_from_body(body, estimate)
    estimate.finalize()
    return estimate


def estimate_chat_usage_from_body(body: dict[str, Any] | None) -> UsageEstimate:
    estimate = UsageEstimate()
    if not isinstance(body, dict):
        return estimate.finalize()
    estimate.text_tokens += estimate_chat_messages_tokens(body.get("messages"))
    if settings.usage_include_tools_schema:
        estimate.tool_schema_tokens += estimate_tools_schema_tokens(body.get("tools"))
    estimate.metadata_tokens += estimate_metadata_tokens(body.get("metadata"), None)
    estimate.finalize()
    estimate.cached_tokens = estimate_cached_tokens_from_body(body, estimate)
    estimate.finalize()
    return estimate


def estimate_request_input_tokens(body: dict[str, Any] | None) -> int:
    return estimate_responses_usage_from_body(body).input_tokens


def estimate_text_tokens(text: str | None, *, json_like: bool = False) -> int:
    if not text:
        return 0
    text = str(text)
    tokenizer_tokens = _estimate_with_tiktoken(text)
    if tokenizer_tokens is not None:
        return tokenizer_tokens
    return _estimate_json_tokens_heuristic(text) if json_like else _weighted_char_estimate(text)


def estimate_json_tokens(value: Any) -> int:
    if value is None:
        return 0
    return estimate_text_tokens(normalize_json_for_usage(value), json_like=True)


def estimate_instructions_tokens(instructions: Any) -> int:
    return estimate_content_tokens(instructions) + (MESSAGE_OVERHEAD if instructions else 0)


def estimate_chat_messages_tokens(messages: Any) -> int:
    if not isinstance(messages, list):
        return estimate_content_tokens(messages)
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            total += MESSAGE_OVERHEAD
            total += estimate_text_tokens(str(msg.get("role") or "user"))
            total += estimate_content_tokens(msg.get("content", ""))
        else:
            total += MESSAGE_OVERHEAD + estimate_content_tokens(msg)
    return total


def estimate_input_items_tokens(input_items: Any) -> tuple[int, int, int, int]:
    if isinstance(input_items, str):
        return estimate_text_tokens(input_items), 0, 0, 0
    if not isinstance(input_items, list):
        return estimate_content_tokens(input_items), 0, estimate_image_tokens(input_items), 0

    text_tokens = 0
    tool_history_tokens = 0
    image_tokens = 0
    reasoning_tokens = 0
    tool_output_seen = 0
    total_tool_outputs = sum(
        1
        for item in input_items
        if isinstance(item, dict)
        and str(item.get("type") or "") in {"function_call_output", "custom_tool_call_output", "tool_result", "tool_search_output", "computer_call_output"}
    )

    for item in input_items:
        if not isinstance(item, dict):
            text_tokens += estimate_content_tokens(item)
            image_tokens += estimate_image_tokens(item)
            continue

        typ = str(item.get("type") or "")
        role = item.get("role")
        if typ == "message" or role:
            text_tokens += MESSAGE_OVERHEAD + estimate_text_tokens(str(role or "message"))
            content = item.get("content", item.get("text", ""))
            text_tokens += estimate_content_tokens(content)
            image_tokens += estimate_image_tokens(content)
            continue

        if typ in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "computer_call"}:
            tool_history_tokens += estimate_json_tokens(item) + FUNCTION_CALL_OVERHEAD
            continue

        if typ in {"function_call_output", "custom_tool_call_output", "tool_result", "tool_search_output", "computer_call_output"}:
            tool_output_seen += 1
            max_output = TOOL_HISTORY_OUTPUT_MAX_CHARS if tool_output_seen > max(0, total_tool_outputs - 3) else OLDER_TOOL_HISTORY_OUTPUT_MAX_CHARS
            tool_history_tokens += estimate_json_tokens(_clip_tool_output_item(item, max_output)) + FUNCTION_CALL_OVERHEAD
            continue

        if typ == "reasoning":
            reasoning_tokens += estimate_reasoning_item_tokens(item)
            continue

        text_tokens += estimate_json_tokens(_scrub_large_media(item))
        image_tokens += estimate_image_tokens(item)

    return text_tokens, tool_history_tokens, image_tokens, reasoning_tokens


def estimate_content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            total += CONTENT_PART_OVERHEAD
            if isinstance(part, dict):
                typ = str(part.get("type") or "")
                if typ in {"input_image", "image"} or "image_url" in part:
                    continue
                if isinstance(part.get("text"), str):
                    total += estimate_text_tokens(part["text"])
                else:
                    total += estimate_json_tokens(_scrub_large_media(part))
            else:
                total += estimate_content_tokens(part)
        return total
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return estimate_text_tokens(content["text"])
        return estimate_json_tokens(_scrub_large_media(content))
    return estimate_text_tokens(str(content))


def estimate_tools_schema_tokens(tools: Any) -> int:
    if not tools:
        return 0
    return estimate_text_tokens(normalize_json_for_usage(tools), json_like=True)


def estimate_reasoning_item_tokens(item: dict[str, Any]) -> int:
    total = 4
    summary = item.get("summary")
    if summary:
        total += estimate_content_tokens(summary)
    if item.get("encrypted_content"):
        total += 16
    return total


def estimate_metadata_tokens(metadata: Any, client_metadata: Any) -> int:
    total = 0
    if metadata:
        total += estimate_json_tokens(metadata)
    if client_metadata:
        total += estimate_json_tokens(client_metadata)
    return total


def estimate_image_tokens(input_items: Any) -> int:
    total = 0
    for image in _iter_image_parts(input_items):
        total += estimate_image_part_tokens(image)
    return total


def estimate_image_part_tokens(part: dict[str, Any]) -> int:
    detail = str(part.get("detail") or "high").lower()
    image_url = part.get("image_url") or part.get("url") or part.get("source") or part.get("data")
    if isinstance(image_url, dict):
        detail = str(image_url.get("detail") or detail)
        image_url = image_url.get("url") or image_url.get("data")
    if isinstance(image_url, str) and image_url.startswith("data:image/") and ";base64," in image_url:
        size = _image_size_from_data_url(image_url)
        if size:
            return estimate_image_tokens_by_size(size[0], size[1], detail)
        raw_bytes = _data_url_raw_size(image_url)
        return _estimate_image_tokens_by_bytes(raw_bytes)
    if isinstance(image_url, str) and image_url:
        return 300 if detail == "low" else 800
    return 0


def estimate_image_tokens_by_size(width: int, height: int, detail: str = "high") -> int:
    if str(detail).lower() == "low":
        return 85
    tiles = max(1, math.ceil(max(1, width) / 512) * math.ceil(max(1, height) / 512))
    return 85 + 170 * tiles


def estimate_cached_tokens_from_body(body: dict[str, Any], estimate: UsageEstimate) -> int:
    input_tokens = max(0, int(estimate.input_tokens))
    client_metadata = body.get("client_metadata") if isinstance(body.get("client_metadata"), dict) else {}
    request_kind = str(body.get("request_kind") or "")
    turn_md = _parse_turn_metadata(client_metadata)
    is_compaction = request_kind == "compaction" or turn_md.get("request_kind") == "compaction" or isinstance(turn_md.get("compaction"), dict)
    if is_compaction:
        return int(input_tokens * 0.90)
    if input_tokens < 4096:
        return 0
    if isinstance(body.get("prompt_cache_key"), str) and body.get("prompt_cache_key"):
        return int(input_tokens * 0.85)
    if isinstance(client_metadata, dict) and client_metadata.get("thread_id"):
        return int(input_tokens * 0.75)
    return int(input_tokens * settings.usage_cache_ratio_default)


def estimate_response_output_tokens(answer: str = "", calls: list[BridgeToolCall] | None = None) -> int:
    if calls:
        return sum(estimate_tool_call_output_tokens(call) for call in calls)
    return estimate_text_tokens(answer)


def estimate_tool_call_output_tokens(call: BridgeToolCall) -> int:
    overhead = CUSTOM_TOOL_CALL_OVERHEAD if call.call_type == "custom" else FUNCTION_CALL_OVERHEAD
    return overhead + estimate_text_tokens(call.name) + estimate_text_tokens(call.arguments, json_like=call.call_type != "custom")


def build_response_usage(n: Any, *, answer: str = "", calls: list[BridgeToolCall] | None = None) -> dict[str, Any]:
    base = _ensure_estimate(n)
    output_tokens = estimate_response_output_tokens(answer, calls)
    return {
        "input_tokens": base.input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": base.input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": base.cached_tokens},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def build_chat_usage(n: Any, *, answer: str = "") -> dict[str, Any]:
    base = _ensure_estimate(n)
    completion_tokens = estimate_text_tokens(answer)
    return {
        "prompt_tokens": base.input_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": base.input_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": base.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def normalize_json_for_usage(value: Any) -> str:
    return json.dumps(_scrub_large_media(value), ensure_ascii=False, separators=(",", ":"))


def usage_estimate_dict(value: Any) -> dict[str, Any]:
    estimate = value if isinstance(value, UsageEstimate) else None
    if estimate is None and hasattr(value, "usage_estimate"):
        estimate = value.usage_estimate if isinstance(value.usage_estimate, UsageEstimate) else None
    return estimate.to_dict() if estimate else {}


def _ensure_estimate(n: Any) -> UsageEstimate:
    estimate = getattr(n, "usage_estimate", None)
    if isinstance(estimate, UsageEstimate):
        return estimate
    fallback = UsageEstimate(input_tokens=max(0, int(getattr(n, "estimated_input_tokens", 0) or 0)))
    fallback.cached_tokens = int(fallback.input_tokens * 0.9) if fallback.input_tokens > 4096 else 0
    return fallback.finalize()


def _estimate_with_tiktoken(text: str) -> int | None:
    global _TIKTOKEN_CHECKED, _TIKTOKEN_ENCODER
    if not settings.usage_prefer_tiktoken or settings.usage_estimator == "heuristic":
        return None
    if len(text) > settings.usage_max_text_for_exact_tokenize:
        return None
    if not _TIKTOKEN_CHECKED:
        _TIKTOKEN_CHECKED = True
        try:
            import tiktoken  # type: ignore

            _TIKTOKEN_ENCODER = tiktoken.get_encoding(TOKENIZER_NAME)
        except Exception:
            _TIKTOKEN_ENCODER = None
    if _TIKTOKEN_ENCODER is None:
        return None
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    cached = _SHA_TEXT_CACHE.get(digest)
    if cached is not None:
        return cached
    tokens = len(_TIKTOKEN_ENCODER.encode(text))
    if len(_SHA_TEXT_CACHE) > 4096:
        _SHA_TEXT_CACHE.clear()
    _SHA_TEXT_CACHE[digest] = tokens
    return max(1, tokens)


@lru_cache(maxsize=4096)
def _weighted_char_estimate_cached(text: str) -> int:
    cjk = 0
    ascii_nonspace = 0
    whitespace = 0
    other = 0
    for ch in text:
        code = ord(ch)
        if ch.isspace():
            whitespace += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF:
            cjk += 1
        elif code < 128:
            ascii_nonspace += 1
        else:
            other += 1
    tokens = cjk / 1.8 + ascii_nonspace / 4.0 + whitespace / 12.0 + other / 2.3
    return max(1, math.ceil(tokens))


def _weighted_char_estimate(text: str) -> int:
    if len(text) <= 100_000:
        return _weighted_char_estimate_cached(text)
    return sum(_weighted_char_estimate_cached(text[i : i + 100_000]) for i in range(0, len(text), 100_000))


def _estimate_json_tokens_heuristic(text: str) -> int:
    if not text:
        return 0
    if len(text) <= 100_000:
        return max(1, math.ceil(len(text) / 3.4))
    return sum(_estimate_json_tokens_heuristic(text[i : i + 100_000]) for i in range(0, len(text), 100_000))


def _iter_image_parts(value: Any):
    if isinstance(value, dict):
        typ = str(value.get("type") or "")
        if typ in {"input_image", "image"} or "image_url" in value:
            yield value
        for child in value.values():
            yield from _iter_image_parts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_image_parts(child)


def _image_size_from_data_url(value: str) -> tuple[int, int] | None:
    try:
        _, b64 = value.split(",", 1)
        raw = base64.b64decode(b64, validate=False)
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return None
        image = Image.open(io.BytesIO(raw))
        return int(image.size[0]), int(image.size[1])
    except Exception:
        return None


def _data_url_raw_size(value: str) -> int:
    try:
        _, b64 = value.split(",", 1)
        return len(base64.b64decode(b64, validate=False))
    except Exception:
        return 0


def _estimate_image_tokens_by_bytes(raw_bytes: int) -> int:
    if raw_bytes <= 0:
        return 800
    if raw_bytes < 50_000:
        return 300
    if raw_bytes < 500_000:
        return 800
    return 1500 + min(3000, raw_bytes // 500_000 * 500)


def _clip_tool_output_item(item: dict[str, Any], max_output_chars: int) -> dict[str, Any]:
    clipped = dict(item)
    for key in ("output", "result"):
        if isinstance(clipped.get(key), str):
            clipped[key] = _clip_text(clipped[key], max_output_chars)
    return _scrub_large_media(clipped)


def _scrub_large_media(value: Any, *, max_string: int = 4096) -> Any:
    if isinstance(value, dict):
        return {str(k): _scrub_large_media(v, max_string=max_string) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_large_media(v, max_string=max_string) for v in value]
    if isinstance(value, str):
        if value.startswith("data:image/") and ";base64," in value:
            return f"<image_data_url chars={len(value)}>"
        return _clip_text(value, max_string)
    return value


def _clip_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head
    return text[:head] + f"\n...[truncated {len(text) - max_chars} chars]...\n" + text[-tail:]


def _parse_turn_metadata(client_metadata: Any) -> dict[str, Any]:
    if not isinstance(client_metadata, dict):
        return {}
    raw = client_metadata.get("x-codex-turn-metadata")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
