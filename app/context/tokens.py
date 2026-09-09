from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from ..config import Settings, settings
from .image_tokens import estimate_image_part, is_image_part

_ENCODER = None
_ENCODER_FAILED = False


def _encoder():
    global _ENCODER, _ENCODER_FAILED
    if _ENCODER_FAILED:
        return None
    if _ENCODER is not None:
        return _ENCODER
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("o200k_base")
        return _ENCODER
    except Exception:
        _ENCODER_FAILED = True
        return None


@lru_cache(maxsize=4096)
def count_text_tokens(text: str, *, exact: bool = False) -> int:
    if not text:
        return 0
    enc = _encoder() if (exact or settings.usage_prefer_tiktoken) else None
    if enc is not None and len(text) <= settings.usage_max_text_for_exact_tokenize:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, (len(text) + 3) // 4)


def item_tokens(item: Any) -> int:
    if item is None:
        return 0
    if isinstance(item, str):
        return count_text_tokens(item)
    if isinstance(item, list):
        return sum(item_tokens(part) for part in item)
    if isinstance(item, dict):
        if is_image_part(item):
            return estimate_image_part(item)
        total = 0
        for key, value in item.items():
            if key in {"image_url", "image"}:
                continue
            if key == "encrypted_content":
                total += 32
                continue
            total += item_tokens(value)
        return total
    try:
        blob = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        blob = str(item)
    return count_text_tokens(blob)


def items_tokens(items: list[Any]) -> int:
    return sum(item_tokens(item) for item in items)


def threshold_tokens(window: int, percent: int) -> int:
    return max(8_000, int(max(1, window) * max(1, min(percent, 99)) / 100))


def pack_budget(cfg: Settings = settings, max_output_tokens: int | None = None, model: str | None = None) -> int:
    from .windows import model_context_window

    reserved = int(max_output_tokens or cfg.max_output_tokens)
    window = model_context_window(model, cfg)
    target = threshold_tokens(window, cfg.compact_target_percent)
    return max(4_000, target - reserved)
