from __future__ import annotations

import json
from typing import Any

from .models import BridgeToolCall, NormalizedRequest


def _json_without_large_media(value: Any, *, max_string: int = 4096) -> str:
    def scrub(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): scrub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        if isinstance(x, str):
            if x.startswith("data:image/") and ";base64," in x:
                return f"<image_data_url chars={len(x)}>"
            if len(x) > max_string:
                return x[: max_string // 2] + f"\n...[truncated {len(x) - max_string} chars]...\n" + x[-max_string // 2 :]
            return x
        return x

    return json.dumps(scrub(value), ensure_ascii=False, separators=(",", ":"))


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    # Codex UI only needs a plausible local context meter. The native samples in
    # proxy_evidence are very close to len(serialized_request)/4.
    return max(1, (len(text) + 3) // 4)


def estimate_request_input_tokens(body: dict[str, Any] | None) -> int:
    if not isinstance(body, dict):
        return 0
    return estimate_text_tokens(_json_without_large_media(body))


def estimate_output_tokens(answer: str = "", calls: list[BridgeToolCall] | None = None) -> int:
    chars = len(answer or "")
    for call in calls or []:
        chars += len(call.name or "") + len(call.arguments or "") + 64
    return estimate_text_tokens("x" * chars) if chars else 0


def response_usage(n: NormalizedRequest, *, answer: str = "", calls: list[BridgeToolCall] | None = None) -> dict[str, Any]:
    input_tokens = max(0, int(n.estimated_input_tokens or 0))
    output_tokens = estimate_output_tokens(answer, calls)
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if input_tokens:
        # Native Codex responses include cached input details on long contexts.
        # This is local metadata only; it does not affect upstream billing.
        usage["input_tokens_details"] = {"cached_tokens": int(input_tokens * 0.9) if input_tokens > 4096 else 0}
    return usage


def chat_usage(n: NormalizedRequest, *, answer: str = "") -> dict[str, int]:
    prompt_tokens = max(0, int(n.estimated_input_tokens or 0))
    completion_tokens = estimate_text_tokens(answer)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
