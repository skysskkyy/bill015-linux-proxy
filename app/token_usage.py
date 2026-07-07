from __future__ import annotations

from typing import Any

from .models import BridgeToolCall, NormalizedRequest
from .usage_estimator import (
    build_chat_usage,
    build_response_usage,
    estimate_request_input_tokens,
    estimate_response_output_tokens,
    estimate_text_tokens,
)


def estimate_output_tokens(answer: str = "", calls: list[BridgeToolCall] | None = None) -> int:
    return estimate_response_output_tokens(answer, calls)


def response_usage(n: NormalizedRequest, *, answer: str = "", calls: list[BridgeToolCall] | None = None) -> dict[str, Any]:
    return build_response_usage(n, answer=answer, calls=calls)


def chat_usage(n: NormalizedRequest, *, answer: str = "") -> dict[str, Any]:
    return build_chat_usage(n, answer=answer)
