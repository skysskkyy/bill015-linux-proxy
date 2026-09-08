from __future__ import annotations

from typing import Any

from ..config import Settings, settings
from ..context.pack import pack_turn_items
from ..protocol.models import Turn
from .prompt import build_instructions
from .schema import build_emit_value_schema


def _gpt56(model: str) -> bool:
    return "gpt-5.6" in (model or "").lower() or model.lower().endswith("-sol")


def _reasoning(turn: Turn, cfg: Settings) -> dict[str, Any] | None:
    if turn.reasoning:
        return dict(turn.reasoning)
    if _gpt56(turn.model) or cfg.reasoning_effort:
        out: dict[str, Any] = {}
        if cfg.reasoning_effort:
            out["effort"] = cfg.reasoning_effort
        elif _gpt56(turn.model):
            out["effort"] = "high"
        if cfg.reasoning_summary:
            out["summary"] = cfg.reasoning_summary
        return out or None
    return None


def build_upstream_payload(turn: Turn, cfg: Settings = settings, *, packed_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    packed = packed_items
    if packed is None:
        packed = pack_turn_items(turn, cfg).items
    max_tokens = turn.max_output_tokens or (cfg.compaction_max_output_tokens if turn.is_compaction else cfg.max_output_tokens)
    try:
        max_tokens = min(int(max_tokens), cfg.max_output_tokens if not turn.is_compaction else cfg.compaction_max_output_tokens)
    except Exception:
        max_tokens = cfg.max_output_tokens
    payload: dict[str, Any] = {
        "model": turn.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "instructions": build_instructions(turn, cfg),
        "input": packed,
        "tools": [build_emit_value_schema(cfg, turn.catalog)],
        "tool_choice": {"type": "function", "name": cfg.function_name},
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
    }
    reasoning = _reasoning(turn, cfg)
    if reasoning:
        payload["reasoning"] = reasoning
    if turn.temperature is not None:
        payload["temperature"] = turn.temperature
    if turn.client_metadata:
        payload["client_metadata"] = turn.client_metadata
    if turn.metadata:
        payload["metadata"] = turn.metadata
    if turn.previous_response_id:
        payload["previous_response_id"] = turn.previous_response_id
    if turn.prompt_cache_key:
        payload["prompt_cache_key"] = turn.prompt_cache_key
    return payload
