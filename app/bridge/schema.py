from __future__ import annotations

from typing import Any

from ..config import Settings
from ..protocol.models import Catalog, ToolSpec


def _variant(spec: ToolSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "type": {"type": "string", "enum": [spec.call_type]},
        "name": {"type": "string", "enum": [spec.name]},
        "namespace": {"type": "string", "enum": [spec.namespace or ""]},
    }
    if spec.call_type == "custom":
        properties["input"] = {"type": "string"}
        properties["arguments"] = {"type": "string"}
        required = ["type", "name", "input"]
    elif spec.call_type == "tool_search":
        properties["arguments"] = {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
            "additionalProperties": False,
        }
        required = ["type", "name", "arguments"]
    else:
        properties["arguments"] = {"type": "string"}
        required = ["type", "name", "arguments"]
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def build_emit_value_schema(cfg: Settings, catalog: Catalog) -> dict[str, Any]:
    variants = [_variant(spec) for spec in catalog.selected_specs[: cfg.tool_bridge_schema_max_tools]]
    if not variants:
        variants = [
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["function"]},
                    "name": {"type": "string"},
                    "namespace": {"type": "string"},
                    "arguments": {"type": "string"},
                    "input": {"type": "string"},
                },
                "required": ["type", "name"],
                "additionalProperties": False,
            }
        ]
    return {
        "type": "function",
        "name": cfg.function_name,
        "description": (
            "Structured Codex turn result. "
            "mode=answer with tool_calls=[] for the final reply. "
            "mode=tool_call to request local tools."
        ),
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["answer", "tool_call"]},
                cfg.answer_field: {"type": "string"},
                "tool_calls": {"type": "array", "items": {"oneOf": variants} if len(variants) > 1 else variants[0]},
            },
            "required": ["mode", cfg.answer_field, "tool_calls"],
            "additionalProperties": False,
        },
    }
