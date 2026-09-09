from __future__ import annotations

from typing import Any

from ..config import Settings
from ..protocol.models import Catalog, ToolSpec

_DEFAULT_SEARCH_ARGUMENTS = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["query"],
    "additionalProperties": False,
}


def _as_object_schema(parameters: dict[str, Any] | None) -> dict[str, Any]:
    params = parameters if isinstance(parameters, dict) else {}
    if not params:
        return {"type": "object", "additionalProperties": True}
    if params.get("type") == "object" or "properties" in params or "required" in params:
        out = dict(params)
        out.setdefault("type", "object")
        return out
    return {"type": "object", "properties": params, "additionalProperties": True}


def _variant(spec: ToolSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "type": {"type": "string", "enum": [spec.call_type]},
        "name": {"type": "string", "enum": [spec.name]},
        "namespace": {"type": "string", "enum": [spec.namespace or ""]},
    }
    if spec.call_type == "custom":
        properties["input"] = {"type": "string"}
        if spec.parameters:
            properties["arguments"] = _as_object_schema(spec.parameters)
        required = ["type", "name", "input"]
    elif spec.call_type == "tool_search":
        properties["arguments"] = _as_object_schema(spec.parameters) if spec.parameters else dict(_DEFAULT_SEARCH_ARGUMENTS)
        required = ["type", "name", "arguments"]
    else:
        properties["arguments"] = _as_object_schema(spec.parameters)
        required = ["type", "name", "arguments"]
    variant: dict[str, Any] = {
        "type": "object",
        "description": spec.description or spec.name,
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if spec.examples not in (None, "", []):
        variant["examples"] = spec.examples if isinstance(spec.examples, list) else [spec.examples]
    if spec.format:
        variant["x-format"] = spec.format
    return variant


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
            "mode=answer with tool_calls=[] only for the finished user-facing reply. "
            "mode=tool_call: non-empty tool_calls; answer may be empty. "
            "If a short status note is included, it must be in the same call as the tools."
        ),
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["answer", "tool_call"]},
                cfg.answer_field: {"type": "string"},
                "tool_calls": {"type": "array", "items": {"oneOf": variants} if len(variants) > 1 else variants[0]},
            },
            "required": ["mode", "tool_calls"],
            "additionalProperties": False,
        },
    }
