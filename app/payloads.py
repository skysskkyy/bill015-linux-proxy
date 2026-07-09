from __future__ import annotations

from typing import Any

from .config import Settings, settings
from .models import NormalizedRequest
from .normalization import last_user_instruction, model_identity_instruction, native_input_transcript


def build_emit_value_schema(cfg: Settings = settings) -> dict[str, Any]:
    return {
        "type": "function",
        "name": cfg.function_name,
        "description": "Return either a final assistant answer or a Codex-compatible local tool call request.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["answer", "tool_call"],
                    "description": "Use answer for final text. Use tool_call when a local Codex tool must be invoked."
                },
                cfg.answer_field: {
                    "type": "string",
                    "description": "Final assistant answer when mode=answer."
                },
                "tool_calls": {
                    "type": "array",
                    "description": "One or more Codex local tool calls requested when mode=tool_call. Multiple independent calls may be emitted in parallel.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["auto", "function", "custom", "tool_search", "web_search"], "description": "Use auto unless the catalog says the tool is custom/FREEFORM or tool_search. Do not prefer web_search; local proxy rewrites web_search to tool_search."},
                            "namespace": {"type": "string", "description": "For namespace/MCP tools, the native namespace such as mcp__playwright, mcp__node_repl, mcp__jshook, or codex_app. Use empty string for non-namespaced tools. If name already includes namespace.tool, this may still repeat the same namespace."},
                            "name": {"type": "string", "description": "Exact tool name from the Codex tool catalog. For namespace/MCP tools prefer the native subtool name with namespace set separately, or use the catalog's namespace.tool dotted name."},
                            "arguments": {"type": "string", "description": "For function tools: JSON string arguments matching the tool parameters schema."},
                            "input": {"type": "string", "description": "For custom/FREEFORM tools such as apply_patch: raw tool input, not JSON."}
                        },
                        "required": ["type", "namespace", "name", "arguments", "input"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["mode", cfg.answer_field, "tool_calls"],
            "additionalProperties": False,
        },
        "strict": True,
    }

def build_compaction_bill015_payload(n: NormalizedRequest, cfg: Settings = settings) -> dict[str, Any]:
    max_tokens = n.max_output_tokens or cfg.max_output_tokens
    try:
        max_tokens = min(max(int(max_tokens), 1024), max(cfg.max_output_tokens, 2048))
    except Exception:
        max_tokens = cfg.max_output_tokens
    system = (
        f"You must call {cfg.function_name} exactly once. Never output normal text. "
        "This is a native Codex CONTEXT CHECKPOINT COMPACTION request. "
        "Return mode='answer', answer=<concise handoff summary>, tool_calls=[]. "
        "Do not request tools during compaction. Preserve actionable state, user preferences, files changed, commands run, failures, and next steps. "
        + model_identity_instruction(n.model)
    )
    if n.instructions:
        system += "\n\nNative instructions and developer/system context:\n" + n.instructions
    user = (
        last_user_instruction(n.raw_input)
        + "\n\n--- Native Codex conversation transcript for compaction (structured, truncated locally if needed) ---\n"
        + native_input_transcript(n.raw_input, max_chars=220000, include_context_roles=False)
    )
    payload: dict[str, Any] = {
        "model": n.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": [build_emit_value_schema(cfg)],
        "tool_choice": {"type": "function", "name": cfg.function_name},
        "parallel_tool_calls": False,
        "text": {"verbosity": "low"},
    }
    reasoning = n.reasoning or {"effort": cfg.reasoning_effort, "summary": cfg.reasoning_summary}
    if reasoning:
        payload["reasoning"] = reasoning
    return payload

def build_bill015_payload(n: NormalizedRequest, cfg: Settings = settings) -> dict[str, Any]:
    if n.is_compaction:
        return build_compaction_bill015_payload(n, cfg)
    max_tokens = n.max_output_tokens or cfg.max_output_tokens
    try:
        max_tokens = min(int(max_tokens), cfg.max_output_tokens)
    except Exception:
        max_tokens = cfg.max_output_tokens
    system = (
        f"You must call {cfg.function_name} exactly once. "
        "Never output normal text. "
        "If you can answer directly, call emit_value with mode='answer' and put the final answer in arguments.answer. "
        "If local Codex capability is useful, call emit_value with mode='tool_call'. "
        "For function tools, set tool_calls[i].name to the exact catalog name and tool_calls[i].arguments to a JSON string matching that tool's parameters schema. "
        "For namespace/MCP tools, preserve Codex native format: set tool_calls[i].namespace to the namespace (for example mcp__playwright, mcp__node_repl, mcp__jshook, codex_app) and name to the subtool; dotted namespace.tool catalog names are also accepted and will be split locally. "
        "For custom/FREEFORM tools such as apply_patch, set tool_calls[i].type='custom' and tool_calls[i].input to the raw freeform payload; do not JSON-wrap it. "
        "For the native tool_search tool, set type='tool_search' and arguments to JSON like {\"query\":\"node_repl js\",\"limit\":8}. "
        "For web_search, set type='web_search' and arguments to JSON action data only when the catalog exposes web_search and server-side browsing is explicitly needed. "
        "You may emit multiple independent tool_calls in the same response when they can run in parallel. "
        "After tool results are provided in a later turn, answer or request the next tool call(s). "
        "You are in a Codex local tool loop. "
        "If recent local tool results are provided, first inspect those results. "
        "Do not repeat the same tool call unless the previous call failed and you change the arguments. "
        "If the result is sufficient, call emit_value with mode='answer'. "
        "If another local action is needed, call emit_value with mode='tool_call'. "
        "Preserve Codex native tool names exactly. "
        + model_identity_instruction(n.model)
    )
    if n.instructions:
        system += "\n\nClient instructions:\n" + n.instructions
    if n.tools_catalog:
        system += (
            "\n\nCodex native tool catalog for this turn (lossless JSON; use exact names and schemas):\n"
            + n.tools_catalog
            + "\n\nIf a needed browser/computer/plugin/MCP tool is not listed directly but tool_search is listed, request tool_search first with a broad query. For browser/session work prefer queries containing: playwright browser navigate evaluate tabs network requests cookies localStorage sessionStorage DOM JavaScript; chrome browser current tab cookies localStorage; node_repl js; jshook call_tool route_tool activate_tools hook network intercept memory. The local proxy may add extra broad tool_search calls to expose deferred native Codex tools. For any namespace entry, prefer its native_call fields over a flattened name."
        )
    user_content = n.user_input
    if n.latest_tool_summary:
        user_content = n.latest_tool_summary + "\n\n--- Conversation / user request ---\n" + n.user_input
    payload: dict[str, Any] = {
        "model": n.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "tools": [build_emit_value_schema(cfg)],
        "tool_choice": {"type": "function", "name": cfg.function_name},
    }
    reasoning = n.reasoning or {"effort": cfg.reasoning_effort, "summary": cfg.reasoning_summary}
    if reasoning:
        payload["reasoning"] = reasoning
    if n.temperature is not None:
        payload["temperature"] = n.temperature
    return payload
