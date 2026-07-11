from __future__ import annotations

from typing import Any

from .config import Settings, settings
from .models import NormalizedRequest
from .normalization import last_user_instruction, model_identity_instruction, native_input_transcript
from .tool_bridge import build_typed_tool_call_schema


def _focused_user_content(n: NormalizedRequest) -> str:
    """Put the active user request before bulky replayed context.

    In exploit mode the upstream model already has to follow the emit_value
    bridge contract.  If we also start the user message with a long native
    transcript, the actual current task can be buried behind system/developer
    notes and old tool output.  Keep the full transcript for state, but make the
    current request the first thing the model sees in the user turn.
    """
    if not isinstance(n.raw_input, list):
        return n.user_input
    current = last_user_instruction(n.raw_input)
    transcript = n.user_input
    if current and current.strip() and current.strip() not in transcript[: max(len(current) + 64, 512)]:
        return "Current user request:\n" + current.strip() + "\n\n--- Native Codex conversation/context transcript ---\n" + transcript
    if current and current.strip():
        return "Current user request:\n" + current.strip() + "\n\n--- Native Codex conversation/context transcript ---\n" + transcript
    return transcript


def build_emit_value_schema(
    cfg: Settings = settings,
    tool_registry: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tool_call_item_schema = build_typed_tool_call_schema(
        tool_registry,
        allow_generic_fallback=cfg.tool_bridge_allow_unknown_tools and not tool_registry,
    )
    return {
        "type": "function",
        "name": cfg.function_name,
        "description": "Return either a final assistant answer or strongly typed Codex-compatible local tool call request.",
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
                    "description": "One or more Codex local tool calls requested when mode=tool_call. Items are generated from the native Codex tool registry for this turn; multiple independent calls may be emitted in parallel.",
                    "items": tool_call_item_schema,
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
        "tools": [build_emit_value_schema(cfg, {})],
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
        "You are Codex running in a local tool loop. Solve the user's task with the same judgment you would use natively: "
        "inspect before editing, use tools when useful, avoid repeating successful calls, and answer concisely when done. "
        + model_identity_instruction(n.model)
        + "\n\nMANDATORY OUTPUT CONTRACT: call "
        + cfg.function_name
        + " exactly once; never emit normal assistant text outside that function call. "
        "Direct final answer: mode='answer', answer=<final text>, tool_calls=[]. "
        "Need local action: mode='tool_call', answer='', tool_calls=[...]. "
        "Function tool: name=exact catalog name, arguments=object matching the embedded typed schema. "
        "Namespace/MCP tool: namespace='mcp__...' or 'codex_app', name=subtool. "
        "Custom/FREEFORM tool such as apply_patch: type='custom', input=raw payload, arguments={}. "
        "tool_search: type='tool_search', name='tool_search', arguments={\"query\":\"...\",\"limit\":8}. "
        "Parallel independent tool calls are allowed. After tool results appear in a later turn, inspect them first, then answer or request the next tool call."
        "\n\nExamples for the function arguments you must produce:"
        "\n- Final: {\"mode\":\"answer\",\"answer\":\"OK\",\"tool_calls\":[]}"
        "\n- Shell: {\"mode\":\"tool_call\",\"answer\":\"\",\"tool_calls\":[{\"type\":\"function\",\"namespace\":\"\",\"name\":\"shell_command\",\"arguments\":{\"command\":\"Get-ChildItem\"},\"input\":\"\"}]}"
        "\n- Patch: {\"mode\":\"tool_call\",\"answer\":\"\",\"tool_calls\":[{\"type\":\"custom\",\"namespace\":\"\",\"name\":\"apply_patch\",\"arguments\":{},\"input\":\"*** Begin Patch\\n...\\n*** End Patch\"}]}"
    )
    if n.instructions:
        system += "\n\nClient instructions:\n" + n.instructions
    if n.tools_catalog:
        system += (
            "\n\nCodex native tool catalog for this turn (lossless JSON; the emit_value schema also embeds typed oneOf variants for these tools):\n"
            + n.tools_catalog
            + "\n\nIf a needed browser/computer/plugin/MCP tool is not listed directly but tool_search is listed, request tool_search first with a broad query. For browser/session work prefer queries containing: playwright browser navigate evaluate tabs network requests cookies localStorage sessionStorage DOM JavaScript; chrome browser current tab cookies localStorage; node_repl js; jshook call_tool route_tool activate_tools hook network intercept memory. The local proxy may add extra broad tool_search calls to expose deferred native Codex tools. For any namespace entry, prefer its native_call fields over a flattened name."
        )
    user_content = _focused_user_content(n)
    if n.latest_tool_summary:
        user_content = n.latest_tool_summary + "\n\n--- Current request and native conversation/context transcript ---\n" + user_content
    payload: dict[str, Any] = {
        "model": n.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "tools": [build_emit_value_schema(cfg, n.tool_registry)],
        "tool_choice": {"type": "function", "name": cfg.function_name},
    }
    reasoning = n.reasoning or {"effort": cfg.reasoning_effort, "summary": cfg.reasoning_summary}
    if reasoning:
        payload["reasoning"] = reasoning
    if n.temperature is not None:
        payload["temperature"] = n.temperature
    return payload
