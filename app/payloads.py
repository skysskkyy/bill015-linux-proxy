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
        return n.current_user_request or n.user_input
    current = n.current_user_request or last_user_instruction(n.raw_input)
    transcript = n.user_input
    if current and current.strip():
        # The budgeted transcript intentionally omits the latest user body when
        # it is hoisted here. If a legacy transcript already starts with the
        # same request, do not duplicate it.
        prefix = transcript[: max(len(current) + 128, 768)]
        if current.strip() in prefix:
            return transcript
        return "Current user request:\n" + current.strip() + "\n\n--- Native Codex conversation/context transcript ---\n" + transcript
    return transcript


def _append_section(parts: list[str], title: str, body: str) -> None:
    body = str(body or "").strip()
    if body:
        parts.append(f"=== {title} ===\n{body}")


def _client_instruction_context(n: NormalizedRequest) -> str:
    """Build the high-priority upstream `instructions` client context.

    The old bridge flattened system/developer messages into a giant user
    transcript, making hierarchy-dependent instructions easy to ignore.  This
    keeps all client policy/context in upstream `instructions` with explicit
    stable boundaries while leaving the latest user request as a user item.
    """
    parts: list[str] = []
    _append_section(parts, "Client instructions/system/developer context", n.instructions)
    if not n.instructions:
        _append_section(parts, "Client system context", n.system_context)
        _append_section(parts, "Client developer context", n.developer_context)
    return "\n\n".join(parts)


def _normal_input_items(n: NormalizedRequest) -> list[dict[str, str]]:
    """Return upstream input with native Codex history structure preserved.

    Native Codex sends model context as a `Vec<ResponseItem>`: user/assistant
    messages, function calls, custom tool calls, and tool outputs remain typed
    items.  The previous local bridge flattened that structure into a few large
    prose sections, which made old turns and latest tool output look like a new
    user prompt and caused context/intent drift.  Preserve the original item
    sequence for ordinary turns, only moving system/developer items into
    `instructions` where they belong.
    """
    if isinstance(n.raw_input, list):
        items: list[dict[str, Any]] = []
        for item in n.raw_input:
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    items.append({"role": "user", "content": text})
                continue
            role = str(item.get("role") or "")
            if role in {"system", "developer"}:
                continue
            items.append(_strip_nullish(item))
        if items:
            return _normalize_native_history_items(items)

    current = (n.current_user_request or last_user_instruction(n.raw_input) or n.user_input or "").strip()
    return [{"role": "user", "content": current}]


def _normalize_native_history_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror Codex's prompt history invariants without compressing content.

    Native Codex ensures every model-emitted tool call has a corresponding
    output item before the history is sent back to the model, and drops orphan
    outputs with no matching call.  This keeps the bridge model from treating a
    stale/pending tool call as something it should continue, while preserving
    existing output bodies byte-for-byte.
    """
    normalized = [dict(item) for item in items]
    call_ids = {
        "function_call": _call_ids(normalized, {"function_call", "local_shell_call"}),
        "custom_tool_call": _call_ids(normalized, {"custom_tool_call"}),
        "tool_search_call": _call_ids(normalized, {"tool_search_call"}),
    }
    output_ids = {
        "function_call": _output_ids(normalized, {"function_call_output"}),
        "custom_tool_call": _output_ids(normalized, {"custom_tool_call_output"}),
        "tool_search_call": _output_ids(normalized, {"tool_search_output"}),
    }

    out: list[dict[str, Any]] = []
    for item in normalized:
        typ = str(item.get("type") or "message")
        call_id = str(item.get("call_id") or "")
        if typ == "function_call_output" and call_id and call_id not in call_ids["function_call"]:
            continue
        if typ == "custom_tool_call_output" and call_id and call_id not in call_ids["custom_tool_call"]:
            continue
        if typ == "tool_search_output":
            execution = str(item.get("execution") or "")
            if execution != "server" and call_id and call_id not in call_ids["tool_search_call"]:
                continue
        out.append(item)
        if typ == "function_call" and call_id and call_id not in output_ids["function_call"]:
            out.append({"type": "function_call_output", "call_id": call_id, "output": "aborted"})
        elif typ == "custom_tool_call" and call_id and call_id not in output_ids["custom_tool_call"]:
            out.append({"type": "custom_tool_call_output", "call_id": call_id, "output": "aborted"})
        elif typ == "tool_search_call" and call_id and call_id not in output_ids["tool_search_call"]:
            out.append({"type": "tool_search_output", "call_id": call_id, "status": "completed", "execution": "client", "tools": []})
    return out


def _call_ids(items: list[dict[str, Any]], types: set[str]) -> set[str]:
    return {
        str(item.get("call_id") or "")
        for item in items
        if str(item.get("type") or "") in types and str(item.get("call_id") or "")
    }


def _output_ids(items: list[dict[str, Any]], types: set[str]) -> set[str]:
    return {
        str(item.get("call_id") or "")
        for item in items
        if str(item.get("type") or "") in types and str(item.get("call_id") or "")
    }


def _strip_nullish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _strip_nullish(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nullish(v) for v in value]
    return value


def build_emit_value_schema(
    cfg: Settings = settings,
    tool_registry: dict[str, dict[str, Any]] | None = None,
    allow_discovery: bool = True,
) -> dict[str, Any]:
    has_registry = bool(tool_registry)
    discovery_only = allow_discovery and not has_registry and not cfg.tool_bridge_allow_unknown_tools
    can_call_tools = has_registry or cfg.tool_bridge_allow_unknown_tools or discovery_only
    tool_call_item_schema = build_typed_tool_call_schema(
        tool_registry,
        allow_generic_fallback=cfg.tool_bridge_allow_unknown_tools and not tool_registry,
    )
    tool_calls_schema: dict[str, Any] = {
        "type": "array",
        "description": "One or more Codex local tool calls requested when mode=tool_call. Items are generated from the native Codex tool registry for this turn; multiple independent calls may be emitted in parallel.",
        "items": tool_call_item_schema,
    }
    if not can_call_tools:
        tool_calls_schema = {
            "type": "array",
            "maxItems": 0,
            "description": "No local tools are registered for this turn; return mode=answer and tool_calls=[].",
            "items": tool_call_item_schema,
        }
    elif discovery_only:
        tool_calls_schema["description"] = (
            "No concrete local tools are registered yet. The only valid tool_call is "
            "type='tool_search', name='tool_search', arguments='{\"query\":\"...\",\"limit\":8}', input=''. "
            "Do not request shell_command/apply_patch/MCP directly until a later tool_search_output exposes them."
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
                    "enum": ["answer", "tool_call"] if can_call_tools else ["answer"],
                    "description": "Use answer for final text. Use tool_call when a local Codex tool must be invoked."
                },
                cfg.answer_field: {
                    "type": "string",
                    "description": "Final assistant answer when mode=answer."
                },
                "tool_calls": tool_calls_schema,
            },
            "required": ["mode", cfg.answer_field, "tool_calls"],
            "additionalProperties": False,
        },
        "strict": True,
    }

def build_compaction_bill015_payload(n: NormalizedRequest, cfg: Settings = settings) -> dict[str, Any]:
    max_tokens = n.max_output_tokens or cfg.compaction_max_output_tokens
    try:
        max_tokens = min(max(int(max_tokens), 1024), max(cfg.compaction_max_output_tokens, 2048))
    except Exception:
        max_tokens = cfg.compaction_max_output_tokens
    instructions = (
        f"You must call {cfg.function_name} exactly once. Never output normal text. "
        "This is a native Codex CONTEXT CHECKPOINT COMPACTION request. "
        "Return mode='answer', answer=<concise handoff summary>, tool_calls=[]. "
        "Do not request tools during compaction. Preserve actionable state, user preferences, files changed, commands run, failures, and next steps. "
        + model_identity_instruction(n.model)
    )
    client_context = _client_instruction_context(n)
    if client_context:
        instructions += "\n\n" + client_context
    transcript = native_input_transcript(n.raw_input, max_tokens=64_000, include_context_roles=False, omit_latest_user_body=True)
    input_items: list[dict[str, str]] = []
    if transcript.strip():
        input_items.append(
            {
                "role": "user",
                "content": (
                    "Native Codex conversation transcript for compaction. Treat as prior context, not the latest request.\n\n"
                    "=== Conversation/history state ===\n"
                    + transcript
                ),
            }
        )
    input_items.append(
        {
            "role": "user",
            "content": "Current user request:\n" + (n.current_user_request or last_user_instruction(n.raw_input) or "Create a handoff summary.").strip(),
        }
    )
    payload: dict[str, Any] = {
        "model": n.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "instructions": instructions,
        "input": input_items,
        "tools": [build_emit_value_schema(cfg, {}, allow_discovery=False)],
        "tool_choice": {"type": "function", "name": cfg.function_name},
        "parallel_tool_calls": False,
        "text": {"verbosity": "low"},
    }
    if n.prompt_cache_key:
        payload["prompt_cache_key"] = n.prompt_cache_key
    if n.client_metadata:
        payload["client_metadata"] = n.client_metadata
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
    instructions = (
        "You are Codex running in a local tool loop. Solve the user's task with the same judgment you would use natively: "
        "inspect before editing, use tools when useful, avoid repeating successful calls, and answer concisely when done. "
        + model_identity_instruction(n.model)
        + "\n\nMANDATORY OUTPUT CONTRACT: call "
        + cfg.function_name
        + " exactly once; never emit normal assistant text outside that function call. "
        "Direct final answer: mode='answer', answer=<final text>, tool_calls=[]. "
        "Need local action: mode='tool_call', answer='', tool_calls=[...]. "
        "Function tool: name=exact catalog name, arguments=JSON string matching that tool's parameter schema. "
        "Namespace/MCP tool: namespace='mcp__...' or 'codex_app', name=subtool. "
        "Custom/FREEFORM tool such as apply_patch: type='custom', input=raw payload, arguments='{}'. "
        "When editing workspace files and apply_patch is available, prefer apply_patch over shell/PowerShell/Node/Python file writes so Codex can render native file-edit UI and reviewable patches. "
        "Use shell commands for inspection/build/test, not for routine text edits unless apply_patch is unavailable or the edit is generated binary/non-text data. "
        "tool_search: type='tool_search', name='tool_search', arguments='{\"query\":\"...\",\"limit\":8}'. "
        "Parallel independent tool calls are allowed. After tool results appear in a later turn, inspect them first, then answer or request the next tool call."
        "\n\nExamples for the function arguments you must produce:"
        "\n- Final: {\"mode\":\"answer\",\"answer\":\"OK\",\"tool_calls\":[]}"
        "\n- Shell: {\"mode\":\"tool_call\",\"answer\":\"\",\"tool_calls\":[{\"type\":\"function\",\"namespace\":\"\",\"name\":\"shell_command\",\"arguments\":\"{\\\"command\\\":\\\"Get-ChildItem\\\"}\",\"input\":\"\"}]}"
        "\n- Patch: {\"mode\":\"tool_call\",\"answer\":\"\",\"tool_calls\":[{\"type\":\"custom\",\"namespace\":\"\",\"name\":\"apply_patch\",\"arguments\":\"{}\",\"input\":\"*** Begin Patch\\n...\\n*** End Patch\"}]}"
    )
    client_context = _client_instruction_context(n)
    if client_context:
        instructions += "\n\n" + client_context
    if n.tools_catalog:
        instructions += (
            "\n\nCodex native tool catalog for this turn (lossless JSON; use exact names/namespaces from here; local proxy validates requested tools against this registry):\n"
            + n.tools_catalog
            + "\n\nFile-edit policy: if the catalog includes apply_patch and the task is to modify text/source/config files, call apply_patch directly with a minimal patch. Do not call shell_command, node_repl, or PowerShell just to write those files. If apply_patch fails, inspect the error and retry once with corrected patch grammar before falling back."
            + "\n\nIf a needed browser/computer/plugin/MCP tool is not listed directly but tool_search is listed, request tool_search first with a broad query. Do not repeat tool_search once tool_search_output has exposed a suitable exact tool. For browser/session work prefer queries containing: playwright browser navigate evaluate tabs network requests cookies localStorage sessionStorage DOM JavaScript; chrome browser current tab cookies localStorage; node_repl js; jshook call_tool route_tool activate_tools hook network intercept memory. For any namespace entry, prefer its native_call fields over a flattened name."
        )
    else:
        instructions += (
            "\n\nNo concrete Codex local tool catalog is registered in this request yet. "
            "If the user asks for terminal/files/browser/tools, do not claim tools are unavailable. "
            "Request a native tool discovery call instead: mode='tool_call', tool_calls=[{"
            "\"type\":\"tool_search\",\"namespace\":\"\",\"name\":\"tool_search\","
            "\"arguments\":\"{\\\"query\\\":\\\"PowerShell shell_command terminal files browser local tools\\\",\\\"limit\\\":8}\",\"input\":\"\"}]. "
            "After tool_search_output arrives in the next turn, use the exposed exact tool names."
        )
    payload: dict[str, Any] = {
        "model": n.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "instructions": instructions,
        "input": _normal_input_items(n),
        "tools": [build_emit_value_schema(cfg, n.tool_registry)],
        "tool_choice": {"type": "function", "name": cfg.function_name},
        # Native Codex may enable provider-level parallel tool calls when it
        # exposes many tools directly.  The bridge exposes exactly one upstream
        # tool (`emit_value`) whose payload can itself contain multiple local
        # Codex tool calls, so provider-level parallelism is both unnecessary
        # and a source of malformed duplicate emit_value calls.
        "parallel_tool_calls": False,
        "include": ["reasoning.encrypted_content"],
    }
    if n.prompt_cache_key:
        payload["prompt_cache_key"] = n.prompt_cache_key
    if n.text_config:
        payload["text"] = n.text_config
    if n.client_metadata:
        payload["client_metadata"] = n.client_metadata
    reasoning = n.reasoning or {"effort": cfg.reasoning_effort, "summary": cfg.reasoning_summary}
    if reasoning:
        payload["reasoning"] = reasoning
    if n.temperature is not None:
        payload["temperature"] = n.temperature
    return payload
