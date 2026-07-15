from __future__ import annotations

import hashlib
import re
from typing import Any

from .config import Settings, settings
from .models import NormalizedRequest
from .normalization import collect_deferred_tools_from_input, last_user_instruction, native_input_transcript
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

    Match native Codex's split: top-level base instructions belong in
    `instructions`; developer/context message items stay in `input[]` for normal
    turns. Compaction requests may still pass extracted context here because the
    compact endpoint returns a replacement history.
    """
    parts: list[str] = []
    _append_section(parts, "Client instructions/system/developer context", n.instructions)
    return "\n\n".join(parts)


def _normal_input_items(n: NormalizedRequest) -> list[dict[str, str]]:
    """Return upstream input with native Codex history structure preserved.

    Native Codex sends model context as a `Vec<ResponseItem>`: user/assistant
    messages, function calls, custom tool calls, and tool outputs remain typed
    items.  The previous local bridge flattened that structure into a few large
    prose sections, which made old turns and latest tool output look like a new
    user prompt and caused context/intent drift.  Preserve the original item
    sequence for ordinary turns, including developer/context messages. Native
    Codex does not flatten developer messages into a budgeted prose summary for
    normal turns; doing so loses AGENTS/skills context before Codex's own
    compaction has a chance to run.
    """
    if isinstance(n.raw_input, list):
        items: list[dict[str, Any]] = []
        for item in n.raw_input:
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    items.append({"role": "user", "content": text})
                continue
            if str(item.get("type") or "") in {"additional_tools", "additionalTools"}:
                # Codex Responses Lite carries the tool definitions as a
                # developer input item.  The bridge extracts those into its
                # catalog/registry in normalization, then exposes exactly one
                # upstream tool (emit_value).  Replaying the raw additional
                # tools item here is duplicate model context and can make the
                # upstream think it should call native tools directly instead
                # of using the bridge contract.
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
        item = _normalize_upstream_item_id(item, typ)
        if typ == "message" and _is_local_proxy_noise_message(item):
            continue
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


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _normalize_upstream_item_id(item: dict[str, Any], typ: str) -> dict[str, Any]:
    """Fix historical tool-call item ids before replaying them upstream.

    Native Codex history can contain local item ids such as ``item_*`` on
    ``function_call`` entries. The upstream Responses API rejects those for
    function-call input items and requires an ``fc_*`` id. Tool outputs link by
    ``call_id``, so rewriting the item id is safe and keeps the transcript
    replayable.
    """
    required_prefixes = {
        "function_call": "fc_",
        "local_shell_call": "fc_",
    }
    prefix = required_prefixes.get(typ)
    if not prefix:
        return item
    current = str(item.get("id") or "")
    if current.startswith(prefix):
        return item
    basis = str(item.get("call_id") or current or item.get("name") or typ)
    safe = _SAFE_ID_RE.sub("_", basis).strip("_")
    if not safe:
        safe = hashlib.sha256(repr(sorted(item.items())).encode("utf-8", errors="replace")).hexdigest()[:24]
    fixed = dict(item)
    fixed["id"] = prefix + safe.removeprefix(prefix)[:64]
    return fixed


def _is_local_proxy_noise_message(item: dict[str, Any]) -> bool:
    role = str(item.get("role") or "")
    if role != "assistant":
        return False
    text = flatten_message_text(item.get("content", ""))
    noise = (
        "I need to resolve the right local tool first.",
        "[local proxy] tool_call mode requested but no valid tool_calls were provided.",
        "[local proxy loop guard]",
    )
    return any(marker in text for marker in noise)


def flatten_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                value = part.get("text") or part.get("output_text") or part.get("content")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if isinstance(content, dict):
        value = content.get("text") or content.get("output_text") or content.get("content")
        return value if isinstance(value, str) else ""
    return ""


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


def use_responses_lite_upstream(n: NormalizedRequest, cfg: Settings = settings) -> bool:
    """Return whether the bridge should emit the Codex Responses Lite wire shape.

    ``n.responses_lite`` describes the *client* request shape.  In strict-zero
    BILL-015 mode we still ingest that shape so GPT-5.6 Codex CLI/Desktop tools
    are discovered, but we deliberately do **not** forward the Lite transport
    upstream.  The Lite transport/header was observed to move gpt-5.6-sol onto a
    provider path that accounts the accepted request even when the local proxy
    closes the stream at the tool-arguments boundary.  The billing-safe bridge
    keeps the exact same local tool catalog inside the single synthetic
    ``emit_value`` function and uses the standard Responses tool fields
    upstream, matching the older gpt-5.5 strict-zero behavior.
    """
    return bool(n.responses_lite and not cfg.strict_zero)


def _native_reasoning_param(n: NormalizedRequest, cfg: Settings = settings) -> dict[str, Any] | None:
    if isinstance(n.reasoning, dict):
        reasoning = dict(n.reasoning)
        if use_responses_lite_upstream(n, cfg):
            reasoning.setdefault("context", "all_turns")
        return reasoning or None
    reasoning: dict[str, Any] = {}
    if cfg.reasoning_effort:
        reasoning["effort"] = cfg.reasoning_effort
    if cfg.reasoning_summary:
        reasoning["summary"] = cfg.reasoning_summary
    # Codex core sets reasoning.context=all_turns for Responses Lite models so
    # reasoning survives the additional_tools-in-input request shape. Preserve
    # that model-visible contract when this proxy detects the same shape.
    if use_responses_lite_upstream(n, cfg):
        reasoning["context"] = "all_turns"
    return reasoning or None


def _append_request_controls(payload: dict[str, Any], n: NormalizedRequest) -> None:
    if n.prompt_cache_key:
        payload["prompt_cache_key"] = n.prompt_cache_key
    if n.prompt_cache_options:
        payload["prompt_cache_options"] = n.prompt_cache_options
    if n.service_tier and n.service_tier != "default":
        payload["service_tier"] = n.service_tier
    if n.truncation:
        payload["truncation"] = n.truncation
    if n.text_config:
        payload["text"] = n.text_config
    if n.client_metadata:
        payload["client_metadata"] = n.client_metadata


_RESPONSES_LITE_MARKER_KEYS = {
    "x-openai-internal-codex-responses-lite",
    "responses_lite",
    "use_responses_lite",
}


def _strip_responses_lite_markers(payload: dict[str, Any]) -> None:
    """Remove body metadata that would re-enable Responses Lite upstream."""
    for key in _RESPONSES_LITE_MARKER_KEYS:
        payload.pop(key, None)
    metadata = payload.get("client_metadata")
    if isinstance(metadata, dict):
        cleaned = {
            key: value
            for key, value in metadata.items()
            if str(key).strip().lower() not in _RESPONSES_LITE_MARKER_KEYS
        }
        if cleaned:
            payload["client_metadata"] = cleaned
        else:
            payload.pop("client_metadata", None)


def _apply_responses_lite_transport(payload: dict[str, Any], n: NormalizedRequest, cfg: Settings = settings) -> dict[str, Any]:
    """Move instructions/tools into input items exactly like Codex Lite.

    The client-provided ``additional_tools`` item is intentionally replaced by
    the bridge's actual upstream tools, so the model sees only tools it can
    really call on this upstream request.
    """
    if not use_responses_lite_upstream(n, cfg):
        _strip_responses_lite_markers(payload)
        return payload
    tools = payload.pop("tools", [])
    instructions = str(payload.pop("instructions", "") or "")
    current_input = payload.get("input")
    input_items = list(current_input) if isinstance(current_input, list) else []
    prefix: list[dict[str, Any]] = [
        {"type": "additional_tools", "role": "developer", "tools": tools}
    ]
    if instructions:
        prefix.append(
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": instructions}],
            }
        )
    payload["input"] = [*prefix, *input_items]
    payload["parallel_tool_calls"] = False
    return payload


def build_emit_value_schema(
    cfg: Settings = settings,
    tool_registry: dict[str, dict[str, Any]] | None = None,
    allow_discovery: bool = True,
    bridge_target: str = "desktop",
) -> dict[str, Any]:
    concrete_tool_registry = _concrete_tool_registry_for_bridge(tool_registry, bridge_target=bridge_target)
    has_registry = bool(concrete_tool_registry)
    # Codex TUI rejects dynamic/client-side discovery calls with:
    # "Dynamic tool calls are not available in TUI yet."  The bridge must only
    # emit concrete tools already present in the native registry.
    can_call_tools = has_registry or cfg.tool_bridge_allow_unknown_tools
    tool_call_item_schema = build_typed_tool_call_schema(
        concrete_tool_registry,
        allow_generic_fallback=cfg.tool_bridge_allow_unknown_tools and not tool_registry,
        # Native tool_search_call is handled by Codex Core on every surface.
        # Do not confuse it with app-server DynamicToolCall, which the TUI
        # currently rejects.
        include_dynamic_tools=True,
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
                    "description": "Final assistant answer when mode=answer. Optional brief progress/commentary text when mode=tool_call; leave empty only if no useful user-visible update."
                },
                "tool_calls": tool_calls_schema,
            },
            "required": ["mode", cfg.answer_field, "tool_calls"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def build_final_answer_tool_schema(cfg: Settings = settings) -> dict[str, Any]:
    return {
        "type": "function",
        "name": cfg.final_answer_tool_name,
        "description": (
            "Return the final user-facing answer. Use this only when the task is complete "
            "and no more local Codex tool action is needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                cfg.answer_field: {
                    "type": "string",
                    "description": "Final answer to show to the user.",
                }
            },
            "required": [cfg.answer_field],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _tool_identity(tool: dict[str, Any]) -> tuple[str, str]:
    return (str(tool.get("type") or ""), str(tool.get("name") or ""))


def _native_tools_for_upstream(n: NormalizedRequest, cfg: Settings) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(tool: Any) -> None:
        if not isinstance(tool, dict):
            return
        ident = _tool_identity(tool)
        if ident in seen:
            return
        seen.add(ident)
        tools.append(_strip_nullish(tool))

    if isinstance(n.raw_tools, list):
        for tool in n.raw_tools:
            add(tool)
    for tool in collect_deferred_tools_from_input(n.raw_input):
        add(tool)

    # Final answers are a function call too, so BILL-015 can still close the
    # upstream stream at response.function_call_arguments.done without waiting
    # for a normal assistant completion.
    add(build_final_answer_tool_schema(cfg))
    return tools


def _base_bridge_instructions() -> str:
    return (
        "You are Codex running in a local tool loop. Solve the user's task with the same judgment you would use natively: "
        "inspect before editing, use tools when useful, avoid repeating successful calls, and answer concisely when done. "
    )


def _append_common_context(instructions: str, n: NormalizedRequest) -> str:
    client_context = _client_instruction_context(n)
    if client_context:
        instructions += "\n\n" + client_context
    return instructions


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
        "tools": [build_emit_value_schema(cfg, {}, allow_discovery=False, bridge_target="tui")],
        "tool_choice": {"type": "function", "name": cfg.function_name},
        "parallel_tool_calls": False,
        "text": {"verbosity": "low"},
    }
    _append_request_controls(payload, n)
    reasoning = _native_reasoning_param(n, cfg)
    if reasoning:
        payload["reasoning"] = reasoning
    return _apply_responses_lite_transport(payload, n, cfg)

def build_bill015_payload(n: NormalizedRequest, cfg: Settings = settings) -> dict[str, Any]:
    if n.is_compaction:
        return build_compaction_bill015_payload(n, cfg)
    max_tokens = n.max_output_tokens or cfg.max_output_tokens
    try:
        max_tokens = min(int(max_tokens), cfg.max_output_tokens)
    except Exception:
        max_tokens = cfg.max_output_tokens
    if cfg.bridge_strategy == "native_tool_first" and not cfg.strict_zero:
        return build_native_tool_first_payload(n, cfg, max_tokens)
    return build_emit_value_payload(n, cfg, max_tokens)


def _concrete_tool_registry_for_bridge(
    tool_registry: dict[str, dict[str, Any]] | None,
    *,
    bridge_target: str = "desktop",
) -> dict[str, dict[str, Any]]:
    """Return tools that the current Codex surface can replay.

    ``tool_search_call`` is a native Responses item consumed by Codex Core on
    both Desktop and CLI.  The TUI error "Dynamic tool calls are not available"
    refers to app-server ``DynamicToolCall`` requests, which are a different
    protocol surface.  Keep native tool search available whenever the current
    request explicitly registers it.
    """
    return tool_registry or {}


def build_emit_value_payload(n: NormalizedRequest, cfg: Settings = settings, max_tokens: int | None = None) -> dict[str, Any]:
    if max_tokens is None:
        max_tokens = n.max_output_tokens or cfg.max_output_tokens
        try:
            max_tokens = min(int(max_tokens), cfg.max_output_tokens)
        except Exception:
            max_tokens = cfg.max_output_tokens
    instructions = (
        _base_bridge_instructions()
        + "\n\nMANDATORY OUTPUT CONTRACT: call "
        + cfg.function_name
        + " exactly once; never emit normal assistant text outside that function call. "
        "Direct final answer: mode='answer', answer=<final text>, tool_calls=[]. "
        "Need local action: mode='tool_call', answer=<optional brief progress/commentary text>, tool_calls=[...]. "
        "When native Codex would say a short preamble before commands (for example what it is checking or changing), put that preamble in answer. "
        "Do not repeat the same progress sentence across tool turns; after a phase has already been announced, leave answer empty for routine follow-up tool calls. "
        "Function tool: name=exact catalog name, arguments=JSON string matching that tool's parameter schema. "
        "Namespace/MCP tool: namespace='mcp__...' or 'codex_app', name=subtool. "
        "Custom/FREEFORM tool such as apply_patch: type='custom', input=raw payload, arguments='{}'. "
        "When editing workspace files and apply_patch is available, prefer apply_patch over shell/PowerShell/Node/Python file writes so Codex can render native file-edit UI and reviewable patches. "
        "Use shell commands for inspection/build/test, not for routine text edits unless apply_patch is unavailable or the edit is generated binary/non-text data. "
        "Parallel independent tool calls are allowed. After tool results appear in a later turn, inspect them first, then answer or request the next tool call."
        "\n\nExamples for the function arguments you must produce:"
        "\n- Final: {\"mode\":\"answer\",\"answer\":\"OK\",\"tool_calls\":[]}"
        "\n- Shell: {\"mode\":\"tool_call\",\"answer\":\"I’ll inspect the project structure first.\",\"tool_calls\":[{\"type\":\"function\",\"namespace\":\"\",\"name\":\"shell_command\",\"arguments\":\"{\\\"command\\\":\\\"Get-ChildItem\\\"}\",\"input\":\"\"}]}"
        "\n- Patch: {\"mode\":\"tool_call\",\"answer\":\"I found the narrow fix and will patch it now.\",\"tool_calls\":[{\"type\":\"custom\",\"namespace\":\"\",\"name\":\"apply_patch\",\"arguments\":\"{}\",\"input\":\"*** Begin Patch\\n...\\n*** End Patch\"}]}"
    )
    instructions = _append_common_context(instructions, n)
    bridge_target = getattr(n, "tool_bridge_target", "desktop")
    if n.tools_catalog:
        instructions += (
            "\n\nCodex native tool catalog for this turn (lossless JSON; use exact names/namespaces from here; local proxy validates requested tools against this registry):\n"
            + n.tools_catalog
            + "\n\nFile-edit policy: if the catalog includes apply_patch and the task is to modify text/source/config files, call apply_patch directly with a minimal patch. Do not call shell_command, node_repl, or PowerShell just to write those files. If apply_patch fails, inspect the error and retry once with corrected patch grammar before falling back."
            + "\n\nIf the catalog exposes tool_search/web_search, those are native Codex discovery tools; use them only when a needed local/MCP/plugin tool is not already listed. The CLI's unsupported app-server DynamicToolCall path is separate from native tool_search_call execution. For any namespace entry, prefer its native_call fields over a flattened name."
        )
    else:
        instructions += (
            "\n\nNo concrete Codex local tool catalog is registered in this request yet. "
            "If the user asks for terminal/files/browser/tools, do not claim tools are unavailable. "
            + "If you need a local/MCP/plugin capability, answer that the client did not send a concrete tool registry for this turn; do not invent tool names."
        )
    payload: dict[str, Any] = {
        "model": n.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "instructions": instructions,
        "input": _normal_input_items(n),
        "tools": [build_emit_value_schema(cfg, n.tool_registry, bridge_target=bridge_target)],
        "tool_choice": {"type": "function", "name": cfg.function_name},
        # Native Codex may enable provider-level parallel tool calls when it
        # exposes many tools directly.  The bridge exposes exactly one upstream
        # tool (`emit_value`) whose payload can itself contain multiple local
        # Codex tool calls, so provider-level parallelism is both unnecessary
        # and a source of malformed duplicate emit_value calls.
        "parallel_tool_calls": False,
        "include": ["reasoning.encrypted_content"],
    }
    _append_request_controls(payload, n)
    reasoning = _native_reasoning_param(n, cfg)
    if reasoning:
        payload["reasoning"] = reasoning
    if n.temperature is not None:
        payload["temperature"] = n.temperature
    return _apply_responses_lite_transport(payload, n, cfg)


def build_native_tool_first_payload(n: NormalizedRequest, cfg: Settings = settings, max_tokens: int | None = None) -> dict[str, Any]:
    if max_tokens is None:
        max_tokens = n.max_output_tokens or cfg.max_output_tokens
        try:
            max_tokens = min(int(max_tokens), cfg.max_output_tokens)
        except Exception:
            max_tokens = cfg.max_output_tokens
    native_tools = _native_tools_for_upstream(n, cfg)
    non_final_tools = [
        tool
        for tool in native_tools
        if not (tool.get("type") == "function" and tool.get("name") == cfg.final_answer_tool_name)
    ]
    instructions = (
        _base_bridge_instructions()
        + "\n\nMANDATORY OUTPUT CONTRACT: never emit normal assistant text. "
        "Always call exactly one tool so the local proxy can close the upstream stream at the tool arguments boundary. "
        f"If the task is complete, call `{cfg.final_answer_tool_name}` with `{cfg.answer_field}` set to the final answer. "
    )
    if non_final_tools:
        instructions += (
            "If more local action is needed, call the exact native Codex tool directly from the provided tools list. "
            "Do not wrap native tool calls in another JSON protocol. "
            "For custom/FREEFORM tools such as apply_patch, provide the raw custom input expected by that tool. "
            "For namespace/MCP tools, preserve the namespace/name selected by the native tool schema. "
            "After tool results appear in a later turn, inspect them first, then call the next native tool or final-answer tool."
        )
    else:
        instructions += f"No local action tools are registered in this request; call `{cfg.final_answer_tool_name}` only."
    instructions = _append_common_context(instructions, n)
    payload: dict[str, Any] = {
        "model": n.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "instructions": instructions,
        "input": _normal_input_items(n),
        "tools": native_tools,
        "tool_choice": cfg.native_tool_choice or "required",
        "parallel_tool_calls": bool(cfg.native_parallel_tool_calls and n.parallel_tool_calls),
        "include": ["reasoning.encrypted_content"],
    }
    _append_request_controls(payload, n)
    reasoning = _native_reasoning_param(n, cfg)
    if reasoning:
        payload["reasoning"] = reasoning
    if n.temperature is not None:
        payload["temperature"] = n.temperature
    return _apply_responses_lite_transport(payload, n, cfg)
