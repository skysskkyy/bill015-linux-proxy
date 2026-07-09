from __future__ import annotations

import json
from typing import Any

from .config import Settings, settings
from .models import NormalizedRequest
from .tool_bridge import build_client_tool_catalog
from .tool_history import parse_tool_history, render_tool_feedback_for_model
from .usage_estimator import estimate_chat_usage_from_body, estimate_responses_usage_from_body


def model_identity_instruction(model: str) -> str:
    # Keep identity aligned with the actual upstream model after alias mapping.
    # Do not hard-code gpt-5.5; that broke user switching to gpt-5.4.
    model = str(model or "the configured model")
    return f"If asked what model you are, answer {model}. Do not claim to be a different model."

def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                typ = item.get("type")
                if "text" in item and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif typ in {"input_text", "output_text", "text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)
    return str(content)

def flatten_responses_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                typ = item.get("type")
                if typ in {"function_call_output", "tool_result", "computer_call_output", "custom_tool_call_output"}:
                    parts.append("[tool_output] " + json.dumps(item, ensure_ascii=False))
                    continue
                if typ in {"function_call", "custom_tool_call"}:
                    parts.append("[assistant_tool_call] " + json.dumps(item, ensure_ascii=False))
                    continue
                role = item.get("role", "user")
                content = flatten_content(item.get("content", item.get("text", "")))
                if content:
                    parts.append(f"[{role}] {content}")
                elif typ or "call_id" in item or "output" in item:
                    parts.append("[event] " + json.dumps(item, ensure_ascii=False))
            else:
                parts.append(flatten_content(item))
        return "\n".join(parts)
    return flatten_content(value)

def flatten_chat_messages(messages: Any) -> tuple[str, str]:
    instructions: list[str] = []
    user_parts: list[str] = []
    if not isinstance(messages, list):
        return "", flatten_content(messages)
    for msg in messages:
        if not isinstance(msg, dict):
            user_parts.append(flatten_content(msg))
            continue
        role = msg.get("role", "user")
        content = flatten_content(msg.get("content", ""))
        if role in {"system", "developer"}:
            instructions.append(content)
        else:
            user_parts.append(f"[{role}] {content}")
    return "\n".join(x for x in instructions if x), "\n".join(x for x in user_parts if x)

def parse_codex_turn_metadata(client_metadata: Any) -> dict[str, Any]:
    if not isinstance(client_metadata, dict):
        return {}
    raw = client_metadata.get("x-codex-turn-metadata")
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}

def detect_request_kind(body: dict[str, Any]) -> tuple[str, bool]:
    cm = body.get("client_metadata") if isinstance(body.get("client_metadata"), dict) else {}
    turn_md = parse_codex_turn_metadata(cm)
    request_kind = str(turn_md.get("request_kind") or body.get("request_kind") or "turn")
    is_compaction = request_kind == "compaction" or isinstance(turn_md.get("compaction"), dict)
    return request_kind, is_compaction

def extract_context_messages(value: Any, *, max_chars: int = 90000) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in {"system", "developer"}:
            continue
        text = flatten_content(item.get("content", item.get("text", "")))
        if text:
            parts.append(f"[{role}]\n{text}")
    out = "\n\n".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n[local proxy truncated developer/system context]"
    return out

def _short_json(value: Any, max_chars: int = 18000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        return text[: max_chars // 2] + f"\n...[truncated {len(text) - max_chars} chars]...\n" + text[-max_chars // 2 :]
    return text

def native_input_transcript(value: Any, *, max_chars: int = 180000, include_context_roles: bool = False) -> str:
    if isinstance(value, str):
        return value[:max_chars]
    if not isinstance(value, list):
        return flatten_content(value)[:max_chars]
    lines: list[str] = []
    omitted_context = 0
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            lines.append(f"[{idx}] {flatten_content(item)}")
            continue
        role = str(item.get("role") or "")
        typ = str(item.get("type") or "message")
        if role in {"system", "developer"} and not include_context_roles:
            omitted_context += 1
            continue
        if typ == "message" or role:
            text = flatten_content(item.get("content", item.get("text", "")))
            if text:
                lines.append(f"[{idx}] {role or 'event'} message:\n{text}")
                continue
        if typ in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "computer_call"}:
            lines.append(f"[{idx}] assistant {typ}: " + _short_json(item, 12000))
            continue
        if typ in {"function_call_output", "custom_tool_call_output", "tool_result", "tool_search_output", "computer_call_output"}:
            lines.append(f"[{idx}] tool output: " + _short_json(item, 14000))
            continue
        if typ == "reasoning":
            summary = item.get("summary")
            if summary:
                lines.append(f"[{idx}] reasoning summary: " + _short_json(summary, 4000))
            else:
                lines.append(f"[{idx}] reasoning: <encrypted/omitted>")
            continue
        lines.append(f"[{idx}] {typ}: " + _short_json(item, 8000))
    if omitted_context:
        lines.insert(0, f"[local proxy] moved {omitted_context} developer/system message(s) into upstream system context.")
    out = "\n\n".join(lines)
    if len(out) > max_chars:
        # Preserve both early setup and latest turn/tool outputs, like a local memento.
        head = max_chars // 3
        tail = max_chars - head
        out = out[:head] + f"\n...[local proxy transcript truncated {len(out) - max_chars} chars; keeping latest context below]...\n" + out[-tail:]
    return out

def last_user_instruction(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "user":
                text = flatten_content(item.get("content", item.get("text", "")))
                if text:
                    return text
    return flatten_content(value)

def request_needs_passthrough(body: dict[str, Any]) -> tuple[bool, str | None]:
    """Decide whether to preserve Codex native Responses behavior.

    Codex Desktop often sends the full local tool schema even for ordinary text
    chat. Therefore, the mere presence of `tools` must NOT force passthrough;
    otherwise every pure-text prompt becomes a billable normal upstream call.

    Passthrough is reserved for requests that actually need native protocol
    semantics: media/file payloads, previous tool outputs, or prompts that
    clearly ask the agent to run local/browser/file-editing tools.
    """

    def walk_for_protocol_need(x: Any) -> str | None:
        if isinstance(x, dict):
            typ = str(x.get("type", "")).lower()
            if typ in {"input_image", "image", "image_url", "input_file", "file", "computer_screenshot"}:
                return typ or "media_content"
            if any(k in x for k in ("image_url", "file_id", "file_data", "mime_type")):
                return "media_or_file_field"
            for v in x.values():
                found = walk_for_protocol_need(v)
                if found:
                    return found
        elif isinstance(x, list):
            for v in x:
                found = walk_for_protocol_need(v)
                if found:
                    return found
        return None

    found = walk_for_protocol_need(body.get("input")) or walk_for_protocol_need(body.get("messages"))
    if found:
        return True, found

    def collect_text(x: Any, out: list[str]) -> None:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            # Do not inspect the tool schema itself; it contains words like shell/browser.
            for k, v in x.items():
                if k == "tools":
                    continue
                collect_text(v, out)
        elif isinstance(x, list):
            for v in x:
                collect_text(v, out)

    def latest_user_text_from_sequence(seq: Any) -> list[str]:
        if isinstance(seq, str):
            return [seq]
        if isinstance(seq, list):
            # Prefer only the latest user message. Codex often sends prior context;
            # old turns mentioning PowerShell/browser must not force passthrough for
            # a new plain-text greeting.
            for item in reversed(seq):
                if isinstance(item, dict) and item.get("role") == "user":
                    out: list[str] = []
                    collect_text(item.get("content", item), out)
                    return out
            out: list[str] = []
            collect_text(seq[-1] if seq else "", out)
            return out
        out: list[str] = []
        collect_text(seq, out)
        return out

    texts: list[str] = []
    texts.extend(latest_user_text_from_sequence(body.get("input")))
    texts.extend(latest_user_text_from_sequence(body.get("messages")))
    # Tool-intent text is now handled by the BILL-015 tool-call bridge, not passthrough.

    # Explicit tool_choice is also handled by the bridge. Only media/file payloads
    # need native passthrough at this stage.
    return False, None

def collect_deferred_tools_from_input(value: Any) -> list[dict[str, Any]]:
    """Return tools exposed by native tool_search_output items.

    In Codex native logs, deferred MCP/browser/node tools are often not added to
    top-level `tools`; they are carried in prior `input[]` items of type
    `tool_search_output`. The model can then call names such as `js` directly.
    We surface those tools in the bridge catalog so the upstream model sees the
    same affordances.
    """
    found: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return found
    for item in value:
        if not isinstance(item, dict) or item.get("type") != "tool_search_output":
            continue
        tools = item.get("tools")
        if isinstance(tools, list):
            found.extend(t for t in tools if isinstance(t, dict))
    return found[-50:]

def combine_tool_catalogs(primary_tools: Any, raw_input: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    catalog, registry = build_client_tool_catalog(primary_tools)
    deferred = collect_deferred_tools_from_input(raw_input)
    if not deferred:
        return catalog, registry
    deferred_catalog, deferred_registry = build_client_tool_catalog(deferred, max_chars=80_000)
    registry.update(deferred_registry)
    if deferred_catalog:
        if catalog:
            catalog = catalog + "\n\nDeferred tools already exposed by native tool_search_output in this thread:\n" + deferred_catalog
        else:
            catalog = deferred_catalog
    return catalog, registry

def normalize_responses_request(body: dict[str, Any], cfg: Settings = settings) -> NormalizedRequest:
    model = cfg.map_model(body.get("model"))
    raw_input = body.get("input", "")
    usage_estimate = estimate_responses_usage_from_body(body)
    tool_history = parse_tool_history(raw_input)
    latest_tool_summary = render_tool_feedback_for_model(tool_history)
    tools_catalog, tool_registry = combine_tool_catalogs(body.get("tools"), raw_input)
    request_kind, is_compaction = detect_request_kind(body)
    base_instructions = flatten_content(body.get("instructions", ""))
    context_messages = extract_context_messages(raw_input)
    instructions = base_instructions
    if context_messages:
        instructions = (instructions + "\n\n" if instructions else "") + "Client developer/system messages from native Codex input:\n" + context_messages
    user_input = native_input_transcript(raw_input, include_context_roles=False) if isinstance(raw_input, list) else flatten_responses_input(raw_input)
    return NormalizedRequest(
        model=model,
        original_model=body.get("model"),
        raw_input=raw_input,
        raw_tools=body.get("tools"),
        request_kind=request_kind,
        is_compaction=is_compaction,
        estimated_input_tokens=usage_estimate.input_tokens,
        usage_estimate=usage_estimate,
        parallel_tool_calls=bool(body.get("parallel_tool_calls", not is_compaction)),
        tool_choice=body.get("tool_choice", "auto"),
        prompt_cache_key=body.get("prompt_cache_key") if isinstance(body.get("prompt_cache_key"), str) else None,
        text_config=body.get("text") if isinstance(body.get("text"), dict) else None,
        tools_summary=tools_catalog,
        tools_catalog=tools_catalog,
        tool_registry=tool_registry,
        tool_history=tool_history,
        latest_tool_summary=latest_tool_summary,
        pending_tool_call_count=len(tool_history.pending_calls),
        latest_tool_failed=bool(tool_history.failed_outputs and tool_history.failed_outputs[-1] in tool_history.latest_outputs),
        instructions=instructions,
        user_input=user_input,
        want_stream=bool(body.get("stream", True)),
        client_api="responses",
        is_primary_path=True,
        temperature=body.get("temperature"),
        max_output_tokens=body.get("max_output_tokens") or body.get("max_tokens"),
        reasoning=body.get("reasoning") if isinstance(body.get("reasoning"), dict) else None,
        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
        client_metadata=body.get("client_metadata") if isinstance(body.get("client_metadata"), dict) else None,
    )

def normalize_chat_request(body: dict[str, Any], cfg: Settings = settings) -> NormalizedRequest:
    instructions, user_input = flatten_chat_messages(body.get("messages", []))
    tools_catalog, tool_registry = build_client_tool_catalog(body.get("tools"))
    usage_estimate = estimate_chat_usage_from_body(body)
    return NormalizedRequest(
        model=cfg.map_model(body.get("model")),
        original_model=body.get("model"),
        raw_input=body.get("messages", []),
        raw_tools=body.get("tools"),
        estimated_input_tokens=usage_estimate.input_tokens,
        usage_estimate=usage_estimate,
        parallel_tool_calls=bool(body.get("parallel_tool_calls", True)),
        tool_choice=body.get("tool_choice", "auto"),
        prompt_cache_key=body.get("prompt_cache_key") if isinstance(body.get("prompt_cache_key"), str) else None,
        text_config=body.get("text") if isinstance(body.get("text"), dict) else None,
        tools_summary=tools_catalog,
        tools_catalog=tools_catalog,
        tool_registry=tool_registry,
        instructions=instructions,
        user_input=user_input,
        want_stream=bool(body.get("stream", False)),
        client_api="chat.completions",
        is_primary_path=False,
        temperature=body.get("temperature"),
        max_output_tokens=body.get("max_tokens") or body.get("max_output_tokens"),
        reasoning=body.get("reasoning") if isinstance(body.get("reasoning"), dict) else None,
        metadata=None,
    )
