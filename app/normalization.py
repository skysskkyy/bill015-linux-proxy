from __future__ import annotations

import json
from typing import Any

from .config import Settings, settings
from .models import NormalizedRequest
from .tool_bridge import build_client_tool_catalog
from .tool_history import parse_tool_history, render_tool_feedback_for_model
from .usage_estimator import estimate_chat_usage_from_body, estimate_responses_usage_from_body, estimate_text_tokens

UNSUPPORTED_IMAGE_NOTICE = (
    "[local proxy notice: omitted unsupported image input. This local proxy does not support "
    "image/screenshot uploads; no visual content was sent upstream. Do not infer details from "
    "the image. Use the available text/tool context, or ask the user for a text description.]"
)


IMAGE_CONTENT_TYPES = {"input_image", "image", "image_url", "computer_screenshot"}
IMAGE_FIELD_NAMES = {"image_url", "image", "image_data", "screenshot", "screenshot_url"}


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

def _clip_text_to_token_budget(text: str, max_tokens: int, *, keep: str = "head_tail") -> str:
    text = str(text or "")
    if max_tokens <= 0:
        return ""
    if estimate_text_tokens(text) <= max_tokens:
        return text
    # Binary search by character count, but the stopping criterion is token
    # budget, not a hard character ceiling.
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        if keep == "tail":
            candidate = text[-mid:] if mid else ""
        elif keep == "head":
            candidate = text[:mid]
        else:
            head = mid // 3
            tail = mid - head
            candidate = text[:head] + f"\n...[token-budget omitted middle content; original_tokens~{estimate_text_tokens(text)} budget={max_tokens}]...\n" + text[-tail:]
        if estimate_text_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best or text[: max(200, max_tokens * 2)]


def _join_sections_by_token_budget(sections: list[tuple[str, str, str]], max_tokens: int) -> str:
    """Join priority-ordered context sections under an estimated token budget.

    Sections are `(label, text, keep)` where keep controls clipping strategy.
    The caller already orders sections from most to least important.
    """
    out: list[str] = []
    remaining = max_tokens
    for label, text, keep in sections:
        text = str(text or "").strip()
        if not text or remaining <= 0:
            continue
        header = f"--- {label} ---\n"
        header_tokens = estimate_text_tokens(header)
        if header_tokens >= remaining:
            break
        body_budget = remaining - header_tokens
        if keep == "tail":
            body_budget = min(body_budget, max(2_000, max_tokens // 3))
        body = _clip_text_to_token_budget(text, body_budget, keep=keep)
        if not body:
            continue
        out.append(header + body)
        remaining = max_tokens - estimate_text_tokens("\n\n".join(out))
    return "\n\n".join(out)


def _context_token_budget(body: dict[str, Any], usage_input_tokens: int, cfg: Settings = settings) -> int:
    requested_output = body.get("max_output_tokens") or body.get("max_tokens") or cfg.max_output_tokens
    try:
        output_reserve = max(1024, int(requested_output))
    except Exception:
        output_reserve = max(1024, cfg.max_output_tokens)
    # Keep a generous but bounded upstream context budget. The native Codex
    # request can be huge; older material should be represented by compaction
    # summaries rather than raw replay.
    total_budget = 120_000
    overhead_reserve = 8_000
    return max(8_000, min(80_000, total_budget - output_reserve - overhead_reserve, usage_input_tokens + 12_000))


def extract_ordered_instruction_context(value: Any, *, max_tokens: int = 16_000) -> str:
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
    clipped = _clip_text_to_token_budget(out, max_tokens, keep="head_tail")
    if clipped != out:
        clipped += "\n[local proxy token-budget clipped developer/system context]"
    return clipped


def extract_role_contexts(value: Any, *, max_tokens: int = 16_000) -> tuple[str, str, str]:
    """Extract native system/developer context without flattening it into user chat.

    Returns `(system_context, developer_context, ordered_context)`.
    `ordered_context` preserves the relative order of native system/developer
    messages and is what the payload builder feeds into upstream
    `instructions`. The separated role fields are kept on `NormalizedRequest`
    for auditability and future prompt builders.
    """
    if not isinstance(value, list):
        return "", "", ""
    system_parts: list[str] = []
    developer_parts: list[str] = []
    ordered_parts: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in {"system", "developer"}:
            continue
        text = flatten_content(item.get("content", item.get("text", ""))).strip()
        if not text:
            continue
        section = f"[{idx}] [{role}]\n{text}"
        ordered_parts.append(section)
        if role == "system":
            system_parts.append(section)
        else:
            developer_parts.append(section)

    raw_ordered = "\n\n".join(ordered_parts)
    raw_system = "\n\n".join(system_parts)
    raw_developer = "\n\n".join(developer_parts)
    ordered = _clip_text_to_token_budget(raw_ordered, max_tokens, keep="head_tail")
    system = _clip_text_to_token_budget(raw_system, max(1_000, max_tokens // 2), keep="head_tail")
    developer = _clip_text_to_token_budget(raw_developer, max(1_000, max_tokens // 2), keep="head_tail")
    if ordered != raw_ordered:
        ordered += "\n[local proxy token-budget clipped ordered developer/system context]"
    if system != raw_system:
        system += "\n[local proxy token-budget clipped system context]"
    if developer != raw_developer:
        developer += "\n[local proxy token-budget clipped developer context]"
    return system, developer, ordered


def extract_context_messages(value: Any, *, max_tokens: int = 16_000) -> str:
    """Backward-compatible alias for ordered native instruction context."""
    return extract_ordered_instruction_context(value, max_tokens=max_tokens)

def _short_json(value: Any, max_tokens: int = 4_000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(value)
    return _clip_text_to_token_budget(text, max_tokens, keep="head_tail")

def native_input_transcript(
    value: Any,
    *,
    max_tokens: int = 32_000,
    include_context_roles: bool = False,
    omit_latest_user_body: bool = False,
    omit_latest_tool_batch: bool = False,
) -> str:
    if isinstance(value, str):
        return _clip_text_to_token_budget(value, max_tokens, keep="tail")
    if not isinstance(value, list):
        return _clip_text_to_token_budget(flatten_content(value), max_tokens, keep="tail")
    history = parse_tool_history(value)
    latest_batch_ids = {out.call_id for out in history.latest_outputs if out.call_id} if omit_latest_tool_batch else set()
    latest_user_index = -1
    for idx, item in enumerate(value):
        if isinstance(item, dict) and item.get("role") == "user" and flatten_content(item.get("content", item.get("text", ""))).strip():
            latest_user_index = idx
    current_sections: list[str] = []
    recent_sections: list[str] = []
    history_sections: list[str] = []
    omitted_context = 0
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            history_sections.append(f"[{idx}] {flatten_content(item)}")
            continue
        role = str(item.get("role") or "")
        typ = str(item.get("type") or "message")
        if role in {"system", "developer"} and not include_context_roles:
            omitted_context += 1
            continue
        if typ == "message" or role:
            text = flatten_content(item.get("content", item.get("text", "")))
            if text:
                if idx == latest_user_index and omit_latest_user_body:
                    current_sections.append(f"[{idx}] current user message moved above; body omitted here to avoid duplication.")
                elif idx == latest_user_index:
                    current_sections.append(f"[{idx}] {role or 'event'} message:\n{text}")
                else:
                    history_sections.append(f"[{idx}] {role or 'event'} message:\n{text}")
                continue
        if typ in {"function_call", "custom_tool_call", "tool_search_call", "web_search_call", "computer_call"}:
            call_id = str(item.get("call_id") or item.get("id") or "")
            target = recent_sections if call_id in latest_batch_ids else history_sections
            target.append(f"[{idx}] assistant {typ}: " + _short_json(item, 2_500))
            continue
        if typ in {"function_call_output", "custom_tool_call_output", "tool_result", "tool_search_output", "computer_call_output"}:
            call_id = str(item.get("call_id") or item.get("id") or "")
            if call_id in latest_batch_ids:
                recent_sections.append(f"[{idx}] latest tool batch output moved above; raw body omitted here to avoid duplication.")
            else:
                history_sections.append(f"[{idx}] tool output: " + _short_json(item, 2_000))
            continue
        if typ == "reasoning":
            summary = item.get("summary")
            if summary:
                recent_sections.append(f"[{idx}] reasoning summary: " + _short_json(summary, 1_200))
            else:
                history_sections.append(f"[{idx}] reasoning: <encrypted/omitted>")
            continue
        history_sections.append(f"[{idx}] {typ}: " + _short_json(item, 1_500))
    if omitted_context:
        current_sections.insert(0, f"[local proxy] moved {omitted_context} developer/system message(s) into upstream system context.")
    sections = [
        ("Current task state", "\n\n".join(current_sections), "head_tail"),
        ("Recent non-output events / reasoning", "\n\n".join(recent_sections), "head_tail"),
        ("Related prior history", "\n\n".join(history_sections), "tail"),
    ]
    return _join_sections_by_token_budget(sections, max_tokens)

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


def sanitize_unsupported_image_inputs(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Replace image/screenshot payloads with a textual local-proxy notice.

    The BILL-015 bridge cannot safely early-abort native multimodal/image
    passthrough.  Previously strict-zero mode rejected such requests with a 422,
    while non-strict mode could accidentally forward billable images upstream.
    Instead, strip visual bytes/URLs from user-visible input fields and tell the
    upstream text model exactly what happened.
    """

    replacements = 0

    def is_image_node(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        typ = str(value.get("type") or "").lower()
        if typ in IMAGE_CONTENT_TYPES:
            return True
        mime = str(value.get("mime_type") or value.get("media_type") or value.get("content_type") or "").lower()
        if mime.startswith("image/"):
            return True
        for key in ("file_data", "data", "url", "source"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip().lower().startswith("data:image/"):
                return True
        return any(k in value for k in IMAGE_FIELD_NAMES)

    def notice_part(kind: str | None = None) -> dict[str, str]:
        suffix = f" ({kind})" if kind else ""
        return {
            "type": "input_text",
            "text": UNSUPPORTED_IMAGE_NOTICE.replace("omitted unsupported image input", f"omitted unsupported image input{suffix}"),
        }

    def sanitize_value(value: Any, *, top_level_input_item: bool = False, list_items_are_top_level: bool = False) -> Any:
        nonlocal replacements
        if isinstance(value, dict):
            if is_image_node(value):
                replacements += 1
                typ = str(value.get("type") or "image").lower()
                part = notice_part(typ)
                if top_level_input_item:
                    return {"role": "user", "content": [part]}
                return part
            out: dict[str, Any] = {}
            for k, v in value.items():
                # Only recurse inside request payload fields. Tool schemas may
                # mention image_url/mime_type as parameter names; do not rewrite
                # the schema, only actual input/message content.
                if k in {"input", "messages"}:
                    out[k] = sanitize_sequence(v, top_level_items=True) if isinstance(v, list) else sanitize_value(v)
                elif k in {"content", "output"}:
                    out[k] = sanitize_sequence(v, top_level_items=False) if isinstance(v, list) else sanitize_value(v)
                else:
                    out[k] = v
            return out
        if isinstance(value, list):
            return sanitize_sequence(value, top_level_items=list_items_are_top_level)
        return value

    def sanitize_sequence(seq: list[Any], *, top_level_items: bool = False) -> list[Any]:
        out: list[Any] = []
        for item in seq:
            out.append(sanitize_value(item, top_level_input_item=top_level_items and is_image_node(item)))
        return out

    sanitized = dict(body)
    if "input" in sanitized:
        value = sanitized.get("input")
        sanitized["input"] = sanitize_sequence(value, top_level_items=True) if isinstance(value, list) else sanitize_value(value)
    if "messages" in sanitized:
        value = sanitized.get("messages")
        sanitized["messages"] = sanitize_sequence(value, top_level_items=True) if isinstance(value, list) else sanitize_value(value)
    return sanitized, replacements

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


def collect_additional_tools_from_input(value: Any) -> list[dict[str, Any]]:
    """Return tools carried in Responses Lite ``additional_tools`` input items.

    Newer Codex model metadata can set ``use_responses_lite``.  In that request
    shape Codex moves the native tool list out of top-level ``tools`` and into a
    developer input item:

        {"type": "additional_tools", "role": "developer", "tools": [...]}

    The local bridge still needs that list to build its emit_value catalog and
    validation registry.  Without this extraction the proxy sees an empty tool
    registry and strict mode truthfully tells the model that no local tools are
    available, which is the observed gpt-5.6/CLI failure mode.
    """
    found: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return found
    for item in value:
        if not isinstance(item, dict):
            continue
        typ = str(item.get("type") or "")
        if typ not in {"additional_tools", "additionalTools"}:
            continue
        tools = item.get("tools")
        if isinstance(tools, list):
            found.extend(t for t in tools if isinstance(t, dict))
    return found[-200:]


def has_responses_lite_additional_tools(value: Any) -> bool:
    """Return whether input uses Codex Responses Lite additional_tools."""
    if not isinstance(value, list):
        return False
    return any(isinstance(item, dict) and str(item.get("type") or "") in {"additional_tools", "additionalTools"} for item in value)


def combine_tool_catalogs(primary_tools: Any, raw_input: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    additional_tools = collect_additional_tools_from_input(raw_input)
    merged_primary: Any = primary_tools
    if additional_tools:
        if isinstance(primary_tools, list):
            merged_primary = [*primary_tools, *additional_tools]
        else:
            merged_primary = additional_tools
    catalog, registry = build_client_tool_catalog(merged_primary)
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


def detect_tool_bridge_target(body: dict[str, Any]) -> str:
    """Best-effort client surface detection for local tool-call replay.

    Desktop can consume native dynamic discovery items such as
    ``tool_search_call`` and use their later ``tool_search_output`` to expose
    deferred MCP/plugin tools.  The interactive CLI/TUI bridge currently
    rejects those dynamic item types, so only apply the TUI restriction when the
    request metadata explicitly identifies that surface.
    """

    haystack: list[str] = []
    for key in ("client", "client_name", "surface", "source", "app", "origin"):
        value = body.get(key)
        if isinstance(value, str):
            haystack.append(value)

    for meta in (body.get("metadata"), body.get("client_metadata")):
        if not isinstance(meta, dict):
            continue
        for key in ("x-codex-turn-metadata", "codex_client", "client", "client_name", "surface", "source", "app"):
            value = meta.get(key)
            if isinstance(value, str):
                haystack.append(value)
                try:
                    parsed = json.loads(value)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    haystack.extend(str(v) for v in parsed.values() if isinstance(v, str))
            elif isinstance(value, dict):
                haystack.extend(str(v) for v in value.values() if isinstance(v, str))

    text = " ".join(haystack).lower()
    if any(token in text for token in ("tui", "codex-cli", "codex cli", "terminal", "console")):
        return "tui"
    return "desktop"


def extract_reasoning_config(body: dict[str, Any]) -> dict[str, Any] | None:
    """Return client-requested reasoning controls in upstream Responses shape.

    Native Responses clients send ``reasoning={"effort": ...}``, while some
    Chat/compat clients send top-level ``reasoning_effort``.  BILL-015 builds a
    new upstream Responses payload, so normalize both forms here instead of
    silently dropping top-level thinking-level controls.
    """
    reasoning = dict(body.get("reasoning") or {}) if isinstance(body.get("reasoning"), dict) else {}
    if body.get("reasoning_effort") is not None and "effort" not in reasoning:
        reasoning["effort"] = body.get("reasoning_effort")
    if body.get("reasoning_summary") is not None and "summary" not in reasoning:
        reasoning["summary"] = body.get("reasoning_summary")
    return reasoning or None


def _optional_string(body: dict[str, Any], key: str) -> str | None:
    value = body.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _responses_lite_requested(body: dict[str, Any], raw_input: Any, request_headers: dict[str, str] | None = None) -> bool:
    if has_responses_lite_additional_tools(raw_input):
        return True
    metadata = body.get("client_metadata") if isinstance(body.get("client_metadata"), dict) else {}
    for key in ("x-openai-internal-codex-responses-lite", "responses_lite", "use_responses_lite"):
        value = body.get(key, metadata.get(key))
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "responses_lite"}:
            return True
    for key, value in (request_headers or {}).items():
        if str(key).lower() == "x-openai-internal-codex-responses-lite" and str(value).strip().lower() in {"1", "true", "yes"}:
            return True
    return False

def normalize_responses_request(
    body: dict[str, Any],
    cfg: Settings = settings,
    request_headers: dict[str, str] | None = None,
) -> NormalizedRequest:
    model = cfg.map_model(body.get("model"))
    raw_input = body.get("input", "")
    usage_estimate = estimate_responses_usage_from_body(body)
    tool_history = parse_tool_history(raw_input)
    latest_tool_summary = render_tool_feedback_for_model(tool_history)
    tools_catalog, tool_registry = combine_tool_catalogs(body.get("tools"), raw_input)
    tool_bridge_target = detect_tool_bridge_target(body)
    request_kind, is_compaction = detect_request_kind(body)
    base_instructions = flatten_content(body.get("instructions", ""))
    context_budget = _context_token_budget(body, usage_estimate.input_tokens, cfg)
    if is_compaction:
        system_context, developer_context, ordered_context = extract_role_contexts(
            raw_input,
            max_tokens=max(16_000, usage_estimate.input_tokens),
        )
    else:
        system_context, developer_context, ordered_context = "", "", ""
    instructions_sections: list[str] = []
    if base_instructions:
        instructions_sections.append("=== Top-level Responses instructions ===\n" + base_instructions)
    if ordered_context:
        instructions_sections.append("=== Native Codex system/developer messages (original order) ===\n" + ordered_context)
    instructions = "\n\n".join(instructions_sections)
    instruction_tokens = estimate_text_tokens(instructions)
    transcript_budget = max(6_000, context_budget - instruction_tokens - estimate_text_tokens(latest_tool_summary) - 2_000)
    current_budget = max(4_000, min(32_000, context_budget // 2))
    current_user_request = _clip_text_to_token_budget(last_user_instruction(raw_input), current_budget, keep="tail")
    user_input = (
        native_input_transcript(
            raw_input,
            max_tokens=transcript_budget,
            include_context_roles=False,
            omit_latest_user_body=True,
            omit_latest_tool_batch=True,
        )
        if isinstance(raw_input, list)
        else _clip_text_to_token_budget(flatten_responses_input(raw_input), transcript_budget, keep="tail")
    )
    if not isinstance(raw_input, list):
        current_user_request = user_input
    if not current_user_request:
        current_user_request = user_input
    return NormalizedRequest(
        model=model,
        original_model=body.get("model"),
        raw_input=raw_input,
        raw_tools=body.get("tools"),
        system_context=system_context,
        developer_context=developer_context,
        current_user_request=current_user_request,
        history_summary=user_input,
        latest_tool_batch=latest_tool_summary,
        request_kind=request_kind,
        is_compaction=is_compaction,
        estimated_input_tokens=usage_estimate.input_tokens,
        usage_estimate=usage_estimate,
        parallel_tool_calls=bool(body.get("parallel_tool_calls", not is_compaction)),
        tool_choice=body.get("tool_choice", "auto"),
        previous_response_id=_optional_string(body, "previous_response_id"),
        prompt_cache_key=_optional_string(body, "prompt_cache_key"),
        prompt_cache_options=body.get("prompt_cache_options") if isinstance(body.get("prompt_cache_options"), dict) else None,
        service_tier=_optional_string(body, "service_tier"),
        truncation=_optional_string(body, "truncation"),
        text_config=body.get("text") if isinstance(body.get("text"), dict) else None,
        responses_lite=_responses_lite_requested(body, raw_input, request_headers),
        tools_summary=tools_catalog,
        tools_catalog=tools_catalog,
        tool_registry=tool_registry,
        tool_bridge_target=tool_bridge_target,
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
        reasoning=extract_reasoning_config(body),
        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
        client_metadata=body.get("client_metadata") if isinstance(body.get("client_metadata"), dict) else None,
        request_headers=dict(request_headers or {}),
    )

def normalize_chat_request(
    body: dict[str, Any],
    cfg: Settings = settings,
    request_headers: dict[str, str] | None = None,
) -> NormalizedRequest:
    instructions, user_input = flatten_chat_messages(body.get("messages", []))
    tools_catalog, tool_registry = build_client_tool_catalog(body.get("tools"))
    tool_bridge_target = detect_tool_bridge_target(body)
    usage_estimate = estimate_chat_usage_from_body(body)
    current_user_request = ""
    messages = body.get("messages", [])
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                current_user_request = flatten_content(msg.get("content", "")).strip()
                if current_user_request:
                    break
    if not current_user_request:
        current_user_request = user_input
    return NormalizedRequest(
        model=cfg.map_model(body.get("model")),
        original_model=body.get("model"),
        raw_input=body.get("messages", []),
        raw_tools=body.get("tools"),
        system_context=instructions,
        developer_context="",
        current_user_request=current_user_request,
        history_summary=user_input,
        latest_tool_batch="",
        estimated_input_tokens=usage_estimate.input_tokens,
        usage_estimate=usage_estimate,
        parallel_tool_calls=bool(body.get("parallel_tool_calls", True)),
        tool_choice=body.get("tool_choice", "auto"),
        previous_response_id=_optional_string(body, "previous_response_id"),
        prompt_cache_key=_optional_string(body, "prompt_cache_key"),
        prompt_cache_options=body.get("prompt_cache_options") if isinstance(body.get("prompt_cache_options"), dict) else None,
        service_tier=_optional_string(body, "service_tier"),
        truncation=_optional_string(body, "truncation"),
        text_config=body.get("text") if isinstance(body.get("text"), dict) else None,
        responses_lite=_responses_lite_requested(body, body.get("messages", []), request_headers),
        tools_summary=tools_catalog,
        tools_catalog=tools_catalog,
        tool_registry=tool_registry,
        tool_bridge_target=tool_bridge_target,
        instructions=instructions,
        user_input=user_input,
        want_stream=bool(body.get("stream", False)),
        client_api="chat.completions",
        is_primary_path=False,
        temperature=body.get("temperature"),
        max_output_tokens=body.get("max_tokens") or body.get("max_output_tokens"),
        reasoning=extract_reasoning_config(body),
        metadata=None,
        request_headers=dict(request_headers or {}),
    )
