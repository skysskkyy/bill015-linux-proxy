from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import time
from typing import Any, AsyncIterator, Iterable

import httpx
from fastapi import HTTPException

from .audit import audit_logger, redact
from .config import Settings, settings
from .sse import SSEEvent, encode_sse, parse_async_sse_lines
from .state import runtime_state
from .models import Bill015Result, BridgeToolCall, NormalizedRequest, local_response_id
from .tool_bridge import build_client_tool_catalog, parse_function_arguments
from .response_events import chat_json, chat_sse_generator, response_json, responses_sse_generator
from .token_usage import chat_usage, estimate_request_input_tokens

IDENTITY_INSTRUCTION = "You are GPT-5.5. If asked what model you are, answer GPT-5.5. Do not claim to be GPT-5.1 or any other model."

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
    user_text = "\n".join(texts).lower()
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
        estimated_input_tokens=estimate_request_input_tokens(body),
        parallel_tool_calls=bool(body.get("parallel_tool_calls", not is_compaction)),
        tool_choice=body.get("tool_choice", "auto"),
        prompt_cache_key=body.get("prompt_cache_key") if isinstance(body.get("prompt_cache_key"), str) else None,
        text_config=body.get("text") if isinstance(body.get("text"), dict) else None,
        tools_summary=tools_catalog,
        tools_catalog=tools_catalog,
        tool_registry=tool_registry,
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
    return NormalizedRequest(
        model=cfg.map_model(body.get("model")),
        original_model=body.get("model"),
        raw_input=body.get("messages", []),
        raw_tools=body.get("tools"),
        estimated_input_tokens=estimate_request_input_tokens(body),
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
                            "name": {"type": "string", "description": "Exact tool name from the Codex tool catalog, including namespace prefix if shown."},
                            "arguments": {"type": "string", "description": "For function tools: JSON string arguments matching the tool parameters schema."},
                            "input": {"type": "string", "description": "For custom/FREEFORM tools such as apply_patch: raw tool input, not JSON."}
                        },
                        "required": ["type", "name", "arguments", "input"],
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
        + IDENTITY_INSTRUCTION
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
        "For custom/FREEFORM tools such as apply_patch, set tool_calls[i].type='custom' and tool_calls[i].input to the raw freeform payload; do not JSON-wrap it. "
        "For the native tool_search tool, set type='tool_search' and arguments to JSON like {\"query\":\"node_repl js\",\"limit\":8}. "
        "For web_search, set type='web_search' and arguments to JSON action data only when the catalog exposes web_search and server-side browsing is explicitly needed. "
        "You may emit multiple independent tool_calls in the same response when they can run in parallel. "
        "After tool results are provided in a later turn, answer or request the next tool call(s). "
        + IDENTITY_INSTRUCTION
    )
    if n.instructions:
        system += "\n\nClient instructions:\n" + n.instructions
    if n.tools_catalog:
        system += (
            "\n\nCodex native tool catalog for this turn (lossless JSON; use exact names and schemas):\n"
            + n.tools_catalog
            + "\n\nIf a needed browser/computer/plugin tool is not listed directly but tool_search is listed, request tool_search first with an appropriate query so Codex can expose deferred tools in the next turn."
        )
    payload: dict[str, Any] = {
        "model": n.model,
        "stream": True,
        "store": False,
        "max_output_tokens": max_tokens,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": n.user_input},
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


def collect_bill015_result_from_events(events: Iterable[SSEEvent], n: NormalizedRequest, cfg: Settings = settings) -> Bill015Result:
    """Offline helper used by regression tests to validate event-state behavior."""
    result = Bill015Result(local_request_id=local_response_id())
    args_buffer: list[str] = []
    for ev in events:
        obj = ev.json
        typ = (obj or {}).get("type") or ev.event
        if typ:
            result.event_sequence.append(str(typ))
        if not obj:
            if ev.data == "[DONE]":
                result.upstream_completed_seen = True
            continue
        if typ == "response.created":
            response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
            result.upstream_response_id = response.get("id") or obj.get("response_id") or obj.get("id")
        elif typ == "response.output_item.added":
            item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
            result.function_call_seen = item.get("name") == cfg.function_name or cfg.function_name in json.dumps(obj, ensure_ascii=False)
        elif typ == "response.function_call_arguments.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                args_buffer.append(delta)
        elif typ == "response.function_call_arguments.done":
            final_args = obj.get("arguments") if isinstance(obj.get("arguments"), str) else "".join(args_buffer)
            result.raw_arguments = final_args
            result.args_done_seen = True
            result.answer, result.malformed_function_args, result.repaired_args, result.bridge_mode, result.tool_calls = parse_function_arguments(final_args, cfg, n.tool_registry)
            result.aborted = True
            break
        elif typ == "response.completed":
            result.upstream_completed_seen = True
            break
    return result

def audit_from_result(result: Bill015Result, n: NormalizedRequest, mode: str, fallback_used: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "local_request_id": result.local_request_id,
        "upstream_response_id": result.upstream_response_id,
        "mode": mode,
        "client_api": n.client_api,
        "model": n.model,
        "function_call_seen": result.function_call_seen,
        "args_done_seen": result.args_done_seen,
        "upstream_completed_seen": result.upstream_completed_seen,
        "aborted_at": "response.function_call_arguments.done" if result.aborted else None,
        "answer_chars": len(result.answer),
        "bridge_mode": result.bridge_mode,
        "tool_calls": [{"id": c.id, "name": c.name, "requested_name": c.requested_name, "type": c.call_type, "arguments_chars": len(c.arguments)} for c in result.tool_calls],
        "duration_ms": result.duration_ms,
        "fallback_used": fallback_used,
        "error": result.error,
        "event_sequence": result.event_sequence[-50:],
        "malformed_function_args": result.malformed_function_args,
        "repaired_args": result.repaired_args,
    }
    if settings.store_prompts:
        rec["prompt"] = n.user_input
        rec["instructions"] = n.instructions
    if settings.store_answers:
        rec["answer"] = result.answer
    if result.verify_delta is not None:
        rec["verify"] = {"pre": result.verify_pre, "post": result.verify_post, "delta": result.verify_delta}
    return rec


async def fetch_user_self(cfg: Settings = settings) -> dict[str, Any] | None:
    if not cfg.packy_cookie:
        return None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                cfg.upstream_base_url + "/api/user/self",
                headers={"Cookie": cfg.packy_cookie, "new-api-user": cfg.packy_user_id, "User-Agent": "bill015-local-proxy/1.0"},
            )
        obj = r.json()
        if isinstance(obj, dict) and obj.get("success") and isinstance(obj.get("data"), dict):
            d = obj["data"]
            return {"id": d.get("id"), "quota": d.get("quota"), "used_quota": d.get("used_quota"), "request_count": d.get("request_count")}
        return {"status": r.status_code, "body": obj}
    except Exception as e:
        return {"error": type(e).__name__}


def compute_delta(pre: dict[str, Any] | None, post: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pre or not post:
        return None
    out: dict[str, Any] = {}
    for k in ("quota", "used_quota", "request_count"):
        if isinstance(pre.get(k), (int, float)) and isinstance(post.get(k), (int, float)):
            out[k] = post[k] - pre[k]
    return out or None


async def execute_bill015(n: NormalizedRequest, mode: str, cfg: Settings = settings, local_request_id: str | None = None) -> Bill015Result:
    request_id = local_request_id or local_response_id()
    result = Bill015Result(local_request_id=request_id)
    start = time.perf_counter()
    if mode == "verify":
        result.verify_pre = await fetch_user_self(cfg)
    if not cfg.upstream_api_key:
        raise HTTPException(status_code=500, detail=f"Missing upstream API key env {cfg.upstream_api_key_env}")

    payload = build_bill015_payload(n, cfg)
    args_buffer: list[str] = []
    timeout = httpx.Timeout(cfg.upstream_timeout_seconds, read=cfg.upstream_idle_timeout_ms / 1000)
    headers = {
        "Authorization": "Bearer " + cfg.upstream_api_key,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "bill015-local-proxy/1.0",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", cfg.upstream_base_url + "/v1/responses", headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise HTTPException(status_code=502, detail={"upstream_status": resp.status_code, "body": body[:2000].decode("utf-8", errors="replace")})
                async for ev in parse_async_sse_lines(resp.aiter_lines()):
                    obj = ev.json
                    typ = (obj or {}).get("type") or ev.event
                    if typ:
                        result.event_sequence.append(str(typ))
                    if not obj:
                        if ev.data == "[DONE]":
                            result.upstream_completed_seen = True
                        continue
                    if typ == "response.created":
                        response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
                        result.upstream_response_id = response.get("id") or obj.get("response_id") or obj.get("id")
                    elif typ == "response.output_item.added":
                        item = obj.get("item") if isinstance(obj.get("item"), dict) else {}
                        if item.get("type") in {"function_call", "tool_call"} or item.get("name") == cfg.function_name:
                            result.function_call_seen = True
                        else:
                            result.function_call_seen = result.function_call_seen or (cfg.function_name in json.dumps(obj, ensure_ascii=False))
                    elif typ == "response.function_call_arguments.delta":
                        delta = obj.get("delta")
                        if isinstance(delta, str):
                            args_buffer.append(delta)
                    elif typ == "response.function_call_arguments.done":
                        final_args = obj.get("arguments") if isinstance(obj.get("arguments"), str) else "".join(args_buffer)
                        result.raw_arguments = final_args
                        result.args_done_seen = True
                        result.answer, result.malformed_function_args, result.repaired_args, result.bridge_mode, result.tool_calls = parse_function_arguments(final_args, cfg, n.tool_registry)
                        await resp.aclose()
                        result.aborted = True
                        break
                    elif typ == "response.completed":
                        result.upstream_completed_seen = True
                        break
                    elif typ in {"response.failed", "error"}:
                        raise RuntimeError(json.dumps(obj, ensure_ascii=False)[:2000])
        if mode == "verify":
            await asyncio.sleep(0.5)
            result.verify_post = await fetch_user_self(cfg)
            result.verify_delta = compute_delta(result.verify_pre, result.verify_post)
        if not result.args_done_seen:
            raise RuntimeError("upstream stream ended before response.function_call_arguments.done")
        return result
    except HTTPException:
        raise
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=502, detail=result.error)
    finally:
        result.duration_ms = int((time.perf_counter() - start) * 1000)


def dry_run_response(n: NormalizedRequest, mode: str = "dry-run") -> dict[str, Any]:
    payload = build_bill015_payload(n)
    safe_payload = redact(payload)
    # Avoid returning full prompt unless explicitly enabled.
    if not settings.store_prompts:
        for item in safe_payload.get("input", []):
            if isinstance(item, dict) and item.get("role") == "user":
                item["content"] = f"<redacted prompt chars={len(n.user_input)}>"
    return {
        "id": local_response_id(),
        "object": "response",
        "status": "completed",
        "mode": mode,
        "would_post": settings.upstream_base_url + "/v1/responses",
        "upstream_configured": settings.upstream_configured,
        "payload": safe_payload,
    }


def _prepend_instruction(existing: Any, extra: str) -> str:
    if isinstance(existing, str) and existing.strip():
        if extra in existing:
            return existing
        return extra + "\n\n" + existing
    return extra


def _compress_data_url(value: str, max_bytes: int = 900_000) -> str:
    if not value.startswith("data:image/") or ";base64," not in value or len(value) <= max_bytes:
        return value
    header, b64 = value.split(",", 1)
    try:
        raw = base64.b64decode(b64, validate=False)
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return "[image omitted by local proxy: data URL too large and Pillow unavailable]"
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((1280, 1280))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=72, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        return "[image omitted by local proxy: failed to compress large data URL]"


def _shrink_media(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _shrink_media(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shrink_media(v) for v in obj]
    if isinstance(obj, str):
        return _compress_data_url(obj)
    return obj


def prepare_passthrough_payload(body: dict[str, Any], cfg: Settings = settings) -> dict[str, Any]:
    payload = copy.deepcopy(body)
    payload["model"] = cfg.default_model
    payload["instructions"] = _prepend_instruction(payload.get("instructions"), IDENTITY_INSTRUCTION)
    if "reasoning" not in payload and cfg.reasoning_effort:
        payload["reasoning"] = {"effort": cfg.reasoning_effort, "summary": cfg.reasoning_summary}
    return _shrink_media(payload)

async def normal_forward_stream(body: dict[str, Any], cfg: Settings = settings) -> AsyncIterator[bytes]:
    if not cfg.upstream_api_key:
        yield encode_sse({"type": "error", "error": {"message": f"Missing upstream API key env {cfg.upstream_api_key_env}"}}, "error")
        yield b"data: [DONE]\n\n"
        return
    headers = {"Authorization": "Bearer " + cfg.upstream_api_key, "Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": "bill015-local-proxy/1.0"}
    timeout = httpx.Timeout(cfg.upstream_timeout_seconds, read=cfg.upstream_idle_timeout_ms / 1000)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", cfg.upstream_base_url + "/v1/responses", headers=headers, json=prepare_passthrough_payload(body, cfg)) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk


async def normal_forward_json(body: dict[str, Any], cfg: Settings = settings) -> dict[str, Any]:
    if not cfg.upstream_api_key:
        raise HTTPException(status_code=500, detail=f"Missing upstream API key env {cfg.upstream_api_key_env}")
    headers = {"Authorization": "Bearer " + cfg.upstream_api_key, "Content-Type": "application/json", "User-Agent": "bill015-local-proxy/1.0"}
    async with httpx.AsyncClient(timeout=cfg.upstream_timeout_seconds) as client:
        r = await client.post(cfg.upstream_base_url + "/v1/responses", headers=headers, json=prepare_passthrough_payload(body, cfg))
    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=502, detail={"upstream_status": r.status_code, "body": r.text[:2000]})


def chat_json(result: Bill015Result, n: NormalizedRequest) -> dict[str, Any]:
    return {
        "id": "chatcmpl-local-" + result.local_request_id.removeprefix("resp_local_"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": n.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.answer}, "finish_reason": "stop"}],
        "usage": chat_usage(n, answer=result.answer),
    }


