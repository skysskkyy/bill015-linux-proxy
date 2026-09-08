from __future__ import annotations

import json
from typing import Any

from ..config import Settings, settings
from ..protocol.models import Turn
from .catalog import build_catalog
from .images import sanitize_images
from .text import flatten_content, item_text, last_user_text
from .web_intent import detect_web_intent


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": value}]}]
    return []


def _strip_additional_tools(items: list[Any]) -> tuple[list[dict[str, Any]], str]:
    kept: list[dict[str, Any]] = []
    developer_bits: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") == "additional_tools":
            extra = flatten_content(item.get("content") or item.get("text") or "")
            if extra.strip():
                developer_bits.append(extra.strip())
            continue
        kept.append(item)
    return kept, "\n".join(developer_bits)


def _role_texts(items: list[dict[str, Any]]) -> tuple[str, str]:
    system_parts: list[str] = []
    developer_parts: list[str] = []
    for item in items:
        role = str(item.get("role") or "")
        text = item_text(item).strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "developer":
            developer_parts.append(text)
    return "\n".join(system_parts), "\n".join(developer_parts)


def parse_codex_turn_metadata(client_metadata: Any) -> dict[str, Any]:
    if not isinstance(client_metadata, dict):
        return {}
    raw = client_metadata.get("x-codex-turn-metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def extract_reasoning(body: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("reasoning", "model_reasoning"):
        value = body.get(key)
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            return {"effort": value.strip()}
    effort = body.get("model_reasoning_effort") or body.get("reasoning_effort")
    summary = body.get("model_reasoning_summary") or body.get("reasoning_summary")
    meta = parse_codex_turn_metadata(body.get("client_metadata"))
    effort = effort or meta.get("model_reasoning_effort") or meta.get("reasoning_effort")
    summary = summary or meta.get("model_reasoning_summary") or meta.get("reasoning_summary")
    if effort or summary:
        out: dict[str, Any] = {}
        if effort:
            out["effort"] = str(effort)
        if summary:
            out["summary"] = str(summary)
        return out
    return None


def _is_lite(body: dict[str, Any], items: list[Any], headers: dict[str, str] | None) -> bool:
    header_blob = " ".join((headers or {}).values()).lower()
    if "codex-responses-lite" in header_blob:
        return True
    if any(isinstance(item, dict) and str(item.get("type") or "") == "additional_tools" for item in items):
        return True
    meta = parse_codex_turn_metadata(body.get("client_metadata"))
    flag = meta.get("responses_lite") or (body.get("client_metadata") or {}).get("responses_lite") if isinstance(body.get("client_metadata"), dict) else None
    return str(flag).lower() in {"1", "true", "yes"}


def _request_kind(body: dict[str, Any]) -> tuple[str, bool]:
    meta = parse_codex_turn_metadata(body.get("client_metadata"))
    kind = str(meta.get("request_kind") or body.get("request_kind") or "turn")
    compaction = kind == "compaction" or bool(meta.get("compaction"))
    return kind, compaction


def normalize_responses_request(body: dict[str, Any], request_headers: dict[str, str] | None = None, cfg: Settings = settings) -> Turn:
    cleaned, image_replacements = sanitize_images(body, cfg)
    items = _as_items(cleaned.get("input"))
    typed_items, extra_dev = _strip_additional_tools(items)
    system_text, developer_text = _role_texts(typed_items)
    if extra_dev:
        developer_text = (developer_text + "\n" + extra_dev).strip()
    current_user = last_user_text(typed_items) or flatten_content(cleaned.get("input"))
    catalog = build_catalog(cleaned.get("tools"), items, query=current_user, cfg=cfg)
    kind, is_compaction = _request_kind(cleaned)
    reasoning = extract_reasoning(cleaned)
    model = cfg.map_model(cleaned.get("model"))
    headers = {str(k).lower(): str(v) for k, v in (request_headers or {}).items()}
    return Turn(
        model=model,
        original_model=str(cleaned.get("model") or "") or None,
        want_stream=bool(cleaned.get("stream", True)),
        client_api="responses",
        items=typed_items,
        current_user=current_user,
        catalog=catalog,
        instructions=str(cleaned.get("instructions") or ""),
        system_text=system_text,
        developer_text=developer_text,
        reasoning=reasoning,
        request_headers=headers,
        max_output_tokens=cleaned.get("max_output_tokens"),
        temperature=cleaned.get("temperature"),
        metadata=cleaned.get("metadata") if isinstance(cleaned.get("metadata"), dict) else None,
        client_metadata=cleaned.get("client_metadata") if isinstance(cleaned.get("client_metadata"), dict) else None,
        request_kind=kind,
        is_compaction=is_compaction,
        responses_lite=_is_lite(cleaned, items, headers),
        raw_body=cleaned,
        image_replacements=image_replacements,
        web_intent=detect_web_intent(current_user),
    )


def normalize_chat_request(body: dict[str, Any], request_headers: dict[str, str] | None = None, cfg: Settings = settings) -> Turn:
    cleaned, image_replacements = sanitize_images(body, cfg)
    messages = cleaned.get("messages") if isinstance(cleaned.get("messages"), list) else []
    items: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        items.append({"type": "message", "role": role, "content": [{"type": "input_text", "text": flatten_content(msg.get("content"))}]})
    current_user = last_user_text(items)
    catalog = build_catalog(cleaned.get("tools"), items, query=current_user, cfg=cfg)
    return Turn(
        model=cfg.map_model(cleaned.get("model")),
        original_model=str(cleaned.get("model") or "") or None,
        want_stream=bool(cleaned.get("stream", False)),
        client_api="chat.completions",
        items=items,
        current_user=current_user,
        catalog=catalog,
        instructions=str(cleaned.get("instructions") or ""),
        request_headers={str(k).lower(): str(v) for k, v in (request_headers or {}).items()},
        max_output_tokens=cleaned.get("max_tokens") or cleaned.get("max_output_tokens"),
        temperature=cleaned.get("temperature"),
        raw_body=cleaned,
        image_replacements=image_replacements,
        web_intent=detect_web_intent(current_user),
    )


def request_needs_passthrough(body: dict[str, Any]) -> tuple[bool, str | None]:
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and str(tool.get("type") or "") in {"web_search", "file_search", "mcp"}:
                if str(tool.get("type") or "") == "web_search":
                    return True, "hosted web_search"
    return False, None
