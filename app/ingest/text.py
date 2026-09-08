from __future__ import annotations

from typing import Any


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "output_text", "input", "content", "output"):
            if key in content:
                return flatten_content(content.get(key))
        return ""
    if isinstance(content, list):
        parts = [flatten_content(part) for part in content]
        return "\n".join(part for part in parts if part)
    return str(content)


def item_type(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("type") or ("message" if item.get("role") else ""))


def item_text(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item or "")
    if item.get("content") is not None:
        return flatten_content(item.get("content"))
    if item.get("text") is not None:
        return flatten_content(item.get("text"))
    if item.get("output") is not None:
        return flatten_content(item.get("output"))
    if item.get("input") is not None:
        return flatten_content(item.get("input"))
    if item.get("arguments") is not None:
        return flatten_content(item.get("arguments"))
    return ""


def last_user_text(items: list[Any]) -> str:
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "") == "user":
            text = item_text(item).strip()
            if text:
                return text
        if item_type(item) == "message" and str(item.get("role") or "") == "user":
            text = item_text(item).strip()
            if text:
                return text
    return ""
