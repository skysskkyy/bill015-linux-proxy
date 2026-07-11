from __future__ import annotations

import hashlib
import re
from http import HTTPStatus
from typing import Any

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


def _reason_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except Exception:
        return "upstream error"


def _decode_body(body: bytes | str) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)


def _compact_text(value: str, limit: int = 300) -> str:
    text = _WS_RE.sub(" ", value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _html_title(text: str) -> str:
    m = _TITLE_RE.search(text)
    if not m:
        return ""
    return _compact_text(_HTML_TAG_RE.sub(" ", m.group(1)), 160)


def _html_to_text_preview(text: str) -> str:
    return _compact_text(_HTML_TAG_RE.sub(" ", text), 300)


def is_html_body(text: str, content_type: str = "") -> bool:
    ctype = content_type.lower()
    prefix = text[:500].lower()
    return "text/html" in ctype or "<!doctype html" in prefix or "<html" in prefix or "<title" in prefix


def sanitize_upstream_error_detail(
    status: int,
    body: bytes | str | dict[str, Any] | None,
    *,
    content_type: str = "",
    preserve_json_body: bool = False,
) -> dict[str, Any]:
    """Return a compact upstream error detail safe for UI/SSE logs.

    HTML error pages (Cloudflare 520 etc.) are summarized and hashed instead of
    being embedded verbatim, which prevents Codex Desktop from rendering a huge
    `stream disconnected before completion: {...body: <html>...}` message.
    """
    detail: dict[str, Any] = {"upstream_status": status}
    if isinstance(body, dict):
        message = ""
        err = body.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or "")
        if not message:
            message = str(body.get("message") or "")
        detail["message"] = message or f"upstream returned HTTP {status}: {_reason_phrase(status)}"
        if preserve_json_body:
            detail["body"] = body
        return detail

    text = _decode_body(body or "")
    title = _html_title(text) if is_html_body(text, content_type) else ""
    if title:
        message = f"upstream returned HTTP {status}: {title}"
    else:
        preview = _compact_text(text, 160)
        message = f"upstream returned HTTP {status}: {preview or _reason_phrase(status)}"
    detail.update(
        {
            "message": message,
            "body_preview": _html_to_text_preview(text) if is_html_body(text, content_type) else _compact_text(text, 300),
            "body_chars": len(text),
            "body_sha256_16": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16],
        }
    )
    if content_type:
        detail["content_type"] = content_type
    return detail


def error_message_from_detail(detail: Any) -> str:
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        status = detail.get("upstream_status")
        if status:
            return f"upstream returned HTTP {status}: {_reason_phrase(int(status)) if isinstance(status, int) else 'upstream error'}"
        err = detail.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
    text = str(detail)
    if is_html_body(text):
        title = _html_title(text)
        return title or _html_to_text_preview(text)
    return _compact_text(text, 1000)
