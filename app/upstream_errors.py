from __future__ import annotations

import hashlib
import json
import re
from http import HTTPStatus
from typing import Any

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
CYBER_POLICY_ERROR_CODE = "cyber_policy"
CYBER_POLICY_ERROR_MESSAGE = (
    "This content was flagged for possible cybersecurity risk. "
    "If this seems wrong, try rephrasing your request. "
    "To get authorized for security work, join the Trusted Access for Cyber program: "
    "https://chatgpt.com/cyber"
)


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


def _decode_json_or_sse_values(value: bytes | str) -> list[Any]:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    try:
        return [json.loads(text)]
    except Exception:
        values: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                values.append(json.loads(payload))
            except Exception:
                continue
        return values


def _normalize_message(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip()


def _is_cyber_policy_error_object(value: dict[str, Any]) -> bool:
    """Match the upstream cyber-policy error shape observed in audit logs.

    Logs show SSE error objects like:
    ``{"type":"error","error":{"type":"invalid_request","code":"cyber_policy","message":"..."}}``.
    Generic sequence numbers or generic rephrase text must not trigger key rotation.
    """
    code = str(value.get("code") or "").strip().lower()
    message = _normalize_message(value.get("message"))
    return code == CYBER_POLICY_ERROR_CODE and message == CYBER_POLICY_ERROR_MESSAGE


def is_cyber_policy_rotation_error(value: Any) -> bool:
    if isinstance(value, (bytes, str)):
        return any(is_cyber_policy_rotation_error(item) for item in _decode_json_or_sse_values(value))
    if isinstance(value, list):
        return any(is_cyber_policy_rotation_error(item) for item in value)
    if not isinstance(value, dict):
        return False

    err = value.get("error")
    if isinstance(err, dict) and _is_cyber_policy_error_object(err):
        typ = str(value.get("type") or "").lower()
        return typ in {"", "error", "response.failed", "failed"} or "response" not in value

    response = value.get("response")
    if isinstance(response, dict):
        response_error = response.get("error")
        if isinstance(response_error, dict) and _is_cyber_policy_error_object(response_error):
            typ = str(value.get("type") or "").lower()
            return typ in {"response.failed", "failed", "error"}

    return _is_cyber_policy_error_object(value)


def key_rotation_error_reason(value: Any) -> str | None:
    """Return the configured key-rotation reason for an upstream error payload."""
    return CYBER_POLICY_ERROR_CODE if is_cyber_policy_rotation_error(value) else None

def is_context_length_exceeded(value: Any) -> bool:
    if isinstance(value, (bytes, str)):
        parsed = _decode_json_or_sse_values(value)
        if parsed:
            return any(is_context_length_exceeded(item) for item in parsed)
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return "context_length_exceeded" in text.lower()
    if isinstance(value, list):
        return any(is_context_length_exceeded(item) for item in value)
    if not isinstance(value, dict):
        return False
    if str(value.get("code") or "").lower() == "context_length_exceeded":
        return True
    return any(is_context_length_exceeded(item) for item in value.values())

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
