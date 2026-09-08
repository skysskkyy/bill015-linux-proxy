from __future__ import annotations

import json
import re
from http import HTTPStatus
from typing import Any

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_WS_RE = re.compile(r"\s+")
CYBER_POLICY_ERROR_CODE = "cyber_policy"
CYBER_POLICY_ERROR_MESSAGE = (
    "This content was flagged for possible cybersecurity risk. "
    "If this seems wrong, try rephrasing your request. "
    "To get authorized for security work, join the Trusted Access for Cyber program: "
    "https://chatgpt.com/cyber"
)


def _reason(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except Exception:
        return "upstream error"


def _decode(body: bytes | str) -> str:
    return body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)


def _compact(value: str, limit: int = 300) -> str:
    text = _WS_RE.sub(" ", value).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def is_html_body(text: str, content_type: str = "") -> bool:
    ctype = content_type.lower()
    prefix = text[:500].lower()
    return "text/html" in ctype or "<!doctype html" in prefix or "<html" in prefix


def sanitize_upstream_error_detail(
    status: int,
    body: bytes | str | dict[str, Any] | None,
    *,
    content_type: str = "",
) -> dict[str, Any]:
    detail: dict[str, Any] = {"upstream_status": status}
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else {}
        message = str((err or {}).get("message") or body.get("message") or "")
        detail["message"] = message or f"upstream returned HTTP {status}: {_reason(status)}"
        if err:
            detail["code"] = err.get("code")
        return detail
    text = _decode(body or b"")
    if is_html_body(text, content_type):
        title = _TITLE_RE.search(text)
        detail["message"] = _compact(title.group(1) if title else _reason(status))
        return detail
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return sanitize_upstream_error_detail(status, obj, content_type=content_type)
    except Exception:
        pass
    detail["message"] = _compact(text) or f"upstream returned HTTP {status}: {_reason(status)}"
    return detail


def _error_fields(payload: Any) -> tuple[str, str]:
    if isinstance(payload, bytes):
        payload = _decode(payload)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return "", payload
    if not isinstance(payload, dict):
        return "", ""
    err = payload.get("error") if isinstance(payload.get("error"), dict) else payload
    if not isinstance(err, dict):
        return "", ""
    return str(err.get("code") or ""), str(err.get("message") or "")


def is_cyber_policy_rotation_error(payload: Any) -> bool:
    code, message = _error_fields(payload)
    return code == CYBER_POLICY_ERROR_CODE and message.strip() == CYBER_POLICY_ERROR_MESSAGE


def key_rotation_error_reason(payload: Any) -> str | None:
    if is_cyber_policy_rotation_error(payload):
        return "cyber_policy"
    return None


def is_context_length_exceeded(payload: Any) -> bool:
    blob = payload
    if isinstance(blob, (bytes, str, dict)):
        code, message = _error_fields(blob)
        text = f"{code} {message}".lower()
        return "context_length_exceeded" in text or "maximum context length" in text
    return False


def error_message_from_detail(detail: Any) -> str:
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail)
