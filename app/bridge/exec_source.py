from __future__ import annotations

import json
import re

UNWRAP_KEYS = ("input", "code", "source", "script", "javascript", "js")
CONTROL_HEAD = re.compile(
    r"^(?:const|let|var|if|for|while|switch|try|class|function|async\s+function|"
    r"return|throw|break|continue|export|import|exit)\b"
)
FENCE = re.compile(r"^```(?:javascript|js|typescript|ts)?\s*\r?\n(.*)\r?\n```\s*$", re.IGNORECASE | re.DOTALL)
PRAGMA_PREFIX = "// @exec:"
WRAP_MARK = "/*__bill015_exec*/"


def is_code_mode_exec(name: str, namespace: str | None) -> bool:
    if (name or "").strip() != "exec":
        return False
    ns = (namespace or "").strip()
    return not ns.startswith("mcp__")


def unwrap_exec_source(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff").strip()
    for _ in range(6):
        if not text:
            return text
        fenced = FENCE.match(text)
        if fenced:
            text = fenced.group(1).strip()
            continue
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            try:
                unquoted = json.loads(text)
            except Exception:
                unquoted = None
            if isinstance(unquoted, str) and unquoted.strip() and unquoted.strip() != text:
                text = unquoted.strip()
                continue
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
            except Exception:
                break
            if not isinstance(obj, dict):
                break
            nxt = None
            for key in UNWRAP_KEYS:
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    nxt = val.strip()
                    break
            if nxt is None:
                break
            text = nxt
            continue
        break
    return text


def _split_pragma(src: str) -> tuple[str, str]:
    lines = src.splitlines()
    if not lines:
        return "", src
    first = lines[0].lstrip()
    if first.startswith(PRAGMA_PREFIX):
        return lines[0], "\n".join(lines[1:]).strip()
    return "", src


def _scan_depth(src: str) -> list[tuple[str, int]]:
    """Yield (char, brace/paren/bracket depth after this char) for statement splitting."""
    out: list[tuple[str, int]] = []
    depth = 0
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in "'\"`":
            quote = ch
            out.append((ch, depth))
            i += 1
            while i < n:
                cur = src[i]
                out.append((cur, depth))
                if cur == "\\" and i + 1 < n:
                    out.append((src[i + 1], depth))
                    i += 2
                    continue
                if cur == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out.append((src[i], depth))
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            out.append((ch, depth))
            out.append((src[i + 1], depth))
            i += 2
            while i < n - 1 and not (src[i] == "*" and src[i + 1] == "/"):
                out.append((src[i], depth))
                i += 1
            continue
        if ch in "{[(":
            depth += 1
        elif ch in "}])" and depth:
            depth -= 1
        out.append((ch, depth))
        i += 1
    return out


def _has_depth0_return(src: str) -> bool:
    scanned = _scan_depth(src)
    token: list[str] = []
    depth = 0
    for ch, depth in scanned:
        if ch.isalnum() or ch == "_":
            token.append(ch)
            continue
        word = "".join(token)
        token = []
        if word == "return" and depth == 0:
            return True
    return "".join(token) == "return" and depth == 0


def _statements(src: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for ch, depth in _scan_depth(src):
        buf.append(ch)
        if ch == ";" and depth == 0:
            parts.append("".join(buf))
            buf = []
    if buf:
        parts.append("".join(buf))
    return parts


def _return_last_expression(src: str) -> str:
    stmts = _statements(src)
    nonempty = [stmt for stmt in stmts if stmt.strip()]
    if not nonempty:
        return src
    last = nonempty[-1]
    stripped = last.strip().rstrip(";").strip()
    if not stripped or stripped.startswith("{"):
        return src
    if CONTROL_HEAD.match(stripped):
        return src
    prefix = "".join(stmts[: stmts.index(last)]) if last in stmts else "".join(nonempty[:-1])
    spacer = "" if not prefix or prefix.endswith(("\n", " ", "\t")) else " "
    return f"{prefix}{spacer}return {stripped};"


def normalize_exec_source(raw: str) -> str:
    text = unwrap_exec_source(raw)
    if not text.strip():
        return text
    if WRAP_MARK in text:
        return text
    pragma, body = _split_pragma(text)
    body = body.strip()
    if not body:
        return text
    if not _has_depth0_return(body):
        body = _return_last_expression(body)
    wrapped = (
        f"{WRAP_MARK}\n"
        "await (async () => {\n"
        f"{body}\n"
        "})().then((v) => { if (v !== undefined && v !== null) text(v); });"
    )
    if pragma:
        return f"{pragma}\n{wrapped}"
    return wrapped
