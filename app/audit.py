from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from .config import settings

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.I),
    re.compile(r"(?i)(cookie\s*[:=]\s*)[^\s,;]{12,}"),
]

SECRET_KEY_NAMES = {
    "authorization",
    "cookie",
    "api_key",
    "api_keys",
    "token",
    "access_token",
    "refresh_token",
    "session",
    "secret",
    "password",
    "admin_token",
}


def _is_secret_key(key: str) -> bool:
    key = str(key or "").lower()
    return (
        key in SECRET_KEY_NAMES
        or key.endswith("_token")
        or key.endswith("_secret")
        or key.endswith("_password")
        or key.endswith("_cookie")
    )


def redact(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: ("<redacted>" if _is_secret_key(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if not isinstance(value, str):
        return value
    out = value
    for pat in SECRET_PATTERNS:
        out = pat.sub(lambda m: (m.group(1) if m.lastindex else "") + "<redacted>", out)
    return out


class AuditLogger:
    def __init__(self, directory: Path, rotate_mb: int = 10):
        self.directory = directory
        self.path = directory / "audit.jsonl"
        self.rotate_bytes = max(1, rotate_mb) * 1024 * 1024
        self._lock = Lock()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _rotate_if_needed(self) -> None:
        if not self.path.exists() or self.path.stat().st_size < self.rotate_bytes:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path.rename(self.directory / f"audit_{stamp}.jsonl")

    def write(self, record: Dict[str, Any]) -> None:
        safe = redact(record)
        safe.setdefault("ts", int(time.time()))
        line = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._rotate_if_needed()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + os.linesep)

    def recent(self, n: int = 20) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(n, 200)):]
            return [json.loads(x) for x in lines if x.strip()]
        except Exception:
            return []


audit_logger = AuditLogger(settings.evidence_dir, settings.rotate_mb)
