from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict


@dataclass
class RuntimeState:
    started_at: int = field(default_factory=lambda: int(time.time()))
    request_count: int = 0
    success_count: int = 0
    args_done_count: int = 0
    abort_count: int = 0
    fallback_count: int = 0
    error_count: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    current_mode_override: str | None = None
    lock: Lock = field(default_factory=Lock)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "started_at": self.started_at,
                "request_count": self.request_count,
                "success_count": self.success_count,
                "args_done_count": self.args_done_count,
                "abort_count": self.abort_count,
                "fallback_count": self.fallback_count,
                "error_count": self.error_count,
                "consecutive_failures": self.consecutive_failures,
                "last_error": self.last_error,
                "mode_override": self.current_mode_override,
            }

    def inc_request(self) -> None:
        with self.lock:
            self.request_count += 1

    def mark_success(self, args_done: bool = False, aborted: bool = False) -> None:
        with self.lock:
            self.success_count += 1
            self.consecutive_failures = 0
            if args_done:
                self.args_done_count += 1
            if aborted:
                self.abort_count += 1

    def mark_error(self, err: str) -> None:
        with self.lock:
            self.error_count += 1
            self.consecutive_failures += 1
            self.last_error = err[:500]

    def mark_fallback(self) -> None:
        with self.lock:
            self.fallback_count += 1


runtime_state = RuntimeState()
