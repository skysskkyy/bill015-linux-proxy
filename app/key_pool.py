from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock
from typing import Iterable

from .config import Settings, settings


@dataclass(frozen=True)
class ApiKeySelection:
    key: str
    index: int
    count: int
    fingerprint: str


class ApiKeyPool:
    """Process-wide, concurrency-safe ordered API key pool."""

    def __init__(self, keys: Iterable[str]):
        self._keys = tuple(dict.fromkeys(str(key).strip() for key in keys if str(key).strip()))
        self._index = 0
        self._lock = Lock()

    @staticmethod
    def _fingerprint(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]

    def _selection(self, index: int) -> ApiKeySelection:
        key = self._keys[index]
        return ApiKeySelection(key=key, index=index, count=len(self._keys), fingerprint=self._fingerprint(key))

    def current(self) -> ApiKeySelection | None:
        with self._lock:
            if not self._keys:
                return None
            return self._selection(self._index)

    def rotate_after_failure(self, failed_key: str, tried_keys: set[str] | None = None) -> ApiKeySelection | None:
        """Advance once for a failed current key, without double-skipping under concurrency."""
        tried = set(tried_keys or ())
        with self._lock:
            if len(self._keys) <= 1:
                return None
            current_key = self._keys[self._index]
            start = (self._index + 1) % len(self._keys) if current_key == failed_key else self._index
            for offset in range(len(self._keys)):
                index = (start + offset) % len(self._keys)
                candidate = self._keys[index]
                if candidate == failed_key or candidate in tried:
                    continue
                self._index = index
                return self._selection(index)
            return None

    def snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            if not self._keys:
                return {"key_count": 0, "current_key_index": None, "current_key_fingerprint": None}
            key = self._keys[self._index]
            return {
                "key_count": len(self._keys),
                "current_key_index": self._index + 1,
                "current_key_fingerprint": self._fingerprint(key),
            }


_pool_lock = Lock()
_pool_signature: tuple[str, ...] = ()
_pool: ApiKeyPool | None = None


def upstream_key_pool(cfg: Settings = settings) -> ApiKeyPool:
    global _pool, _pool_signature
    signature = tuple(cfg.upstream_api_keys)
    with _pool_lock:
        if _pool is None or signature != _pool_signature:
            _pool = ApiKeyPool(signature)
            _pool_signature = signature
        return _pool
