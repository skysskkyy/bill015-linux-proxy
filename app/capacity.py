from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .state import RuntimeState


class QueueFullError(Exception):
    pass


class QueueWaitTimeoutError(Exception):
    pass


@dataclass
class CapacityLease:
    limiter: "CapacityLimiter"
    queue_wait_ms: int
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        await self.limiter.release()


class CapacityLimiter:
    """Bound active work and the number of coroutines allowed to queue."""

    def __init__(self, max_active: int, max_queue: int, queue_timeout_ms: int, state: RuntimeState) -> None:
        self.max_active = max(1, int(max_active))
        self.max_queue = max(0, int(max_queue))
        self.queue_timeout_seconds = max(0.001, int(queue_timeout_ms) / 1000)
        self._workers = asyncio.Semaphore(self.max_active)
        self._admission_lock = asyncio.Lock()
        self._admitted = 0
        self._state = state

    async def acquire(self) -> CapacityLease:
        async with self._admission_lock:
            if self._admitted >= self.max_active + self.max_queue:
                self._state.mark_busy_rejected()
                raise QueueFullError
            self._admitted += 1
            self._state.mark_queued()

        started = time.perf_counter()
        try:
            await asyncio.wait_for(self._workers.acquire(), timeout=self.queue_timeout_seconds)
        except TimeoutError as exc:
            async with self._admission_lock:
                self._admitted = max(0, self._admitted - 1)
            self._state.mark_queue_left()
            self._state.mark_queue_timeout()
            raise QueueWaitTimeoutError from exc
        except BaseException:
            async with self._admission_lock:
                self._admitted = max(0, self._admitted - 1)
            self._state.mark_queue_left()
            raise

        self._state.mark_queue_left(active=True)
        return CapacityLease(self, int((time.perf_counter() - started) * 1000))

    async def release(self) -> None:
        self._workers.release()
        async with self._admission_lock:
            self._admitted = max(0, self._admitted - 1)
        self._state.mark_active_finished()
