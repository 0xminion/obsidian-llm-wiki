"""One shared asynchronous remote-request concurrency/rate limiter."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

__all__ = ["AsyncRequestLimiter", "ThreadsafeRequestGate"]


class AsyncRequestLimiter:
    """Bound concurrent starts across every nested phase of one pipeline run."""

    def __init__(self, concurrency: int, *, min_interval_seconds: float = 0.0) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must not be negative")
        self.concurrency = concurrency
        self.min_interval_seconds = min_interval_seconds
        self._semaphore = asyncio.Semaphore(concurrency)
        self._start_lock = asyncio.Lock()
        self._next_start = 0.0
        self._active = 0
        self._max_active = 0
        self.requests_started = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def max_active(self) -> int:
        return self._max_active

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire shared backpressure and optional inter-request pacing."""
        async with self._semaphore:
            if self.min_interval_seconds:
                async with self._start_lock:
                    now = time.monotonic()
                    delay = self._next_start - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                    self._next_start = time.monotonic() + self.min_interval_seconds
            self._active += 1
            self._max_active = max(self._max_active, self._active)
            self.requests_started += 1
            try:
                yield
            finally:
                self._active -= 1


class ThreadsafeRequestGate:
    """Expose one async limiter to synchronous requests running in worker threads."""

    def __init__(self, limiter: AsyncRequestLimiter, loop: asyncio.AbstractEventLoop) -> None:
        self.limiter = limiter
        self.loop = loop

    @contextmanager
    def slot(self) -> Iterator[None]:
        context: Any = self.limiter.slot()
        asyncio.run_coroutine_threadsafe(context.__aenter__(), self.loop).result()
        try:
            yield
        finally:
            asyncio.run_coroutine_threadsafe(
                context.__aexit__(None, None, None), self.loop
            ).result()
