"""Per-extraction side-effect policy, propagated explicitly to worker threads."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_WRITES_ALLOWED: ContextVar[bool] = ContextVar("extraction_writes_allowed", default=True)


def writes_allowed() -> bool:
    """Whether this extraction may persist caches or usage records."""
    return _WRITES_ALLOWED.get()


@contextmanager
def extraction_policy(*, allow_writes: bool) -> Iterator[None]:
    token = _WRITES_ALLOWED.set(allow_writes)
    try:
        yield
    finally:
        _WRITES_ALLOWED.reset(token)
