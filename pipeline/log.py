"""Structured logging with correlation IDs.

Provides a context-variable-based correlation system so that a single URL
can be traced through Extract -> Plan -> Create -> Compile.

Usage:
    from pipeline.log import get_logger, set_correlation

    log = get_logger(__name__)
    set_correlation(batch_id="abc123", source_hash="def456")
    log.info("Processing source")  # output includes [abc123/def456]
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import time
from typing import Generator, Optional

_batch_id: contextvars.ContextVar[str] = contextvars.ContextVar("batch_id", default="")
_source_hash: contextvars.ContextVar[str] = contextvars.ContextVar("source_hash", default="")
_stage: contextvars.ContextVar[str] = contextvars.ContextVar("stage", default="")


def set_correlation(
    batch_id: Optional[str] = None,
    source_hash: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    if batch_id is not None:
        _batch_id.set(batch_id)
    if source_hash is not None:
        _source_hash.set(source_hash)
    if stage is not None:
        _stage.set(stage)


def clear_correlation() -> None:
    _batch_id.set("")
    _source_hash.set("")
    _stage.set("")


@contextlib.contextmanager
def stage_timer(stage_name: str, logger: Optional[logging.Logger] = None) -> Generator[None, None, None]:
    """Context manager that sets the stage correlation and logs elapsed time."""
    log = logger or logging.getLogger("pipeline")
    _stage.set(stage_name)
    log.info("=== Stage %s started ===", stage_name)
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        log.info("=== Stage %s finished (%.1fs) ===", stage_name, elapsed)
        _stage.set("")


class CorrelationFilter(logging.Filter):
    """Injects batch_id, source_hash, and stage into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.batch_id = _batch_id.get("")  # type: ignore[attr-defined]
        record.source_hash = _source_hash.get("")  # type: ignore[attr-defined]
        record.stage = _stage.get("")  # type: ignore[attr-defined]
        return True


_CORRELATED_FORMAT = "%(asctime)s [%(levelname)s] %(correlation)s%(name)s: %(message)s"


class CorrelationFormatter(logging.Formatter):
    """Formatter that prepends [batch_id/source_hash] when available."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt or _CORRELATED_FORMAT, datefmt=datefmt or "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        batch = getattr(record, "batch_id", "")
        source = getattr(record, "source_hash", "")
        stage = getattr(record, "stage", "")
        parts = [p for p in (batch, source, stage) if p]
        record.correlation = f"[{'/'.join(parts)}] " if parts else ""  # type: ignore[attr-defined]
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the correlation filter attached."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, CorrelationFilter) for f in logger.filters):
        logger.addFilter(CorrelationFilter())
    return logger


def install_correlation_logging() -> None:
    """Install the correlation formatter on the root logger's handlers."""
    root = logging.getLogger()
    corr_filter = CorrelationFilter()
    if not any(isinstance(f, CorrelationFilter) for f in root.filters):
        root.addFilter(corr_filter)
    formatter = CorrelationFormatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
