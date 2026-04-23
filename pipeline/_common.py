"""Shared infrastructure for the pipeline.

Ports common.sh utilities to Python:
- check_dependencies(): Preflight dependency check
- VaultLock: Directory-based lock with PID/time-based stale detection
- run_with_retry(): Retry with exponential backoff
- append_log_md(): Append structured entries to log.md (Karpathy-style)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

# Required CLI tools for the pipeline
REQUIRED_TOOLS: list[str] = ["python3", "hermes", "curl"]
# Optional but recommended tools
OPTIONAL_TOOLS: list[str] = ["qmd", "yt-dlp", "ffmpeg"]

log = logging.getLogger(__name__)


class VaultLock:
    """Directory-based lock for vault operations with stale lock detection.

    Uses pathlib mkdir for atomic locking (mkdir is atomic on POSIX).
    Detects stale locks via PID check and 30-minute time expiry.
    Supports context manager protocol.

    Example::

        lock = VaultLock(vault_path, "pipeline")
        with lock:
            # critical section
            pass
    """

    STALE_TIMEOUT_SECONDS: float = 1800.0  # 30 minutes

    def __init__(self, vault_path: Path, name: str = "pipeline") -> None:
        self.vault_path = Path(vault_path)
        self.name = name
        # Per-user lock dir — avoids world-writable /tmp race conditions
        lock_root = Path.home() / ".local" / "obsidian-llm-wiki" / "locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        vault_hash = hashlib.md5(str(self.vault_path.resolve()).encode()).hexdigest()[:8]
        self.lock_dir = lock_root / f"{name}-{vault_hash}.lock"
        self._acquired: bool = False

    def _pid_running(self, pid: int) -> bool:
        """Check if a process with given PID is currently running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _is_stale(self) -> bool:
        """Determine if an existing lock is stale."""
        if not self.lock_dir.exists():
            return False

        pid_file = self.lock_dir / "pid"

        # Time-based stale detection
        try:
            lock_age = time.time() - self.lock_dir.stat().st_mtime
            if lock_age > self.STALE_TIMEOUT_SECONDS:
                log.warning(
                    "Stale lock detected (%.0fs old, exceeding %ds threshold)",
                    lock_age,
                    int(self.STALE_TIMEOUT_SECONDS),
                )
                return True
        except OSError:
            pass

        # PID-based stale detection
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                if not self._pid_running(old_pid):
                    log.warning(
                        "Stale lock detected (PID %d no longer running)", old_pid
                    )
                    return True
            except (ValueError, OSError):
                log.warning("Stale lock detected (unreadable PID file)")
                return True
        else:
            # No PID file — legacy or very old instance
            log.warning("Stale lock detected (no PID file)")
            return True

        return False

    def _force_release(self) -> None:
        """Force-remove the lock directory."""
        try:
            shutil.rmtree(self.lock_dir, ignore_errors=True)
        except OSError:
            pass

    def acquire(self) -> bool:
        """Attempt to acquire the lock.

        Returns:
            True if lock was acquired, False if another instance holds it.
        """
        while True:
            try:
                self.lock_dir.mkdir(exist_ok=False)
                self._acquired = True
                try:
                    (self.lock_dir / "pid").write_text(str(os.getpid()))
                except OSError:
                    pass
                return True
            except FileExistsError:
                # Check for stale lock
                if self._is_stale():
                    self._force_release()
                    continue
                return False

    def release(self) -> None:
        """Release the lock if currently held."""
        if self._acquired:
            try:
                shutil.rmtree(self.lock_dir, ignore_errors=True)
            except OSError:
                pass
            self._acquired = False

    def __enter__(self) -> VaultLock:
        if not self.acquire():
            raise RuntimeError(
                f"Could not acquire lock '{self.name}'. "
                f"Another instance may be running. Lock path: {self.lock_dir}"
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


def run_with_retry(
    fn: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
    description: str = "",
    on_retry: Optional[Callable[[int, float, Exception], None]] = None,
    retryable_errors: Optional[tuple[type[Exception], ...]] = None,
) -> T:
    """Retry a callable with exponential backoff.

    Args:
        fn: Callable to execute (no arguments — use functools.partial or lambda).
        max_retries: Maximum number of attempts.
        initial_delay: Seconds to wait before first retry.
        backoff_factor: Multiplier for delay after each retry.
        description: Human-readable description for logging.
        on_retry: Optional callback(attempt, delay, exception) on each retry.
        retryable_errors: If set, only retry on these exception types.

    Returns:
        The return value of fn on success.

    Raises:
        The last exception raised by fn after exhausting retries.
    """
    delay = initial_delay
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            log.debug(
                "Attempt %d/%d: %s", attempt, max_retries, description or str(fn)
            )
            result = fn()
            if attempt > 1:
                log.info("SUCCESS after %d attempts: %s", attempt, description)
            return result
        except Exception as exc:
            last_exc = exc
            if retryable_errors and not isinstance(exc, retryable_errors):
                log.debug(
                    "Non-retryable error (%s), not retrying: %s",
                    type(exc).__name__,
                    exc,
                )
                raise

            log.warning(
                "FAILED (attempt %d/%d): %s — %s: %s",
                attempt,
                max_retries,
                description,
                type(exc).__name__,
                exc,
            )

            if attempt < max_retries:
                log.info("Waiting %.1fs before retry (exponential backoff)...", delay)
                if on_retry:
                    on_retry(attempt, delay, exc)
                time.sleep(delay)
                delay *= backoff_factor

    log.error("GIVING UP after %d attempts: %s", max_retries, description)
    raise last_exc  # type: ignore[misc]


# Karpathy-style log.md header
_LOG_MD_HEADER = """\
# Wiki Activity Log

Chronological record of all operations on the knowledge base.
Use `grep "^## \\[" log.md | tail -N` to see the last N operations.

---
"""


def append_log_md(
    vault_path: Path,
    operation: str,
    title: str,
    details: str = "",
) -> None:
    """Append a structured entry to log.md (Karpathy-style).

    Args:
        vault_path: Path to the vault root.
        operation: Operation type (e.g. "ingest", "compile", "setup").
        title: Short description of the operation.
        details: Optional details text (bullet list, etc.).
    """
    log_md = Path(vault_path) / "06-Config" / "log.md"
    log_md.parent.mkdir(parents=True, exist_ok=True)

    if not log_md.exists():
        log_md.write_text(_LOG_MD_HEADER, encoding="utf-8")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = f"\n## [{date_str}] {operation} | {title}\n{details}\n"

    with log_md.open("a", encoding="utf-8") as f:
        f.write(entry)
