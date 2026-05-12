"""PID-based lock file for preventing concurrent compilation.

Ported from llm-wiki-compiler/src/utils/lock.ts.

Uses O_CREAT | O_EXCL for atomic lock creation. Handles stale lock
reclamation via a two-lock protocol to serialize cleanup.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path


def acquire_lock(lock_path: str | Path) -> bool:
    """Acquire a PID-based lock file. Returns True on success.

    If a stale lock is detected (dead PID), reclaims it atomically.
    """
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(2):  # max 2 attempts (stale reclamation may need retry)
        created = _try_create_lock(lock)
        if created:
            return True

        if not _is_lock_stale(lock):
            print(f"⚠ Another compilation is running (pid={_read_pid(lock)}).",
                  file=sys.stderr)
            return False

        # Stale lock — reclaim
        reclaimed = _reclaim_stale_lock(lock)
        if reclaimed:
            return True

    print("⚠ Could not acquire lock after retrying.", file=sys.stderr)
    return False


def release_lock(lock_path: str | Path) -> None:
    """Release the lock. Safe to call even if lock doesn't exist."""
    with contextlib.suppress(FileNotFoundError):
        Path(lock_path).unlink()


def _try_create_lock(lock_path: Path) -> bool:
    """Atomically create lock with our PID. Returns True if created."""
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def _read_pid(lock_path: Path) -> int | None:
    """Read PID from an existing lock file."""
    try:
        return int(lock_path.read_text().strip())
    except (ValueError, FileNotFoundError):
        return None


def _is_lock_stale(lock_path: Path) -> bool:
    """Check if the lock-holding process is dead."""
    pid = _read_pid(lock_path)
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
        return False  # process alive
    except (OSError, ProcessLookupError):
        return True  # dead


def _reclaim_stale_lock(lock_path: Path) -> bool:
    """Reclaim a stale lock. Returns True if we got it."""
    reclaim_path = lock_path.with_suffix(lock_path.suffix + ".reclaim")

    # Get reclamation lock
    if not _try_create_lock(reclaim_path):
        # Reclaim lock exists — check if stale
        if _is_lock_stale(reclaim_path):
            with contextlib.suppress(FileNotFoundError):
                reclaim_path.unlink()
        return False

    try:
        # Re-verify staleness under reclamation lock
        if not _is_lock_stale(lock_path):
            return False

        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()

        acquired = _try_create_lock(lock_path)
        if acquired:
            print("♻ Reclaimed stale lock from dead process.", file=sys.stderr)
        return acquired
    finally:
        with contextlib.suppress(FileNotFoundError):
            reclaim_path.unlink()
