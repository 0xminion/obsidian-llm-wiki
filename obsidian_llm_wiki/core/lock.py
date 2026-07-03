"""PID-based compile lock — prevents concurrent compilations."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["acquire_lock", "release_lock"]


def acquire_lock(lock_file: str | Path) -> bool:
    """Try to acquire a compile lock.  Returns True on success."""
    lf = Path(lock_file)
    if lf.exists():
        try:
            pid = int(lf.read_text().strip())
            # Check if process is still running.
            if _pid_alive(pid):
                return False
        except (ValueError, OSError):
            pass  # Stale/corrupt lock — reclaim it.

    lf.parent.mkdir(parents=True, exist_ok=True)
    lf.write_text(str(os.getpid()))
    return True


def release_lock(lock_file: str | Path) -> None:
    """Release the compile lock if we hold it."""
    lf = Path(lock_file)
    if not lf.exists():
        return
    try:
        pid = int(lf.read_text().strip())
        if pid == os.getpid():
            lf.unlink()
    except (ValueError, OSError):
        pass


def _pid_alive(pid: int) -> bool:
    """Check if a process with ``pid`` is running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
