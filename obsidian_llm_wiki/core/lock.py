"""PID-based compile lock — prevents concurrent compilations."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["acquire_lock", "release_lock"]


def acquire_lock(lock_file: str | Path) -> bool:
    """Try to acquire a compile lock.  Returns True on success.

    Uses O_CREAT|O_EXCL for atomic creation — no TOCTOU race.
    """
    lf = Path(lock_file)
    if lf.exists():
        try:
            pid = int(lf.read_text().strip())
            # Check if process is still running.
            if _pid_alive(pid):
                return False
        except (ValueError, OSError):
            pass  # Stale/corrupt lock — reclaim it.
        # Remove the stale/corrupt lock file so O_EXCL can create a new one.
        try:
            lf.unlink()
        except OSError:
            return False

    lf.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Atomic create-or-fail — eliminates TOCTOU race.
        fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        # Another process created the lock between our check and create.
        return False


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
