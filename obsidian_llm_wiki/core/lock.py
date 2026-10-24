"""Compile lock backed by a stable OS advisory lock and a legacy PID file.

``<lock_file>.guard`` is deliberately persistent: unlinking a locked guard can
split waiters across different inodes and admit two owners. Never remove it
while any compiler may be running. The PID file is only compatibility metadata;
all inspection, stale cleanup, publication and release happen under the guard.

A pre-existing live PID still blocks acquisition, including a legacy compiler
which does not know about the guard. Concurrent *old* implementations cannot be
made race-free by this module; upgrade all workers sharing a compile directory.
"""

from __future__ import annotations

import errno
import os
import threading
from contextlib import suppress
from pathlib import Path

from obsidian_llm_wiki.core.generated_paths import checked_generated_path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

__all__ = ["acquire_lock", "release_lock"]

# Keep the descriptor open for the entire compilation, not just publication.
_held_locks: dict[Path, int] = {}
_state_lock = threading.Lock()


def _after_fork() -> None:
    # flock is shared by duplicated descriptors. Close, never LOCK_UN, in the
    # child: explicitly unlocking would also unlock the parent's compilation.
    for fd in _held_locks.values():
        os.close(fd)
    _held_locks.clear()
    _state_lock.release()


if hasattr(os, "register_at_fork"):
    # Serialize fork with acquisition/release so no unregistered fd or mutex
    # owned by a vanished thread can be inherited by the child.
    os.register_at_fork(
        before=_state_lock.acquire,
        after_in_parent=_state_lock.release,
        after_in_child=_after_fork,
    )


def _try_guard(fd: int) -> bool:
    """Take a nonblocking exclusive lock; close(fd) releases it on either OS."""
    try:
        if os.name == "nt":
            # Windows permits locking a range past EOF, so an empty file works.
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return False
        raise
    return True


def acquire_lock(lock_file: str | Path) -> bool:
    """Try to acquire a compile lock without waiting for another owner.

    A dead, empty or malformed PID file is reclaimed only while holding the
    guard. OS/read errors are not evidence that another owner is dead.
    """
    # Keep a lexical ownership key: resolving a replaced path loses the held
    # descriptor and can turn stale-PID reclamation into an outside deletion.
    lf = Path(lock_file).absolute()
    with _state_lock:
        checked_generated_path(lf)
        guard = checked_generated_path(Path(str(lf) + ".guard"))
        if lf in _held_locks:
            return False
        lf.parent.mkdir(parents=True, exist_ok=True)
        guard_fd = os.open(guard, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        keep_guard = False
        created_pid_file = False
        try:
            if not _try_guard(guard_fd):
                return False
            checked_generated_path(lf)
            try:
                pid = int(lf.read_text(encoding="utf-8").strip())
            except (FileNotFoundError, ValueError):
                pass  # Missing, empty, non-UTF-8 or malformed PID metadata.
            except OSError:
                return False
            else:
                if _pid_alive(pid):
                    return False
            # No guarded worker can replace this stale file between read/unlink.
            checked_generated_path(lf)
            try:
                lf.unlink(missing_ok=True)
            except OSError:
                return False
            try:
                fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                return False
            created_pid_file = True
            try:
                remaining = str(os.getpid()).encode("ascii")
                while remaining:
                    written = os.write(fd, remaining)
                    if written == 0:
                        raise OSError(errno.EIO, "Could not write compile lock PID")
                    remaining = remaining[written:]
            finally:
                os.close(fd)
            _held_locks[lf] = guard_fd
            keep_guard = True
            return True
        except BaseException:
            keep_guard = False
            raise
        finally:
            if not keep_guard:
                try:
                    _held_locks.pop(lf, None)
                    if created_pid_file:
                        # Includes KeyboardInterrupt/SystemExit during publication.
                        with suppress(OSError, ValueError):
                            checked_generated_path(lf).unlink(missing_ok=True)
                finally:
                    os.close(guard_fd)


def release_lock(lock_file: str | Path) -> None:
    """Release only this process's acquisition; leave the stable guard intact."""
    lf = Path(lock_file).absolute()
    with _state_lock:
        guard_fd = _held_locks.pop(lf, None)
        if guard_fd is None:
            return
        try:
            try:
                checked_generated_path(lf)
                if int(lf.read_text(encoding="utf-8").strip()) == os.getpid():
                    checked_generated_path(lf).unlink()
            except (ValueError, OSError):
                pass
        finally:
            os.close(guard_fd)


def _pid_alive(pid: int) -> bool:
    """Check liveness conservatively; uncertainty must not evict an owner."""
    if not 0 < pid <= 0xFFFFFFFF:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) uses TerminateProcess on Windows, not a liveness probe!
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OverflowError):
        return False
    except OSError:
        return True


def _windows_pid_alive(pid: int) -> bool:
    """Wait on a process handle without signaling or terminating the process."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL

    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        # Only ERROR_INVALID_PARAMETER proves that the PID does not exist.
        return ctypes.get_last_error() != 87
    try:
        # Only WAIT_OBJECT_0 proves termination; timeout/failure means keep it.
        return kernel.WaitForSingleObject(handle, 0) != 0
    finally:
        kernel.CloseHandle(handle)
