"""Platform branches, including Win32 ABI calls exercised with test doubles."""

from __future__ import annotations

import ctypes
import errno
import os
from ctypes import wintypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from obsidian_llm_wiki.core import lock


@pytest.mark.parametrize(
    ("handle", "wait_result", "last_error", "alive"),
    [
        (0x123456789, 258, 0, True),  # WAIT_TIMEOUT: running
        (0x123456789, 0, 0, False),  # WAIT_OBJECT_0: exited
        (0x123456789, 0xFFFFFFFF, 0, True),  # WAIT_FAILED: fail closed
        (0, 0, 87, False),  # ERROR_INVALID_PARAMETER: nonexistent PID
        (0, 0, 5, True),  # ERROR_ACCESS_DENIED: do not steal
        (0, 0, 8, True),  # Other OS failure: not proof of death
    ],
)
def test_windows_pid_probe_never_signals_process(
    monkeypatch, handle, wait_result, last_error, alive
):
    kernel = SimpleNamespace(
        OpenProcess=Mock(return_value=handle),
        WaitForSingleObject=Mock(return_value=wait_result),
        CloseHandle=Mock(return_value=1),
    )
    load_dll = Mock(return_value=kernel)
    monkeypatch.setattr(ctypes, "WinDLL", load_dll, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error, raising=False)
    # Replacing only this module's os reference avoids changing pathlib's OS.
    kill = Mock(side_effect=AssertionError("os.kill(pid, 0) terminates processes on Windows"))
    monkeypatch.setattr(lock, "os", SimpleNamespace(name="nt", kill=kill))

    assert lock._pid_alive(1234) is alive
    kill.assert_not_called()
    load_dll.assert_called_once_with("kernel32", use_last_error=True)
    kernel.OpenProcess.assert_called_once_with(0x00100000, False, 1234)
    assert kernel.OpenProcess.restype is wintypes.HANDLE
    if handle:
        kernel.WaitForSingleObject.assert_called_once_with(handle, 0)
        kernel.CloseHandle.assert_called_once_with(handle)
    else:
        kernel.WaitForSingleObject.assert_not_called()
        kernel.CloseHandle.assert_not_called()


@pytest.mark.parametrize("pid", [0, -1, -42, 999999999999999999999999999])
def test_invalid_pid_does_not_probe_process_group(tmp_path, monkeypatch, pid):
    marker = tmp_path / "compile.lock"
    marker.write_text(str(pid))
    kill = Mock()
    monkeypatch.setattr(lock.os, "kill", kill)
    try:
        assert lock.acquire_lock(marker)
        kill.assert_not_called()
    finally:
        lock.release_lock(marker)


@pytest.mark.parametrize("error", [None, errno.EACCES, errno.EAGAIN, errno.EDEADLK, errno.EIO])
def test_windows_guard_locks_first_byte_without_waiting(monkeypatch, error):
    locking = Mock(side_effect=OSError(error, "lock failed") if error else None)
    fake_msvcrt = SimpleNamespace(locking=locking, LK_NBLCK=2)
    seek = Mock()
    monkeypatch.setattr(lock, "msvcrt", fake_msvcrt, raising=False)
    monkeypatch.setattr(lock, "os", SimpleNamespace(name="nt", lseek=seek, SEEK_SET=os.SEEK_SET))
    if error == errno.EIO:
        with pytest.raises(OSError, match="lock failed"):
            lock._try_guard(42)
    else:
        assert lock._try_guard(42) is (error is None)
    seek.assert_called_once_with(42, 0, os.SEEK_SET)
    locking.assert_called_once_with(42, fake_msvcrt.LK_NBLCK, 1)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process errors")
@pytest.mark.parametrize(
    ("error", "alive"),
    [
        (ProcessLookupError(), False),
        (OverflowError(), False),
        (OSError(errno.EIO, "unknown"), True),
    ],
)
def test_posix_liveness_only_reclaims_proven_dead_pid(monkeypatch, error, alive):
    monkeypatch.setattr(lock.os, "kill", Mock(side_effect=error))
    assert lock._pid_alive(1234) is alive
