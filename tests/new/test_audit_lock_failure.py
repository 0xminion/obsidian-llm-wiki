"""Failure paths must not leave a live process wedging future acquisition."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_llm_wiki.core import lock as lock_module
from obsidian_llm_wiki.core.lock import acquire_lock, release_lock


@pytest.mark.skipif(os.name == "nt", reason="POSIX liveness probe")
def test_legacy_pid_permission_denied_is_still_live(tmp_path):
    lock = tmp_path / "compile.lock"
    lock.write_text(str(os.getpid()))
    try:
        with patch("os.kill", side_effect=PermissionError):
            assert acquire_lock(lock) is False
        assert lock.read_text() == str(os.getpid())
    finally:
        release_lock(lock)


@pytest.mark.parametrize("exception", [OSError("disk failed"), KeyboardInterrupt(), SystemExit()])
def test_interrupted_pid_publication_is_rolled_back(tmp_path, exception):
    lock = tmp_path / "compile.lock"
    original_write = os.write

    def interrupted_write(fd, data):
        original_write(fd, data)
        raise exception

    try:
        with patch("os.write", interrupted_write), pytest.raises(type(exception)):
            acquire_lock(lock)
        assert not lock.exists(), "failed acquisition left our live PID behind"
        assert acquire_lock(lock), "failed acquisition leaked the advisory guard"
    finally:
        release_lock(lock)


def test_short_pid_writes_are_completed(tmp_path):
    lock = tmp_path / "compile.lock"
    original_write = os.write
    try:
        with patch("os.write", side_effect=lambda fd, data: original_write(fd, data[:1])):
            assert acquire_lock(lock)
        assert lock.read_text() == str(os.getpid())
    finally:
        release_lock(lock)


def test_zero_length_write_rolls_back(tmp_path):
    lock = tmp_path / "compile.lock"
    try:
        with patch("os.write", return_value=0), pytest.raises(OSError):
            acquire_lock(lock)
        assert not lock.exists()
        assert acquire_lock(lock)
    finally:
        release_lock(lock)


def test_interrupted_ownership_registration_is_rolled_back(tmp_path, monkeypatch):
    class InterruptedRegistry(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise KeyboardInterrupt

    registry = InterruptedRegistry()
    monkeypatch.setattr(lock_module, "_held_locks", registry)
    lock = tmp_path / "compile.lock"
    with pytest.raises(KeyboardInterrupt):
        acquire_lock(lock)
    assert not lock.exists()
    assert not registry, "interrupted registration retained a closed guard descriptor"


def test_unreadable_pid_file_is_not_treated_as_stale(tmp_path):
    lock = tmp_path / "compile.lock"
    lock.write_text("do not delete")
    with patch.object(Path, "read_text", side_effect=PermissionError):
        assert acquire_lock(lock) is False
    assert lock.read_text() == "do not delete"
    try:
        assert acquire_lock(lock), "read failure leaked the guard"
    finally:
        release_lock(lock)


def test_non_owner_release_cannot_remove_legacy_live_pid(tmp_path):
    lock = tmp_path / "compile.lock"
    lock.write_text(str(os.getpid()))
    release_lock(lock)
    assert lock.read_text() == str(os.getpid())
