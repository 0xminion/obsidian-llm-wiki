"""Tests for core/lock.py — PID-based compile lock.

Covers: acquire on fresh lock, fail when PID alive, reclaim stale lock,
release by owning PID only.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from obsidian_llm_wiki.core.lock import acquire_lock, release_lock

# ── acquire_lock ─────────────────────────────────────────────────────────


def test_acquire_lock_fresh(tmp_path: Path):
    """Acquiring on a non-existent lock file succeeds."""
    lock = tmp_path / "compile.lock"
    assert acquire_lock(lock) is True
    assert lock.exists()
    # Lock file should contain our PID.
    assert lock.read_text() == str(os.getpid())


def test_acquire_lock_idempotent_same_pid(tmp_path: Path):
    """Re-acquiring when the lock already holds *our* PID should fail
    (the lock is held — even by us, O_EXCL prevents re-create)."""
    lock = tmp_path / "compile.lock"
    assert acquire_lock(lock) is True
    # Second acquire should fail because the file exists and the PID is alive.
    assert acquire_lock(lock) is False


def test_acquire_lock_fails_when_pid_alive(tmp_path: Path):
    """Lock held by a live PID (ourselves) → acquisition fails."""
    lock = tmp_path / "compile.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()), encoding="utf-8")
    assert acquire_lock(lock) is False


def test_acquire_lock_reclaims_stale_pid(tmp_path: Path):
    """Lock held by a dead PID → acquisition succeeds (stale reclaim)."""
    lock = tmp_path / "compile.lock"
    # Use a PID that is very likely dead (PID 0xFFFFFFFF — kernel reserved).
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("99999999", encoding="utf-8")
    # Patch _pid_alive to simulate a dead process deterministically.
    with patch("obsidian_llm_wiki.core.lock._pid_alive", return_value=False):
        assert acquire_lock(lock) is True
    assert lock.read_text() == str(os.getpid())


def test_acquire_lock_reclaims_corrupt_lock(tmp_path: Path):
    """Lock file with non-integer content → reclaim (corrupt lock)."""
    lock = tmp_path / "compile.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("not-a-pid", encoding="utf-8")
    assert acquire_lock(lock) is True
    assert lock.read_text() == str(os.getpid())


def test_acquire_lock_creates_parent_dirs(tmp_path: Path):
    """Lock in a nested path that doesn't exist yet → parent dirs created."""
    lock = tmp_path / "a" / "b" / "c" / "compile.lock"
    assert acquire_lock(lock) is True
    assert lock.exists()


# ── release_lock ─────────────────────────────────────────────────────────


def test_release_lock_by_owner(tmp_path: Path):
    """Releasing a lock we own removes the file."""
    lock = tmp_path / "compile.lock"
    acquire_lock(lock)
    assert lock.exists()
    release_lock(lock)
    assert not lock.exists()


def test_release_lock_nonexistent_is_noop(tmp_path: Path):
    """Releasing a non-existent lock does not raise."""
    lock = tmp_path / "nope.lock"
    # Should not raise.
    release_lock(lock)


def test_release_lock_wrong_pid_preserves(tmp_path: Path):
    """Releasing a lock held by a different PID does NOT remove it."""
    lock = tmp_path / "compile.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Write a different PID (not ours).
    fake_pid = os.getpid() + 1
    lock.write_text(str(fake_pid), encoding="utf-8")
    release_lock(lock)
    assert lock.exists()
    assert lock.read_text() == str(fake_pid)


def test_release_lock_corrupt_file_is_noop(tmp_path: Path):
    """Releasing a corrupt lock file does not raise."""
    lock = tmp_path / "compile.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("garbage", encoding="utf-8")
    release_lock(lock)
    # File may or may not be removed, but no exception should be raised.
    # Corrupt content with int() failing → pass → file preserved.
    assert lock.exists()


# ── acquire + release lifecycle ──────────────────────────────────────────


def test_acquire_release_acquire_cycle(tmp_path: Path):
    """Full lifecycle: acquire → release → acquire again."""
    lock = tmp_path / "cycle.lock"
    assert acquire_lock(lock) is True
    release_lock(lock)
    assert not lock.exists()
    # Should be able to acquire again after release.
    assert acquire_lock(lock) is True
    assert lock.exists()
