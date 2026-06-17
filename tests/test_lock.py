"""Tests for pipeline.lock — PID-based lock acquisition, release, stale reclamation."""

from __future__ import annotations

import os
from pathlib import Path

from pipeline.lock import acquire_lock, release_lock


class TestAcquireLock:
    """acquire_lock scenarios."""

    def test_fresh_directory_succeeds(self, tmp_path: Path) -> None:
        """Lock on a directory with no existing lock file should succeed."""
        lock_path = tmp_path / "compile.lock"
        assert acquire_lock(lock_path) is True
        assert lock_path.exists()
        # Lock file should contain our PID
        assert lock_path.read_text().strip() == str(os.getpid())

    def test_live_pid_fails(self, tmp_path: Path) -> None:
        """Lock with a live PID should fail to acquire."""
        lock_path = tmp_path / "compile.lock"
        # Create lock with our own PID (which is alive)
        lock_path.write_text(str(os.getpid()))
        assert acquire_lock(lock_path) is False
        # Lock file should still exist with original PID
        assert lock_path.read_text().strip() == str(os.getpid())

    def test_dead_pid_reclaims(self, tmp_path: Path) -> None:
        """Lock with a dead PID should be reclaimed."""
        lock_path = tmp_path / "compile.lock"
        # Use a PID that is very unlikely to exist (max int)
        fake_pid = 2**31 - 1
        lock_path.write_text(str(fake_pid))
        assert acquire_lock(lock_path) is True
        # Lock file should now contain our PID
        assert lock_path.read_text().strip() == str(os.getpid())

    def test_garbage_in_lock_file(self, tmp_path: Path) -> None:
        """Lock file with garbage (non-numeric) content should be treated as stale and reclaimed."""
        lock_path = tmp_path / "compile.lock"
        lock_path.write_text("not-a-pid-garbage\n")
        # _read_pid returns None for garbage, _is_lock_stale returns True → reclaim
        assert acquire_lock(lock_path) is True
        assert lock_path.read_text().strip() == str(os.getpid())

    def test_empty_lock_file(self, tmp_path: Path) -> None:
        """Empty lock file should be treated as stale and reclaimed."""
        lock_path = tmp_path / "compile.lock"
        lock_path.write_text("")
        assert acquire_lock(lock_path) is True
        assert lock_path.read_text().strip() == str(os.getpid())


class TestReleaseLock:
    """release_lock scenarios."""

    def test_release_removes_lock(self, tmp_path: Path) -> None:
        """release_lock should remove an existing lock file."""
        lock_path = tmp_path / "compile.lock"
        lock_path.write_text(str(os.getpid()))
        assert lock_path.exists()
        release_lock(lock_path)
        assert not lock_path.exists()

    def test_release_no_lock_no_crash(self, tmp_path: Path) -> None:
        """release_lock on non-existent lock should not raise."""
        lock_path = tmp_path / "nonexistent.lock"
        # Should not raise FileNotFoundError
        release_lock(lock_path)

    def test_release_after_acquire(self, tmp_path: Path) -> None:
        """Full acquire-then-release cycle should clean up."""
        lock_path = tmp_path / "compile.lock"
        assert acquire_lock(lock_path) is True
        assert lock_path.exists()
        release_lock(lock_path)
        assert not lock_path.exists()

    def test_release_idempotent(self, tmp_path: Path) -> None:
        """Calling release_lock multiple times should not crash."""
        lock_path = tmp_path / "compile.lock"
        lock_path.write_text(str(os.getpid()))
        release_lock(lock_path)
        release_lock(lock_path)
        release_lock(lock_path)
        assert not lock_path.exists()


class TestReclaimProtocol:
    """Tests for the two-lock stale reclamation protocol."""

    def test_reclaim_cleans_reclaim_file(self, tmp_path: Path) -> None:
        """Reclaiming a stale lock should not leave .reclaim file behind."""
        lock_path = tmp_path / "compile.lock"
        reclaim_path = tmp_path / "compile.lock.reclaim"
        fake_pid = 2**31 - 1
        lock_path.write_text(str(fake_pid))
        assert acquire_lock(lock_path) is True
        # .reclaim file should have been cleaned up in finally block
        assert not reclaim_path.exists()

    def test_reacquire_after_release(self, tmp_path: Path) -> None:
        """After releasing, a new acquire should succeed."""
        lock_path = tmp_path / "compile.lock"
        assert acquire_lock(lock_path) is True
        release_lock(lock_path)
        # Second acquire should succeed since lock was released
        assert acquire_lock(lock_path) is True
        release_lock(lock_path)
