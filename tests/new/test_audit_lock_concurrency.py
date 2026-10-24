"""Deterministic process interleavings for the compile lock."""

from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_llm_wiki.core.lock import acquire_lock, release_lock

_TIMEOUT = 15


def _owner(lock, connection, finish, pause_at=None, paused=None, resume=None):
    """Hold a successful acquisition until the controller finishes assertions."""
    lock = Path(lock)
    original_open = os.open
    original_write = os.write
    original_unlink = Path.unlink
    did_pause = False

    def pause_once():
        nonlocal did_pause
        if not did_pause:
            assert paused is not None and resume is not None
            did_pause = True
            paused.set()
            assert resume.wait(_TIMEOUT), "controller did not resume acquisition"

    def paused_open(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if pause_at == "empty" and Path(path) == lock and flags & os.O_EXCL:
            pause_once()
        return fd

    def paused_unlink(path, *args, **kwargs):
        if pause_at == "stale" and path == lock:
            pause_once()
        return original_unlink(path, *args, **kwargs)

    def paused_write(fd, data):
        written = original_write(fd, data)
        if pause_at == "published":
            pause_once()
        return written

    try:
        with (
            patch("os.open", paused_open),
            patch("os.write", paused_write),
            patch.object(Path, "unlink", paused_unlink),
        ):
            acquired = acquire_lock(lock)
            connection.send(("acquired", acquired))
            if acquired:
                assert finish.wait(_TIMEOUT), "controller did not finish"
                release_lock(lock)
    except BaseException as exc:
        connection.send(("error", repr(exc)))
        raise
    finally:
        connection.close()


def _result(connection):
    assert connection.poll(_TIMEOUT), "worker did not report acquisition"
    kind, value = connection.recv()
    assert kind == "acquired", value
    return value


def _stop(processes, finish, resume=None):
    finish.set()
    if resume is not None:
        resume.set()
    for process in processes:
        process.join(_TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join(_TIMEOUT)
        assert process.exitcode == 0


@pytest.mark.parametrize("pause_at", ["empty", "stale"])
def test_in_progress_acquisition_cannot_be_reclaimed(tmp_path, pause_at):
    """A second process must not steal an empty or stale-but-observed PID file."""
    ctx = multiprocessing.get_context("spawn")
    lock = tmp_path / "compile.lock"
    if pause_at == "stale":
        lock.write_text("corrupt")
    paused, resume, finish = ctx.Event(), ctx.Event(), ctx.Event()
    first_rx, first_tx = ctx.Pipe(duplex=False)
    second_rx, second_tx = ctx.Pipe(duplex=False)
    first = ctx.Process(target=_owner, args=(lock, first_tx, finish, pause_at, paused, resume))
    second = ctx.Process(target=_owner, args=(lock, second_tx, finish))
    processes = [first]
    first.start()
    try:
        assert paused.wait(_TIMEOUT), "worker did not reach race window"
        second.start()
        processes.append(second)
        second_acquired = _result(second_rx)
        resume.set()
        first_acquired = _result(first_rx)
        assert (first_acquired, second_acquired) == (True, False)
    finally:
        _stop(processes, finish, resume)
        for connection in (first_rx, first_tx, second_rx, second_tx):
            connection.close()


def _idle_child(ready, finish):
    ready.set()
    assert finish.wait(_TIMEOUT)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_idle_fork_child_does_not_extend_parent_lock_lifetime(tmp_path):
    """Inherited flock descriptors must be closed even if the child never locks."""
    ctx = multiprocessing.get_context("fork")
    ready, finish = ctx.Event(), ctx.Event()
    lock = tmp_path / "compile.lock"
    assert acquire_lock(lock)
    child = ctx.Process(target=_idle_child, args=(ready, finish))
    child.start()
    try:
        assert ready.wait(_TIMEOUT)
        release_lock(lock)
        assert acquire_lock(lock), "idle child retained the parent's advisory lock"
    finally:
        release_lock(lock)
        _stop([child], finish)


@pytest.mark.parametrize("pause_at", ["empty", "published", None])
def test_process_death_releases_guard_even_during_acquisition(tmp_path, pause_at):
    ctx = multiprocessing.get_context("spawn")
    lock = tmp_path / "compile.lock"
    paused, resume, finish = ctx.Event(), ctx.Event(), ctx.Event()
    receiver, sender = ctx.Pipe(duplex=False)
    owner = ctx.Process(target=_owner, args=(lock, sender, finish, pause_at, paused, resume))
    owner.start()
    try:
        if pause_at is None:
            assert _result(receiver)
        else:
            assert paused.wait(_TIMEOUT)
        assert acquire_lock(lock) is False
        owner.terminate()
        owner.join(_TIMEOUT)
        assert not owner.is_alive()
        assert owner.exitcode != 0
        assert acquire_lock(lock), "dead process retained its lock"
    finally:
        if owner.is_alive():
            owner.terminate()
            owner.join(_TIMEOUT)
        release_lock(lock)
        receiver.close()
        sender.close()


def _race_owner(lock, connection, start, finish, done, rounds):
    for _ in range(rounds):
        start.wait(_TIMEOUT)
        acquired = acquire_lock(lock)
        connection.send(("acquired", acquired))
        finish.wait(_TIMEOUT)
        if acquired:
            release_lock(lock)
        done.wait(_TIMEOUT)
    connection.close()


@pytest.mark.parametrize("initial", [None, "stale"])
def test_simultaneous_processes_have_exactly_one_owner(tmp_path, initial):
    ctx = multiprocessing.get_context("spawn")
    count, rounds = 6, 5
    start, finish, done = (ctx.Barrier(count + 1) for _ in range(3))
    lock = tmp_path / "compile.lock"
    if initial:
        lock.write_text(initial)
    channels = [ctx.Pipe(duplex=False) for _ in range(count)]
    processes = [
        ctx.Process(target=_race_owner, args=(lock, sender, start, finish, done, rounds))
        for _, sender in channels
    ]
    for process in processes:
        process.start()
    try:
        for _ in range(rounds):
            start.wait(_TIMEOUT)
            results = [_result(receiver) for receiver, _ in channels]
            assert sum(results) == 1, results
            winner = processes[results.index(True)]
            assert lock.read_text() == str(winner.pid)
            finish.wait(_TIMEOUT)
            done.wait(_TIMEOUT)
            assert not lock.exists()
        for process in processes:
            process.join(_TIMEOUT)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(_TIMEOUT)
        for receiver, sender in channels:
            receiver.close()
            sender.close()


def test_guard_survives_release_without_changing_identity(tmp_path):
    lock = tmp_path / "compile.lock"
    guard = tmp_path / "compile.lock.guard"
    try:
        assert acquire_lock(lock)
        first = guard.stat()
        release_lock(lock)
        assert not lock.exists()
        assert guard.exists()
        assert acquire_lock(lock)
        second = guard.stat()
        assert (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)
    finally:
        release_lock(lock)


def test_missing_pid_file_does_not_allow_a_second_owner(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    lock = tmp_path / "compile.lock"
    finish = ctx.Event()
    receiver, sender = ctx.Pipe(duplex=False)
    owner = ctx.Process(target=_owner, args=(lock, sender, finish))
    owner.start()
    try:
        assert _result(receiver)
        lock.unlink()
        assert acquire_lock(lock) is False
    finally:
        _stop([owner], finish)
        release_lock(lock)
        receiver.close()
        sender.close()


def test_simultaneous_threads_have_one_owner(tmp_path):
    lock = tmp_path / "compile.lock"
    start = threading.Barrier(8)

    def contend(_):
        start.wait(_TIMEOUT)
        return acquire_lock(lock)

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(contend, range(8)))
        assert sum(results) == 1
        assert lock.read_text() == str(os.getpid())
    finally:
        release_lock(lock)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_fork_child_cannot_release_parents_guard(tmp_path):
    lock = tmp_path / "compile.lock"
    assert acquire_lock(lock)
    # Remove metadata so only the guard can exclude the fork child.
    lock.unlink()
    ctx = multiprocessing.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    finish = ctx.Event()
    child = ctx.Process(target=_owner, args=(lock, sender, finish))
    child.start()
    try:
        assert _result(receiver) is False
    finally:
        _stop([child], finish)
        release_lock(lock)
        receiver.close()
        sender.close()
