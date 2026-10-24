"""WAL negotiation can return SQLITE_BUSY despite a configured busy timeout."""
import sqlite3

import pytest

from obsidian_llm_wiki.core.registry import VaultRegistry


@pytest.mark.parametrize("failures", [1, 2])
def test_registry_retries_transient_wal_negotiation(tmp_path, monkeypatch, failures):
    connect = sqlite3.connect
    attempts = []

    class ContendedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA journal_mode = WAL":
                attempts.append(sql)
                if len(attempts) <= failures:
                    error = sqlite3.OperationalError("database is locked")
                    error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                    raise error
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: connect(
        *a, **kw, factory=ContendedConnection
    ))
    with VaultRegistry(tmp_path) as registry:
        assert registry.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert len(attempts) == failures + 1


def test_registry_closes_connection_on_terminal_wal_failure(tmp_path, monkeypatch):
    connect = sqlite3.connect
    opened = []

    class BrokenConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA journal_mode = WAL":
                error = sqlite3.OperationalError("disk I/O error")
                error.sqlite_errorcode = sqlite3.SQLITE_IOERR
                raise error
            return super().execute(sql, *args, **kwargs)

    def open_connection(*args, **kwargs):
        connection = connect(*args, **kwargs, factory=BrokenConnection)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", open_connection)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        VaultRegistry(tmp_path)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
