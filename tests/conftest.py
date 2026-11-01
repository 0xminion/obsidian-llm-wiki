"""Shared test fixtures and environment isolation.

Ensures os.environ is restored after each test so load_dotenv(override=True)
in one test doesn't leak SYNTHESIS_MODE, COMPILE_CONCURRENCY, etc. into the next.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Isolate fallback paths before collection, not just inside test fixtures.

    A test calling monkeypatch.undo() must return to this sandbox rather than
    the caller's actual home. Session artifacts remain under the temp root for
    inspection; no user cache is cleared or migrated.
    """
    original = dict(os.environ)
    original_tempdir = tempfile.tempdir
    root = Path(tempfile.mkdtemp(prefix="olw-test-session-"))
    for name in ("home", "cache", "config", "data", "tmp"):
        (root / name).mkdir()
    os.environ.update({
        "HOME": str(root / "home"),
        "USERPROFILE": str(root / "home"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_DATA_HOME": str(root / "data"),
        "TMPDIR": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TEMP": str(root / "tmp"),
    })
    forbidden = {
        "VAULT_PATH", "RESIDENTIAL_PROXY_URL", "YOUTUBE_COOKIES_FILE",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    }
    for key in list(os.environ):
        upper = key.upper()
        if upper in forbidden or upper.endswith(("_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD")):
            os.environ.pop(key, None)
    tempfile.tempdir = None

    def restore() -> None:
        os.environ.clear()
        os.environ.update(original)
        tempfile.tempdir = original_tempdir

    config.add_cleanup(restore)


@pytest.fixture(autouse=True)
def _isolate_env() -> Iterator[None]:
    """Snapshot and restore os.environ around every test."""
    snapshot = deepcopy(os.environ)
    yield
    # Remove keys that were added during the test
    for key in list(os.environ):
        if key not in snapshot:
            del os.environ[key]
    # Restore original values
    for key, val in snapshot.items():
        os.environ[key] = val
