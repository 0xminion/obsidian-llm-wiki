"""Shared test fixtures and environment isolation.

Ensures os.environ is restored after each test so load_dotenv(override=True)
in one test doesn't leak SYNTHESIS_MODE, COMPILE_CONCURRENCY, etc. into the next.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from copy import deepcopy

import pytest


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
