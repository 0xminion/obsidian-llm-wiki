"""Compatibility shim: registry-level identity fixtures use bilateral proof.

The former ``assess_same(registry, synthesis)`` recorded a one-sided ``same``
with a fabricated ``span-1`` support token. Under the approved bilateral
contract such a record never binds (no revision-scoped context on either
side), so it is gone. Fixtures now declare the exact source bytes of both
contributions and the explicit fact that makes them the same referent; see
:mod:`tests.new.bilateral_fixture_helpers`.
"""

from __future__ import annotations

from tests.new.bilateral_fixture_helpers import (
    assess,
    assess_same,
    fact_verdict,
    reviewed,
    upsert_reviewed,
)

__all__ = ["assess", "assess_same", "fact_verdict", "reviewed", "upsert_reviewed"]
