"""Vault migration adapter — re-exports the legacy migrate module."""

from __future__ import annotations

# Re-export from the legacy pipeline package.
# The migrate logic is stable and well-tested (357 tests).
# It will be rewritten to use obsidian_llm_wiki directly in a future phase.
from pipeline.migrate import migrate_vault  # noqa: F401

__all__ = ["migrate_vault"]
