"""Shared CLI helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from obsidian_llm_wiki.config import Config, load_config

__all__ = ["resolve_vault", "print_result_summary"]


def resolve_vault(vault: str) -> tuple[Path, Config]:
    """Resolve vault path and load config.  Returns (vault_path, config)."""
    vault_path = Path(vault).expanduser().resolve()
    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    config = load_config(env_file=env_file, VAULT_PATH=str(vault_path))
    return vault_path, config


def print_result_summary(result, *, additional_errors: Sequence[str] = ()) -> None:
    """Pretty-print a CompileResult."""
    errors = [*additional_errors, *result.errors]
    status = "❌ Finished with errors" if errors else "✅ Complete"
    print(
        f"\n{status}: "
        f"{result.compiled} compiled, "
        f"{len(result.concepts)} concepts, "
        f"{result.deleted} deleted"
    )
    if result.skipped:
        print(f"   Skipped: {result.skipped} (unchanged)")
    if errors:
        print(f"   Errors:  {len(errors)}")
        for err in errors[:10]:
            print(f"     - {err}")
        if len(errors) > 10:
            print(f"     ... and {len(errors) - 10} more")
