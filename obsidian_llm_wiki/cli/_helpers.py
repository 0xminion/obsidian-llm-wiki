"""Shared CLI helpers."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.config import Config, load_config

__all__ = ["resolve_vault", "print_result_summary"]


def resolve_vault(vault: str) -> tuple[Path, Config]:
    """Resolve vault path and load config.  Returns (vault_path, config)."""
    vault_path = Path(vault).expanduser().resolve()
    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    config = load_config(env_file=env_file, VAULT_PATH=str(vault_path))
    return vault_path, config


def print_result_summary(result) -> None:
    """Pretty-print a CompileResult."""
    print(
        f"\n✅ Complete: "
        f"{result.compiled} compiled, "
        f"{len(result.concepts)} concepts, "
        f"{result.deleted} deleted"
    )
    if result.skipped:
        print(f"   Skipped: {result.skipped} (unchanged)")
    if result.errors:
        print(f"   Errors:  {len(result.errors)}")
        for err in result.errors[:10]:
            print(f"     - {err}")
        if len(result.errors) > 10:
            print(f"     ... and {len(result.errors) - 10} more")
