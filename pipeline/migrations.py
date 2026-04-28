"""Versioned vault migrations for installed CLI maintenance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import Config
from pipeline.vault_setup import ensure_vault_ready

CURRENT_SCHEMA_VERSION = 1


def _version_file(cfg: Config) -> Path:
    return cfg.config_dir / "schema-version.json"


def read_schema_version(cfg: Config) -> int:
    path = _version_file(cfg)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return int(data.get("schema_version", 0) or 0)


def migrate_vault_schema(cfg: Config, *, yes: bool = False) -> dict:
    """Apply idempotent vault migrations and record schema version."""
    actions: list[str] = []
    state = ensure_vault_ready(cfg.vault_path, force=True)
    actions.append(f"vault_ready:{state}")
    cfg.config_dir.mkdir(parents=True, exist_ok=True)

    before = read_schema_version(cfg)
    after = max(before, CURRENT_SCHEMA_VERSION)
    path = _version_file(cfg)
    payload = {
        "schema_version": after,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "migrations": ["vault-structure-assets-backfill"] if before < CURRENT_SCHEMA_VERSION else [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if before < CURRENT_SCHEMA_VERSION:
        actions.append(f"schema_version:{before}->{after}")
    return {
        "ok": True,
        "schema_version": after,
        "previous_schema_version": before,
        "actions": actions,
        "version_file": str(path),
    }
