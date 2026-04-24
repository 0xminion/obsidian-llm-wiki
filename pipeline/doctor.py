"""First-run and configuration diagnostics."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pipeline.config import Config

_SECRET_FIELDS = {"transcript_api_key", "supadata_api_key", "assemblyai_api_key", "llm_api_key"}


def _redacted_config(cfg: Config) -> dict[str, Any]:
    data = asdict(cfg)
    for key in list(data):
        value = data[key]
        if isinstance(value, Path):
            data[key] = str(value)
        elif key in _SECRET_FIELDS:
            data[key] = "[REDACTED]" if value else ""
    return data


def _check(name: str, ok: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "ok": ok, "severity": severity, "detail": detail}


def run_doctor(cfg: Config) -> dict[str, Any]:
    """Return machine-readable diagnostics for first-run support and config drift."""
    checks: list[dict[str, Any]] = []
    checks.append(_check("vault_exists", cfg.vault_path.exists(), str(cfg.vault_path)))
    checks.append(_check("wiki_dir", cfg.sources_dir.parent.exists(), str(cfg.sources_dir.parent)))
    for label, path in [
        ("inbox_dir", cfg.inbox_dir),
        ("entries_dir", cfg.entries_dir),
        ("sources_dir", cfg.sources_dir),
        ("concepts_dir", cfg.concepts_dir),
        ("mocs_dir", cfg.mocs_dir),
        ("config_dir", cfg.config_dir),
    ]:
        checks.append(_check(label, path.exists(), str(path)))
    for cmd in ("curl", "python3"):
        checks.append(_check(f"command_{cmd}", shutil.which(cmd) is not None, cmd))
    if cfg.llm_provider == "openrouter":
        checks.append(_check("llm_api_key", bool(cfg.llm_api_key), "required for openrouter"))
    else:
        checks.append(_check("llm_api_key", True, "not required for current provider", "info"))
    for err in cfg.validate():
        checks.append(_check("config_validate", False, err))

    ok = all(c["ok"] or c["severity"] in {"warning", "info"} for c in checks)
    return {
        "ok": ok,
        "vault_path": str(cfg.vault_path),
        "extract_dir": str(cfg.resolved_extract_dir),
        "config": _redacted_config(cfg),
        "checks": checks,
    }
