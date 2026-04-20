"""Vault structure detection, setup, and migration.

Three states:
  - "new"      — vault path doesn't exist or is empty → full setup
  - "existing" — all required dirs present → skip
  - "incomplete" — some dirs missing → offer migration
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Canonical vault structure — every dir the pipeline touches
REQUIRED_DIRS = [
    "01-Raw",
    "02-Clippings",
    "03-Queries",
    "04-Wiki/sources",
    "04-Wiki/entries",
    "04-Wiki/concepts",
    "04-Wiki/mocs",
    "05-Outputs/answers",
    "05-Outputs/visualizations",
    "06-Config",
    "07-WIP",
    "08-Archive-Raw",
    "09-Archive-Queries",
    "Meta/Scripts",
    "Meta/Templates",
    "Meta/lib",
    "Meta/prompts",
]

# Dirs that must exist for the pipeline to not crash (minimal subset)
CRITICAL_DIRS = [
    "01-Raw",
    "04-Wiki/sources",
    "04-Wiki/entries",
    "04-Wiki/concepts",
    "04-Wiki/mocs",
    "06-Config",
    "08-Archive-Raw",
]

# Seed files — created on setup, never overwritten
_SEED_FILES: dict[str, str] = {
    "06-Config/edges.tsv": "source\ttarget\ttype\tdescription\n",
    "06-Config/wiki-index.md": "# Wiki Index\n\nAuto-generated. Do not edit manually.\n",
    "06-Config/url-index.tsv": "url\tfilename\thash\tdate\n",
    "06-Config/log.md": "# Pipeline Log\n\n",
    "06-Config/tag-registry.md": "# Tag Registry\n\nCanonical tags used across the vault.\n",
}

# Files that setup.sh copies from repo → vault (Python replaces these)
_REPO_COPY_MAP = {
    "prompts/": "Meta/prompts/",
    "templates/": "Meta/Templates/",
    "lib/": "Meta/lib/",
    "scripts/": "Meta/Scripts/",
}


class VaultState:
    """Result of vault detection."""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.exists = vault_path.exists()
        self.is_empty = not any(vault_path.iterdir()) if self.exists else True
        self.missing_dirs: list[str] = []
        self.extra_dirs: list[str] = []
        self.missing_files: list[str] = []
        self._check()

    def _check(self) -> None:
        if not self.exists or self.is_empty:
            return
        for d in REQUIRED_DIRS:
            if not (self.vault_path / d).is_dir():
                self.missing_dirs.append(d)
        for f, _ in _SEED_FILES.items():
            if not (self.vault_path / f).exists():
                self.missing_files.append(f)

    @property
    def state(self) -> str:
        if not self.exists or self.is_empty:
            return "new"
        if not self.missing_dirs:
            return "existing"
        # Check if it's a vault at all (has 04-Wiki or 01-Raw)
        has_wiki = (self.vault_path / "04-Wiki").is_dir()
        has_raw = (self.vault_path / "01-Raw").is_dir()
        if has_wiki or has_raw:
            return "incomplete"
        return "new"

    @property
    def summary(self) -> str:
        if self.state == "new":
            return f"Fresh vault at {self.vault_path} — no structure detected"
        if self.state == "existing":
            return f"Vault at {self.vault_path} — all directories present"
        lines = [f"Incomplete vault at {self.vault_path}:"]
        if self.missing_dirs:
            lines.append(f"  Missing dirs: {', '.join(self.missing_dirs)}")
        if self.missing_files:
            lines.append(f"  Missing files: {', '.join(self.missing_files)}")
        return "\n".join(lines)


def detect_vault(vault_path: Path) -> VaultState:
    """Detect vault state without modifying anything."""
    return VaultState(vault_path)


def setup_vault(vault_path: Path, repo_root: Optional[Path] = None, quiet: bool = False) -> list[str]:
    """Create full vault structure. Returns list of actions taken.

    Args:
        vault_path: Target vault directory.
        repo_root: Path to obsidian-automation repo (for copying prompts/templates).
        quiet: If True, don't log individual actions.

    Returns:
        List of human-readable action descriptions.
    """
    actions: list[str] = []

    # 1. Create directories
    for d in REQUIRED_DIRS:
        target = vault_path / d
        if not target.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            actions.append(f"Created directory: {d}")

    # 2. Create seed files (never overwrite)
    for relpath, content in _SEED_FILES.items():
        target = vault_path / relpath
        if not target.exists():
            target.write_text(content, encoding="utf-8")
            actions.append(f"Created seed file: {relpath}")

    # 3. Copy repo files if repo_root provided
    if repo_root and repo_root.is_dir():
        for src_dir, dst_dir in _REPO_COPY_MAP.items():
            src = repo_root / src_dir
            dst = vault_path / dst_dir
            if not src.is_dir():
                continue
            for src_file in src.glob("*"):
                if not src_file.is_file():
                    continue
                dst_file = dst / src_file.name
                if not dst_file.exists() or dst_file.read_bytes() != src_file.read_bytes():
                    shutil.copy2(src_file, dst_file)
                    actions.append(f"Copied: {dst_dir}{src_file.name}")

    # 4. Create .env if missing
    env_path = vault_path / "Meta/Scripts/.env"
    if not env_path.exists():
        env_example = (repo_root / ".env.example") if repo_root else None
        if env_example and env_example.exists():
            shutil.copy2(env_example, env_path)
            actions.append("Created .env from .env.example")
        else:
            env_path.write_text(
                "# API Keys\nTRANSCRIPT_API_KEY=\nSU..._API_KEY=\nASSEMBLYAI_API_KEY=\n"
                "# Vault path\nVAULT_PATH=$HOME/MyVault\n"
                "# Agent\nAGENT_CMD=hermes\n# Parallelism\nPARALLEL=3\n",
                encoding="utf-8",
            )
            actions.append("Created .env template")

    # 5. Create run.sh wrapper
    run_sh = vault_path / "run.sh"
    if not run_sh.exists():
        run_sh.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'export VAULT_PATH="$SCRIPT_DIR"\n'
            'if command -v pipeline &>/dev/null; then\n'
            '  exec pipeline ingest "$SCRIPT_DIR" "$@"\n'
            'elif python3 -c "import pipeline.cli" 2>/dev/null; then\n'
            '  exec python3 -m pipeline.cli ingest "$SCRIPT_DIR" "$@"\n'
            'else\n'
            '  echo "ERROR: Python pipeline not found." >&2\n'
            '  exit 1\n'
            'fi\n',
            encoding="utf-8",
        )
        run_sh.chmod(0o755)
        actions.append("Created run.sh wrapper")

    if not quiet:
        for a in actions:
            log.info(a)

    return actions


def migrate_vault(vault_path: Path, state: VaultState, repo_root: Optional[Path] = None) -> list[str]:
    """Migrate an incomplete vault — add missing dirs/files, never delete.

    Returns list of actions taken.
    """
    actions: list[str] = []

    for d in state.missing_dirs:
        target = vault_path / d
        target.mkdir(parents=True, exist_ok=True)
        actions.append(f"Created missing directory: {d}")

    for relpath, content in _SEED_FILES.items():
        target = vault_path / relpath
        if not target.exists():
            target.write_text(content, encoding="utf-8")
            actions.append(f"Created missing file: {relpath}")

    # Copy repo files
    if repo_root and repo_root.is_dir():
        for src_dir, dst_dir in _REPO_COPY_MAP.items():
            src = repo_root / src_dir
            dst = vault_path / dst_dir
            if not src.is_dir():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for src_file in src.glob("*"):
                if not src_file.is_file():
                    continue
                dst_file = dst / src_file.name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    actions.append(f"Copied: {dst_dir}{src_file.name}")

    for a in actions:
        log.info(a)

    return actions


def ensure_vault_ready(vault_path: Path, repo_root: Optional[Path] = None, force: bool = False) -> str:
    """Entry point: detect → setup/migrate/skip. Returns state string.

    - "new"      → setup performed
    - "existing" → nothing done
    - "migrated" → incomplete vault fixed
    - "failed"   → user rejected migration (when interactive)

    Args:
        vault_path: Vault directory path.
        repo_root: Repo root for copying files.
        force: If True, auto-migrate without asking.
    """
    state = detect_vault(vault_path)

    if state.state == "existing":
        log.info("Vault ready: %s", state.summary)
        return "existing"

    if state.state == "new":
        log.info("Setting up new vault at %s", vault_path)
        actions = setup_vault(vault_path, repo_root=repo_root)
        log.info("Setup complete: %d actions", len(actions))
        return "new"

    # Incomplete — needs migration
    log.warning("%s", state.summary)

    if force:
        actions = migrate_vault(vault_path, state, repo_root=repo_root)
        log.info("Migration complete: %d actions", len(actions))
        return "migrated"

    # Interactive prompt
    print(f"\n{state.summary}")
    print(f"\nMissing directories prevent pipeline from running.")
    response = input("Migrate vault structure? [Y/n] ").strip().lower()
    if response in ("", "y", "yes"):
        actions = migrate_vault(vault_path, state, repo_root=repo_root)
        print(f"Migration complete: {len(actions)} actions taken.")
        return "migrated"
    else:
        print("Migration skipped. Pipeline may fail.")
        return "failed"
