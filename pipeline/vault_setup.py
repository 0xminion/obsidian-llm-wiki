"""Vault structure detection, setup, and migration.

Three states:
  - "new"      — vault path doesn't exist or is empty → full setup
  - "existing" — all required dirs present → skip
  - "incomplete" — some dirs missing → offer migration
"""

from __future__ import annotations

import logging
import re
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
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
        if not self.missing_dirs and not self.missing_files:
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
        repo_root: Path to obsidian-llm-wiki repo (for copying prompts/templates).
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
                "# API Keys\n"
                "TRANSCRIPT_API_KEY=***\n"
                "SUPADATA_API_KEY=***\n"
                "ASSEMBLYAI_API_KEY=***\n\n"
                "# Vault path\n"
                "VAULT_PATH=$HOME/MyVault\n\n"
                "# Agent\n"
                "AGENT_CMD=hermes\n"
                "# Parallelism\n"
                "PARALLEL=3\n",
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
    print("\nMissing directories prevent pipeline from running.")
    response = input("Migrate vault structure? [Y/n] ").strip().lower()
    if response in ("", "y", "yes"):
        actions = migrate_vault(vault_path, state, repo_root=repo_root)
        print(f"Migration complete: {len(actions)} actions taken.")
        return "migrated"
    else:
        print("Migration skipped. Pipeline may fail.")
        return "failed"


# ═══════════════════════════════════════════════════════════
# SCAN / FIX / MIGRATE — Ported from scripts/migrate-vault.sh
# ═══════════════════════════════════════════════════════════

# Directories to scan for note issues
_SCAN_DIRS = [
    "04-Wiki/entries",
    "04-Wiki/concepts",
    "04-Wiki/mocs",
    "04-Wiki/sources",
    "entries",
    "concepts",
    "mocs",
    "sources",
    "notes",
    "wiki",
]

# Basenames to skip during scanning
_SKIP_BASENAMES = re.compile(
    r"^(README|CHANGELOG|LICENSE|CODE_REVIEW|PRD)\.md$"
)

# Path components to skip
_SKIP_PATH_PATTERNS = [
    "/v1/",
    "/.git/",
    "/06-Config/",
    "/08-Archive",
    "/09-Archive",
    "/Meta/",
]

# Required dirs matching shell script
_EXPECTED_DIRS = [
    "01-Raw",
    "02-Clippings",
    "03-Queries",
    "04-Wiki/entries",
    "04-Wiki/concepts",
    "04-Wiki/mocs",
    "04-Wiki/sources",
    "05-Outputs",
    "06-Config",
    "07-WIP",
    "08-Archive-Raw",
    "09-Archive-Queries",
]


@dataclass
class ScanIssue:
    """A single issue found during vault scan."""
    file_path: str
    issue_type: str  # "no_frontmatter", "missing_field"
    field: str = ""  # field name for missing_field type
    note_type: str = ""  # "entry", "concept", "moc", "unknown"

    def __str__(self) -> str:
        if self.issue_type == "no_frontmatter":
            return f"No frontmatter: {self.file_path}"
        return f"Missing `{self.field}:` ({self.note_type}): {self.file_path}"


@dataclass
class ScanResult:
    """Result of scanning a vault."""
    total_notes: int = 0
    issues: list[ScanIssue] = field(default_factory=list)
    missing_dirs: list[str] = field(default_factory=list)
    scanned_dirs: list[str] = field(default_factory=list)

    @property
    def missing_frontmatter_count(self) -> int:
        return sum(1 for i in self.issues if i.issue_type == "no_frontmatter")

    @property
    def missing_reviewed_count(self) -> int:
        return sum(1 for i in self.issues if i.field == "reviewed")

    @property
    def missing_template_count(self) -> int:
        return sum(1 for i in self.issues if i.field == "template")

    @property
    def missing_review_notes_count(self) -> int:
        return sum(1 for i in self.issues if i.field == "review_notes")

    @property
    def missing_aliases_count(self) -> int:
        return sum(1 for i in self.issues if i.field == "aliases")

    def summary_lines(self) -> list[str]:
        """Generate human-readable summary lines."""
        lines = [
            f"Notes scanned: {self.total_notes}",
            f"Missing frontmatter: {self.missing_frontmatter_count}",
            f"Missing reviewed: {self.missing_reviewed_count}",
            f"Missing template: {self.missing_template_count}",
            f"Missing review_notes: {self.missing_review_notes_count}",
            f"Missing aliases: {self.missing_aliases_count}",
        ]
        if self.missing_dirs:
            lines.append(f"Missing directories: {len(self.missing_dirs)}")
        return lines


def _has_frontmatter(content: str) -> bool:
    """Check if content starts with YAML frontmatter."""
    return content.startswith("---")


def _has_field(content: str, field_name: str) -> bool:
    """Check if a field exists in YAML frontmatter."""
    pattern = rf"^{re.escape(field_name)}:"
    return bool(re.search(pattern, content, re.MULTILINE))


def _classify_note(content: str, file_path: Path) -> str:
    """Classify a note as entry, concept, moc, or unknown."""
    dir_name = file_path.parent.name
    if dir_name == "entries":
        return "entry"
    if dir_name == "concepts":
        return "concept"
    if dir_name == "mocs":
        return "moc"

    # Heuristic from frontmatter
    if _has_frontmatter(content):
        status_match = re.search(r"^status:\s*(review|evergreen|seed)$", content, re.MULTILINE)
        has_source = _has_field(content, "source")
        if status_match and has_source:
            return "entry"
        if re.search(r"^entry_refs:", content, re.MULTILINE):
            return "concept"
        if re.search(r"^type:\s*moc$", content, re.MULTILINE):
            return "moc"
    return "unknown"


def _should_skip_file(file_path: Path, vault_path: Path) -> bool:
    """Check if a file should be skipped during scanning."""
    basename = file_path.name
    if _SKIP_BASENAMES.match(basename):
        return True
    rel = str(file_path.relative_to(vault_path))
    for pattern in _SKIP_PATH_PATTERNS:
        if pattern in rel:
            return True
    return False


def scan_vault(vault_path: Path) -> ScanResult:
    """Scan vault for issues: missing frontmatter, missing fields.

    Args:
        vault_path: Path to the Obsidian vault root.

    Returns:
        ScanResult with issues found.
    """
    vault_path = Path(vault_path)
    result = ScanResult()

    if not vault_path.is_dir():
        log.error("Vault path does not exist: %s", vault_path)
        return result

    # Find scan directories
    scan_dirs = []
    for d in _SCAN_DIRS:
        candidate = vault_path / d
        if candidate.is_dir():
            scan_dirs.append(candidate)
    result.scanned_dirs = [str(d.relative_to(vault_path)) for d in scan_dirs]

    # If no standard dirs, scan vault root
    if not scan_dirs:
        scan_dirs = [vault_path]
        result.scanned_dirs = ["."]

    # Check directory structure
    for d in _EXPECTED_DIRS:
        if not (vault_path / d).is_dir():
            result.missing_dirs.append(d)

    # Scan each .md file
    for scan_dir in scan_dirs:
        for md_file in sorted(scan_dir.rglob("*.md")):
            if _should_skip_file(md_file, vault_path):
                continue

            result.total_notes += 1
            rel_path = str(md_file.relative_to(vault_path))

            try:
                content = md_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                log.warning("Could not read %s: %s", md_file, e)
                continue

            if not _has_frontmatter(content):
                result.issues.append(ScanIssue(
                    file_path=rel_path,
                    issue_type="no_frontmatter",
                ))
                continue

            note_type = _classify_note(content, md_file)

            # Check required fields for entry/concept notes
            if note_type == "entry":
                for fld in ("reviewed", "template", "review_notes", "aliases"):
                    if not _has_field(content, fld):
                        result.issues.append(ScanIssue(
                            file_path=rel_path,
                            issue_type="missing_field",
                            field=fld,
                            note_type="entry",
                        ))
            elif note_type == "concept":
                if not _has_field(content, "aliases"):
                    result.issues.append(ScanIssue(
                        file_path=rel_path,
                        issue_type="missing_field",
                        field="aliases",
                        note_type="concept",
                    ))

    return result


def fix_frontmatter(file_path: Path, fixes: dict[str, str | None]) -> bool:
    """Patch frontmatter in a markdown file in-place.

    Args:
        file_path: Path to the .md file.
        fixes: Dict mapping field names to default values.
               e.g. {"reviewed": "null", "template": "standard"}
               Value of None means "[]" for aliases.

    Returns:
        True if any changes were made.
    """
    file_path = Path(file_path)
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        log.error("Could not read %s: %s", file_path, e)
        return False

    if not _has_frontmatter(content):
        return False

    # Parse frontmatter boundaries
    first_break = content.index("---")
    second_break = content.index("---", first_break + 3)
    fm_body = content[first_break + 3:second_break]
    after_fm = content[second_break:]

    changes_made = False
    fm_lines = fm_body.split("\n")

    for field_name, default_value in fixes.items():
        if _has_field(content, field_name):
            continue

        value = default_value if default_value is not None else "[]"
        field_line = f"{field_name}: {value}"

        # Find insertion point: after status:, after reviewed:, or before closing ---
        inserted = False
        if field_name == "reviewed":
            # Insert after status: line
            for i, line in enumerate(fm_lines):
                if line.startswith("status:"):
                    fm_lines.insert(i + 1, field_line)
                    inserted = True
                    break
        elif field_name == "review_notes":
            # Insert after reviewed: line
            for i, line in enumerate(fm_lines):
                if line.startswith("reviewed:"):
                    fm_lines.insert(i + 1, field_line)
                    inserted = True
                    break
        elif field_name == "template":
            # Insert after review_notes: or reviewed:
            for anchor in ("review_notes:", "reviewed:"):
                for i, line in enumerate(fm_lines):
                    if line.startswith(anchor):
                        fm_lines.insert(i + 1, field_line)
                        inserted = True
                        break
                if inserted:
                    break
        elif field_name == "aliases":
            # Insert before closing (at end of frontmatter)
            fm_lines.append(field_line)
            inserted = True

        if not inserted:
            # Fallback: append before closing
            fm_lines.append(field_line)
            changes_made = True
        else:
            changes_made = True

    if changes_made:
        new_fm_body = "\n".join(fm_lines)
        new_content = "---" + new_fm_body + after_fm
        file_path.write_text(new_content, encoding="utf-8")

    return changes_made


def _build_url_index(vault_path: Path) -> int:
    """Build url-index.tsv from existing source files. Returns entry count."""
    url_index = vault_path / "06-Config" / "url-index.tsv"
    entries: dict[str, str] = {}

    for source_dir in [vault_path / "01-Raw", vault_path / "04-Wiki" / "sources"]:
        if not source_dir.is_dir():
            continue
        for md_file in source_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            m = re.search(r"^source_url:\s*[\"']?(.*?)[\"']?\s*$", content, re.MULTILINE)
            if m:
                url = m.group(1).strip()
                if url:
                    # Normalize URL
                    normalized = re.sub(r"^https?://", "", url)
                    normalized = normalized.rstrip("/")
                    if normalized not in entries:
                        entries[normalized] = str(md_file.name)

    if entries:
        url_index.parent.mkdir(parents=True, exist_ok=True)
        lines = ["url\tfilename\n"]
        for url, filename in sorted(entries.items()):
            lines.append(f"{url}\t{filename}\n")
        url_index.write_text("".join(lines), encoding="utf-8")

    return len(entries)


def _create_backup(vault_path: Path) -> Path:
    """Create a .tar.gz backup of 04-Wiki and 06-Config. Returns backup path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_file = vault_path / f"migration-backup-{timestamp}.tar.gz"

    with tarfile.open(backup_file, "w:gz") as tar:
        for subdir in ["04-Wiki", "06-Config"]:
            source = vault_path / subdir
            if source.is_dir():
                tar.add(source, arcname=subdir)

    return backup_file


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    issues_found: int = 0
    issues_fixed: int = 0
    notes_checked: int = 0
    backup_path: Optional[str] = None
    actions: list[str] = field(default_factory=list)
    scan_result: Optional[ScanResult] = None


def migrate_vault_full(
    vault_path: Path,
    dry_run: bool = False,
    backup: bool = True,
) -> MigrationResult:
    """Full vault migration: scan, fix frontmatter, build indexes, backup.

    Args:
        vault_path: Path to the Obsidian vault.
        dry_run: If True, return plan without applying changes.
        backup: If True, create backup before executing changes.

    Returns:
        MigrationResult with details of what was done.
    """
    vault_path = Path(vault_path)
    result = MigrationResult()

    if not vault_path.is_dir():
        log.error("Vault path does not exist: %s", vault_path)
        return result

    # Step 1: Scan
    scan = scan_vault(vault_path)
    result.scan_result = scan
    result.notes_checked = scan.total_notes
    result.issues_found = len(scan.issues)

    if dry_run:
        result.actions.append(f"Would scan {scan.total_notes} notes")
        result.actions.append(f"Found {len(scan.issues)} issues")

        for issue in scan.issues:
            if issue.issue_type == "no_frontmatter":
                result.actions.append(f"  SKIP (no frontmatter): {issue.file_path}")
            else:
                result.actions.append(f"  WOULD FIX: {issue.file_path} — add {issue.field}: null")

        # Directory structure
        if scan.missing_dirs:
            result.actions.append(f"Would create {len(scan.missing_dirs)} missing directories")
            for d in scan.missing_dirs:
                result.actions.append(f"  mkdir -p {d}")

        # URL index
        url_index = vault_path / "06-Config" / "url-index.tsv"
        if not url_index.exists():
            result.actions.append("Would build url-index.tsv from existing sources")
        else:
            result.actions.append("url-index.tsv exists — skipping")

        # Edges file
        edges = vault_path / "06-Config" / "edges.tsv"
        if not edges.exists():
            result.actions.append("Would create edges.tsv")

        return result

    # Step 2: Ensure directory structure
    for d in REQUIRED_DIRS:
        target = vault_path / d
        if not target.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            result.actions.append(f"Created directory: {d}")

    # Step 3: Fix frontmatter in entry and concept notes
    _DEFAULT_FIXES: dict[str, str | None] = {
        "reviewed": "null",
        "review_notes": "null",
        "template": "standard",
        "aliases": None,  # Means "[]"
    }

    for issue in scan.issues:
        if issue.issue_type == "no_frontmatter":
            result.actions.append(f"SKIP (no frontmatter): {issue.file_path} — needs manual review")
            continue

        note_path = vault_path / issue.file_path
        if not note_path.is_file():
            continue

        # Determine which fixes apply based on note type
        if issue.note_type == "entry":
            fixes = _DEFAULT_FIXES
        elif issue.note_type == "concept":
            fixes = {"aliases": None}
        else:
            continue

        if fix_frontmatter(note_path, fixes):
            result.issues_fixed += 1
            result.actions.append(f"FIXED: {issue.file_path}")

    # Step 4: Build URL index if missing
    url_index = vault_path / "06-Config" / "url-index.tsv"
    if not url_index.exists():
        count = _build_url_index(vault_path)
        result.actions.append(f"Built url-index.tsv ({count} entries)")
    else:
        result.actions.append("url-index.tsv exists — skipping")

    # Step 5: Ensure edges.tsv exists
    edges = vault_path / "06-Config" / "edges.tsv"
    if not edges.exists():
        edges.parent.mkdir(parents=True, exist_ok=True)
        edges.write_text("source\ttarget\ttype\tdescription\n", encoding="utf-8")
        result.actions.append("Created edges.tsv")

    # Step 6: Backup
    if backup:
        backup_path = _create_backup(vault_path)
        result.backup_path = str(backup_path)
        result.actions.append(f"Created backup: {backup_path.name}")

    return result
