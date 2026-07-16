"""Atomic, bounded backups for automated page rewrites.

Backups are grouped by a stable digest of the source path so same-named files
from different vault directories never overwrite each other.  This module only
copies data; it never restores or rewrites a vault page.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["backup_file", "create_backup", "list_backups"]


def _source_directory(source: Path, backups_root: Path) -> Path:
    key = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
    return backups_root / key


def _validate_max_backups(max_backups: int) -> None:
    if isinstance(max_backups, bool) or not isinstance(max_backups, int) or max_backups < 1:
        raise ValueError("max_backups must be a positive integer")


def list_backups(source_path: str | Path, backups_root: str | Path) -> list[Path]:
    """Return retained backups for a source in oldest-to-newest order."""
    source = Path(source_path)
    directory = _source_directory(source, Path(backups_root))
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.bak") if path.is_file())


def backup_file(
    source_path: str | Path,
    backups_root: str | Path,
    *,
    max_backups: int = 10,
) -> Path:
    """Atomically snapshot a regular source file and retain only newest backups.

    The snapshot is written under ``backups_root`` with a unique chronological
    name.  Rotation occurs only after the new backup has been atomically
    installed, so a failed write cannot discard a previously retained backup.
    """
    _validate_max_backups(max_backups)
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise ValueError(f"source must be a regular file: {source}")

    directory = _source_directory(source, Path(backups_root))
    directory.mkdir(parents=True, exist_ok=True)
    name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = f"{name}-{uuid.uuid4().hex}-"
    suffix = ".bak"
    max_source_name_bytes = 255 - len(os.fsencode(prefix + suffix))
    source_name = os.fsdecode(os.fsencode(source.name)[:max_source_name_bytes])
    destination = directory / f"{prefix}{source_name}{suffix}"

    fd, temporary_name = tempfile.mkstemp(dir=directory, prefix=".backup-", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_file, os.fdopen(fd, "wb") as output_file:
            while chunk := input_file.read(1024 * 1024):
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise

    backups = list_backups(source, backups_root)
    for old_backup in backups[:-max_backups]:
        old_backup.unlink()
    return destination


create_backup = backup_file
