"""Release hygiene checks for version/docs/package alignment."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def check_release_hygiene(repo_root: Path) -> dict[str, Any]:
    """Check cheap release metadata invariants before tagging or publishing."""
    repo_root = Path(repo_root)
    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = str(data["project"]["version"])
        source_tree = True
    else:
        try:
            version = metadata.version("obsidian-llm-wiki")
        except metadata.PackageNotFoundError:
            version = "unknown"
        source_tree = False
    checks = [
        _check("pyproject_version", bool(re.match(r"^\d+\.\d+\.\d+", version)), version),
        _check("source_tree", source_tree, "source checkout" if source_tree else "installed package; docs checks skipped"),
    ]
    if not source_tree:
        return {"ok": checks[0]["ok"], "version": version, "checks": checks}
    checks.extend([
        _check("readme_exists", (repo_root / "README.md").exists(), "README.md"),
        _check("docs_index", (repo_root / "docs" / "README.md").exists(), "docs/README.md"),
        _check("changelog", (repo_root / "docs" / "release" / "CHANGELOG.md").exists(), "docs/release/CHANGELOG.md"),
        _check("release_notes", (repo_root / "docs" / "release" / "RELEASE.md").exists(), "docs/release/RELEASE.md"),
        _check("architecture", (repo_root / "docs" / "architecture" / "ARCHITECTURE.md").exists(), "docs/architecture/ARCHITECTURE.md"),
    ])
    changelog = repo_root / "docs" / "release" / "CHANGELOG.md"
    if changelog.exists():
        text = changelog.read_text(encoding="utf-8", errors="replace")
        checks.append(_check("version_in_changelog", version in text or "Unreleased" in text, version))
    return {"ok": all(c["ok"] for c in checks), "version": version, "checks": checks}
