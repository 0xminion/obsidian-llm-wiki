"""Single source of truth for package version.

Prefer local source (pyproject.toml next to this file) for development,
fallback to importlib.metadata for installed packages.
"""

from __future__ import annotations

import os

def _read_local_version() -> str | None:
    """Read version from pyproject.toml in the source tree."""
    try:
        import tomllib
    except ImportError:
        return None
    # Traverse up from this file to find pyproject.toml
    path = os.path.abspath(__file__)
    for _ in range(3):
        dir_path = os.path.dirname(path)
        toml = os.path.join(dir_path, "pyproject.toml")
        if os.path.exists(toml):
            try:
                with open(toml, "rb") as f:
                    data = tomllib.load(f)
                return data.get("project", {}).get("version")
            except Exception:
                return None
        path = dir_path
    return None


def _read_installed_version() -> str | None:
    """Read version from installed package metadata."""
    try:
        from importlib.metadata import PackageNotFoundError, version as _version
        return _version("obsidian-llm-wiki")
    except (PackageNotFoundError, ImportError):
        return None


__version__ = _read_local_version() or _read_installed_version() or "unknown"
