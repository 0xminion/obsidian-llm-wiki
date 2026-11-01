"""Boundary for supported domain writers, not a filesystem access control."""

from __future__ import annotations

import json
from pathlib import Path

from obsidian_llm_wiki.core.generated_paths import checked_generated_path
from obsidian_llm_wiki.core.immutable_publication import NAMESPACE, OWNER


def require_domain_workspace(path: Path) -> None:
    """Refuse installed/legacy paths in a vault with a publication namespace.

    The publisher's private staging vault remains writable. Namespace ownership
    is inspected read-only; no registry or publisher constructors are opened.
    """
    path = Path(path).absolute()
    checked_generated_path(path)
    for wiki in (path, *path.parents):
        if wiki.name != "04-Wiki":
            continue
        root = checked_generated_path(wiki / NAMESPACE)
        if not root.exists():
            continue
        marker = checked_generated_path(root / "owner.json")
        try:
            owned = json.loads(marker.read_bytes()) == OWNER
        except (OSError, ValueError):
            owned = False
        if not owned:
            raise RuntimeError(f"publication namespace collision: {root}")
        relative = path.relative_to(wiki)
        parts = relative.parts
        if (len(parts) >= 4 and parts[0] == NAMESPACE and parts[1] == ".staging"
                and parts[2] and parts[3] == "04-Wiki"):
            continue
        raise RuntimeError(
            "immutable publication or legacy page cannot be modified by this domain API; "
            "use the build pipeline to publish a new generation"
        )
