"""Typed app navigation, separate from opaque prose and semantic identity.

Only emission sites call this module. Never discover links by searching a
rendered document: source bodies, evidence, code and human references are data.
The context exists only during rendering, not during synthesis/review/cache keys.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from obsidian_llm_wiki.core.generated_paths import checked_generated_path
from obsidian_llm_wiki.render.frontmatter import extract_wikilinks, make_wikilink

LinkKind = Literal["sources", "entries", "concepts", "mocs", "views", "root"]
_FOLDERS = {kind: kind for kind in ("sources", "entries", "concepts", "mocs", "views")}
_FOLDERS["root"] = ""


@dataclass
class LinkPresentation:
    """Vault-root presentation prefix and an inventory of app-emitted targets."""

    prefix: str
    targets: set[str] = field(default_factory=set)

    def payload(self) -> dict:
        return {"format": "typed-generated-links-v1", "prefix": self.prefix,
                "targets": sorted(self.targets)}


ACTIVE_LINK_PRESENTATION: ContextVar[LinkPresentation | None] = ContextVar(
    "active_link_presentation", default=None,
)


def generated_link(
    kind: LinkKind, slug: str, alias: str | None = None, *, legacy_target: str | None = None,
) -> str:
    """Emit an explicitly typed target; no basename search or priority fallback.

    Standalone legacy render APIs keep their historical spelling. Immutable
    generation renderers set the presentation context and require exact files.
    """
    if kind not in _FOLDERS:
        raise ValueError(f"untyped generated wikilink: {kind}")
    stem, anchor, heading = slug.partition("#")
    if not stem or any(c in stem for c in "/\\|[]\n\r") or stem in (".", ".."):
        raise ValueError(f"invalid generated wikilink target: {slug}")
    presentation = ACTIVE_LINK_PRESENTATION.get()
    if presentation is None:
        return make_wikilink(legacy_target or slug, alias)
    relative = str(PurePosixPath(_FOLDERS[kind]) / (stem.removesuffix(".md") + ".md"))
    presentation.targets.add(relative)
    target = presentation.prefix + "/" + relative.removesuffix(".md")
    if anchor:
        target += anchor + heading
    return make_wikilink(target, alias)


def generated_folder(kind: LinkKind) -> str:
    """Vault-root folder for app-generated Dataview queries (not captured code)."""
    folder = _FOLDERS[kind]
    presentation = ACTIVE_LINK_PRESENTATION.get()
    return presentation.prefix + "/" + folder if presentation else folder


def navigation_shards(content: str, kind: LinkKind, starts_with: tuple[str, ...]) -> set[str]:
    """Read shard names only from app navigation, including copied generations.

    This is inventory discovery for cleanup, never a Markdown transformation.
    Ownership checks still guard any deletion. Other namespaces stay pinned.
    """
    stems = set()
    for target, _alias in extract_wikilinks(content):
        parts = PurePosixPath(target).parts
        if len(parts) == 1:
            stem = parts[0]
        elif (len(parts) == 6 and parts[:3] == ("04-Wiki", "OLW-Publications", "generations")
              and parts[4] == kind):
            stem = parts[5]
        else:
            continue
        if stem.startswith(starts_with) and not any(c in stem for c in "#\\[]\n\r"):
            stems.add(stem.removesuffix(".md"))
    return stems


def validate_link_targets(wiki: Path, payload: dict, *, prefix: str) -> None:
    """Fail closed on a missing/wrong exact typed destination, without editing text."""
    if (payload.get("format") != "typed-generated-links-v1"
            or payload.get("prefix") != prefix or not isinstance(payload.get("targets"), list)):
        raise RuntimeError("invalid generated link inventory")
    for relative in payload["targets"]:
        if not isinstance(relative, str):
            raise RuntimeError("invalid generated link target")
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != relative
                or path.suffix != ".md" or len(path.parts) not in (1, 2)
                or (len(path.parts) == 2 and path.parts[0] not in _FOLDERS)):
            raise RuntimeError(f"invalid generated link target: {relative}")
        destination = checked_generated_path(wiki / relative, wiki)
        if not destination.is_file():
            raise RuntimeError(f"missing generated wikilink target: {relative}")


def validate_link_inventory(wiki: Path, prefix: str) -> None:
    """Validate the renderer's inventory at the final publication seam."""
    path = checked_generated_path(wiki / ".llmwiki/generated-links.json", wiki)
    # Old direct callers of the finalization seam may have no render inventory.
    # New immutable builds always produce one inside the render transaction.
    if path.is_file():
        validate_link_targets(wiki, json.loads(path.read_text()), prefix=prefix)
    elif (wiki / ".llmwiki/identity-publication.json").exists():
        raise RuntimeError("missing generated link inventory")
