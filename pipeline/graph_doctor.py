"""Graph integrity diagnostics for vault notes and derived edge artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.config import Config

_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def _note_dirs(cfg: Config) -> list[tuple[Path, str]]:
    return [
        (cfg.sources_dir, "source"),
        (cfg.entries_dir, "entry"),
        (cfg.concepts_dir, "concept"),
        (cfg.mocs_dir, "moc"),
    ]


def collect_graph_diagnostics(cfg: Config) -> dict:
    """Return unresolved wikilinks, stale edges, and duplicate Obsidian stems."""
    notes: dict[str, dict] = {}
    duplicate_stems: dict[str, list[str]] = {}
    unresolved_links: list[dict] = []

    for note_dir, note_type in _note_dirs(cfg):
        if not note_dir.exists():
            continue
        for md in sorted(note_dir.glob("*.md")):
            rel = str(md.relative_to(cfg.vault_path))
            duplicate_stems.setdefault(md.stem, []).append(rel)
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            notes[md.stem] = {"path": rel, "type": note_type, "links": sorted(set(_LINK_RE.findall(text)))}

    duplicate_stems = {stem: paths for stem, paths in duplicate_stems.items() if len(paths) > 1}
    for stem, info in notes.items():
        for target in info["links"]:
            if target not in notes:
                unresolved_links.append({"source": stem, "target": target, "path": info["path"]})

    stale_edges: list[dict] = []
    malformed_edges: list[dict] = []
    if cfg.edges_file.exists():
        for line_no, line in enumerate(cfg.edges_file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip() or line.startswith("source\t") or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                malformed_edges.append({"line": line_no, "content": line})
                continue
            source, target, edge_type = parts[:3]
            if source not in notes or target not in notes:
                stale_edges.append({"line": line_no, "source": source, "target": target, "type": edge_type})

    report = {
        "ok": not unresolved_links and not stale_edges and not malformed_edges and not duplicate_stems,
        "notes": len(notes),
        "unresolved_links": unresolved_links,
        "stale_edges": stale_edges,
        "malformed_edges": malformed_edges,
        "duplicate_stems": duplicate_stems,
    }
    return report
