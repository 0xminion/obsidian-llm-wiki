"""Vault stats module — dashboard metrics for the wiki.

Generates a health/growth dashboard at 06-Config/dashboard.md.
Consolidates vault-stats.sh into Python.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from pipeline.config import Config
from pipeline.utils import count_md, extract_frontmatter_field


def _edge_type_counts(cfg: Config) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not cfg.edges_file.exists():
        return counts
    for line in cfg.edges_file.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2]:
            counts[parts[2]] = counts.get(parts[2], 0) + 1
    return dict(sorted(counts.items()))


def collect_stats(cfg: Config) -> dict:
    """Collect dashboard metrics in a stable JSON-serializable shape."""
    entries = count_md(cfg.entries_dir)
    concepts = count_md(cfg.concepts_dir)
    mocs = count_md(cfg.mocs_dir)
    sources = count_md(cfg.sources_dir)
    return {
        "vault_path": str(cfg.vault_path),
        "entries": entries,
        "concepts": concepts,
        "mocs": mocs,
        "sources": sources,
        "total": entries + concepts + mocs + sources,
        "graph": {
            "node_types": ["concept", "entry", "moc", "source"],
            "edge_types": _edge_type_counts(cfg),
            "semantics": "source notes are first-class nodes; edges are wikilink-derived relationships between existing notes",
        },
    }


def generate_dashboard(cfg: Config) -> str:
    """Generate the dashboard markdown content."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"# Wiki Dashboard — {today}", ""]

    # ─── Vault Size ────────────────────────────────────────────────────────────
    stats = collect_stats(cfg)
    entries = stats["entries"]
    concepts = stats["concepts"]
    mocs = stats["mocs"]
    sources = stats["sources"]
    total = stats["total"]

    lines.extend([
        "## Vault Size",
        "",
        "| Type | Count |",
        "|------|-------|",
        f"| Entries | {entries} |",
        f"| Concepts | {concepts} |",
        f"| MoCs | {mocs} |",
        f"| Sources | {sources} |",
        f"| **Total** | **{total}** |",
        "",
    ])

    # ─── Growth (last 7 days) ──────────────────────────────────────────────────
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    recent_entries = 0
    if cfg.entries_dir.exists():
        for md in cfg.entries_dir.glob("*.md"):
            content = md.read_text(encoding="utf-8", errors="replace")
            entry_date = extract_frontmatter_field(content, "date_entry")
            if entry_date and entry_date > cutoff:
                recent_entries += 1

    lines.extend([
        "## Growth (last 7 days)",
        "",
        f"- New entries (7d): {recent_entries}",
        "",
    ])

    # ─── Review Status ─────────────────────────────────────────────────────────
    reviewed_count = 0
    unreviewed_count = 0
    if cfg.entries_dir.exists():
        for md in cfg.entries_dir.glob("*.md"):
            content = md.read_text(encoding="utf-8", errors="replace")
            reviewed = extract_frontmatter_field(content, "reviewed")
            if not reviewed or reviewed in ("", "null", "None"):
                unreviewed_count += 1
            else:
                reviewed_count += 1

    lines.extend([
        "## Review Status",
        "",
        "| Status | Count |",
        "|--------|-------|",
        f"| Reviewed | {reviewed_count} |",
        f"| Unreviewed | {unreviewed_count} |",
        "",
    ])

    # ─── Health Indicators ─────────────────────────────────────────────────────
    # Orphan check — O(N) indexed approach: build reference set once
    all_refs: set[str] = set()
    for ref_dir in (cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir):
        if not ref_dir.exists():
            continue
        for md in ref_dir.glob("*.md"):
            content = md.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r'\[\[([^\]]+)\]\]', content):
                ref = m.group(1).split('|')[0].split('#')[0]
                all_refs.add(ref)

    orphan_count = 0
    if cfg.entries_dir.exists():
        for md in cfg.entries_dir.glob("*.md"):
            if md.stem not in all_refs:
                orphan_count += 1

    # Edges count
    edge_count = 0
    if cfg.edges_file.exists():
        content = cfg.edges_file.read_text(encoding="utf-8", errors="replace").strip()
        edge_count = max(0, len(content.split("\n")) - 1)

    # Last ingest
    log_file = cfg.config_dir / "log.md"
    last_ingest = "never"
    if log_file.exists():
        log_content = log_file.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"^## \[.*?\] ingest", log_content, re.MULTILINE)
        if matches:
            last_ingest = matches[-1]

    # URL index size
    url_index_size = 0
    if cfg.url_index.exists():
        url_index_size = len(cfg.url_index.read_text(encoding="utf-8", errors="replace").strip().split("\n"))

    lines.extend([
        "## Health",
        "",
        "| Indicator | Status |",
        "|-----------|--------|",
        f"| Orphaned entries | {orphan_count} |",
        f"| Typed edges | {edge_count} |",
        f"| Last ingest | {last_ingest} |",
        f"| URL index size | {url_index_size} entries |",
        "",
    ])

    # ─── Knowledge Staleness ───────────────────────────────────────────────────
    try:
        from pipeline.lint import check_staleness
    except Exception:
        check_staleness = None  # type: ignore[misc]

    if check_staleness is not None:
        stale_issues = check_staleness(cfg.vault_path)
        stale_high = [i for i in stale_issues if i.severity.name == "WARNING"]
        stale_info = [i for i in stale_issues if i.severity.name == "INFO"]
        lines.extend([
            "## Knowledge Staleness",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Stale notes (high volatility) | {len(stale_high)} |",
            f"| Stale notes (medium/low) | {len(stale_info)} |",
            "",
        ])

    # ─── Graph Semantics ───────────────────────────────────────────────────────
    edge_types = stats["graph"]["edge_types"]
    lines.extend([
        "## Graph Semantics",
        "",
        "source notes are first-class nodes; edges are wikilink-derived relationships between existing notes.",
        "",
        "| Edge Type | Count |",
        "|-----------|-------|",
    ])
    if edge_types:
        for edge_type, count in edge_types.items():
            lines.append(f"| {edge_type} | {count} |")
    else:
        lines.append("| (none) | 0 |")
    lines.append("")

    # ─── Recent Activity ───────────────────────────────────────────────────────
    lines.extend(["## Recent Activity", ""])
    if log_file.exists():
        log_content = log_file.read_text(encoding="utf-8", errors="replace")
        recent = re.findall(r"^## \[.*?\].*", log_content, re.MULTILINE)
        lines.append("```")
        lines.extend(recent[-5:] if recent else ["(no activity)"])
        lines.append("```")
    else:
        lines.append("(no log.md found)")
    lines.append("")
    lines.append(f"*Generated by pipeline stats on {today}*")
    lines.append("")

    return "\n".join(lines)


def run_stats(cfg: Config) -> dict:
    """Generate and write dashboard. Returns summary dict."""
    content = generate_dashboard(cfg)
    dashboard_path = cfg.config_dir / "dashboard.md"
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(content, encoding="utf-8")

    summary = collect_stats(cfg)
    summary["dashboard_path"] = str(dashboard_path)
    return summary
