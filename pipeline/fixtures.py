"""Deterministic fixture vault generation for examples and regression tests."""

from __future__ import annotations

from pathlib import Path


FIXTURE_DATE = "2026-01-01"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_example_vault(vault_path: Path, *, overwrite: bool = False) -> dict[str, int | str]:
    """Create a small deterministic vault with one source, entry, concept, and MoC.

    The fixture is intentionally boring: stable filenames, stable dates, stable
    content. That makes it useful for golden/snapshot tests and demos without
    accidentally turning examples into another moving target.
    """
    vault_path = Path(vault_path)
    files = {
        "01-Raw/example.url": "https://example.com/research-builder-signals\n",
        "04-Wiki/sources/research-builder-signals-source.md": f"""---
type: source
title: Research Builder Signals
source_url: https://example.com/research-builder-signals
date_extracted: {FIXTURE_DATE}
---

# Research Builder Signals

Original source material for the deterministic example vault.
""",
        "04-Wiki/entries/research-builder-signals.md": f"""---
type: entry
title: Research Builder Signals
date_entry: {FIXTURE_DATE}
reviewed: {FIXTURE_DATE}
tags:
  - research
  - builders
source: [[research-builder-signals-source]]
---

# Research Builder Signals

Builder-facing research signals become more useful when they link source evidence,
concept notes, and maps of content.

## Key Takeaways

- Source notes are first-class graph nodes.
- [[builder-signal]] captures reusable interpretation.
- This entry belongs in [[research-signals]].
""",
        "04-Wiki/concepts/builder-signal.md": f"""---
type: concept
title: Builder Signal
created: {FIXTURE_DATE}
tags:
  - research
---

# Builder Signal

A builder signal is a reusable observation that helps decide what to build next.

## Evidence

- Introduced by [[research-builder-signals]].
""",
        "04-Wiki/mocs/research-signals.md": f"""---
type: moc
title: Research Signals
created: {FIXTURE_DATE}
---

# Research Signals

- [[research-builder-signals]]
- [[builder-signal]]
""",
        "06-Config/edges.tsv": "source\ttarget\ttype\tdescription\nresearch-builder-signals\tbuilder-signal\trelates_to\tEntry discusses concept\nresearch-builder-signals\tresearch-signals\tpart_of\tEntry belongs to MoC\nresearch-builder-signals\tresearch-builder-signals-source\trelates_to\tEntry cites source note\n",
        "06-Config/wiki-index.md": "# Wiki Index\n\n- [[research-builder-signals]]: deterministic example entry (entry)\n- [[builder-signal]]: deterministic example concept (concept)\n- [[research-signals]]: deterministic example MoC (moc)\n- [[research-builder-signals-source]]: deterministic example source (source)\n",
        "06-Config/url-index.tsv": "url\thash\ttitle\nhttps://example.com/research-builder-signals\tfixture001\tResearch Builder Signals\n",
        "06-Config/log.md": f"# Wiki Activity Log\n\n## [{FIXTURE_DATE}] fixture\n\nCreated deterministic example vault.\n",
        "Meta/Scripts/.gitkeep": "",
    }

    written = 0
    for rel, content in sorted(files.items()):
        path = vault_path / rel
        if path.exists() and not overwrite:
            continue
        _write(path, content)
        written += 1

    return {
        "files_written": written,
        "sources": 1,
        "entries": 1,
        "concepts": 1,
        "mocs": 1,
    }
