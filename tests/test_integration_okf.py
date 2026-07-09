"""Integration test: full OKF v0.1 pipeline produces compliant output.

This test exercises the entire OKF pipeline end-to-end using synthetic data
and mock LLM responses (no real LLM calls). It verifies that the pipeline
produces OKF v0.1 compliant output by:

1. Creating a temp vault with .env config
2. Creating synthetic source content
3. Running the OKF linter before ingest (empty bundle → 0 files)
4. Creating concept files via okf_renderer functions
5. Resolving links, generating indices, logs
6. Linting the complete bundle (assert 0 errors)
7. Verifying frontmatter, no wikilinks, indices, log
8. Generating visualization
9. Exporting and importing the bundle
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.bundle_io import export_bundle, import_bundle
from pipeline.migrate import extract_wikilinks
from pipeline.okf_indexgen import (
    generate_bundle_index,
    generate_directory_index,
    generate_log,
)
from pipeline.okf_lint import lint_bundle
from pipeline.okf_markdown import atomic_write, parse_frontmatter
from pipeline.okf_models import LogEntry
from pipeline.okf_renderer import (
    render_concept_page,
    render_moc_page,
    render_source_page,
)
from pipeline.okf_resolver import resolve_links
from pipeline.okf_visualizer import generate_visualization

# ── Synthetic data ──────────────────────────────────────────────────────

_TS = "2025-06-17T10:00:00Z"
_LOG_DATE = "2025-06-17"

# Three synthetic source files for 02-Clippings/
_SOURCES = [
    {
        "filename": "source1.md",
        "title": "Understanding Neural Networks",
        "url": "https://example.com/neural-networks",
        "content": (
            "Neural networks are computational models inspired by the human "
            "brain. They consist of layers of interconnected nodes that process "
            "information."
        ),
    },
    {
        "filename": "source2.md",
        "title": "Introduction to Transformers",
        "url": "https://example.com/transformers",
        "content": (
            "Transformers are a type of neural network architecture that relies "
            "entirely on attention mechanisms, dispensing with recurrence and "
            "convolutions."
        ),
    },
    {
        "filename": "source3.md",
        "title": "Knowledge Graphs Explained",
        "url": "https://example.com/knowledge-graphs",
        "content": (
            "A knowledge graph is a knowledge base that uses a graph-structured "
            "data model to represent entities and their relationships."
        ),
    },
]

# Concepts derived from sources (simulating LLM extraction)
_CONCEPTS = [
    {
        "filename": "neural-networks.md",
        "concept_id": "concepts/neural-networks",
        "title": "Neural Networks",
        "summary": "Computational models inspired by biological neural networks.",
        "body": (
            "Neural networks are composed of layers of artificial neurons. "
            "Each neuron applies a weighted sum of inputs followed by an "
            "activation function. Training occurs through backpropagation, "
            "adjusting weights to minimise a loss function."
        ),
        "tags": ["machine-learning", "neural-networks"],
        "source_ids": ["sources/source1"],
    },
    {
        "filename": "attention-mechanism.md",
        "concept_id": "concepts/attention-mechanism",
        "title": "Attention Mechanism",
        "summary": "A technique allowing models to focus on relevant input parts.",
        "body": (
            "The attention mechanism computes a weighted combination of values "
            "where weights are derived from query-key similarity. This enables "
            "the model to selectively focus on different parts of the input "
            "sequence when producing each output element."
        ),
        "tags": ["machine-learning", "transformers", "attention"],
        "source_ids": ["sources/source2"],
    },
    {
        "filename": "knowledge-graphs.md",
        "concept_id": "concepts/knowledge-graphs",
        "title": "Knowledge Graphs",
        "summary": "Graph-structured knowledge bases representing entities and relations.",
        "body": (
            "Knowledge graphs store information as triples (subject, predicate, "
            "object) and enable reasoning over relationships. They are widely "
            "used in search engines, recommendation systems, and question "
            "answering."
        ),
        "tags": ["knowledge-representation", "graphs"],
        "source_ids": ["sources/source3"],
    },
]

# MOC linking concepts together
_MOC = {
    "filename": "machine-learning-overview.md",
    "concept_id": "mocs/machine-learning-overview",
    "title": "Machine Learning Overview",
    "summary": "A curated index of key machine learning concepts.",
    "tags": ["machine-learning", "overview"],
    "concept_links": [
        ("concepts/neural-networks", "Neural Networks"),
        ("concepts/attention-mechanism", "Attention Mechanism"),
        ("concepts/knowledge-graphs", "Knowledge Graphs"),
    ],
}


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_env(vault: Path) -> Path:
    """Write a minimal .env config file into the vault root."""
    env_path = vault / ".env"
    env_path.write_text(
        f"VAULT_PATH={vault}\n"
        "LLM_PROVIDER=ollama\n"
        "LLM_HOST=http://localhost:11434\n"
        "LLM_MODEL=test-model\n"
        "OKF_VERSION=0.1\n",
        encoding="utf-8",
    )
    return env_path


def _write_clippings(vault: Path) -> list[Path]:
    """Create 2-3 synthetic source .md files in 02-Clippings/."""
    clippings = vault / "02-Clippings"
    clippings.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in _SOURCES:
        p = clippings / src["filename"]
        p.write_text(
            f"# {src['title']}\n\n{src['content']}\n",
            encoding="utf-8",
        )
        paths.append(p)
    return paths


# ── Main integration test ─────────────────────────────────────────────────


def test_full_okf_pipeline_integration(tmp_path: Path):
    """End-to-end integration test of the OKF v0.1 pipeline.

    Exercises renderer → resolver → indexgen → lint → visualizer →
    export → import, verifying OKF compliance at each stage.
    """
    # ── Step 1: Create a temp vault with .env config ───────────────────
    vault = tmp_path / "testvault"
    vault.mkdir(parents=True)
    env_path = _write_env(vault)
    assert env_path.exists(), ".env file was not created"

    bundle_dir = vault / "04-Wiki"

    # ── Step 2: Create synthetic source content in 02-Clippings/ ──────
    clippings_paths = _write_clippings(vault)
    assert len(clippings_paths) == 3
    for p in clippings_paths:
        assert p.exists()
        assert p.parent.name == "02-Clippings"
        assert p.read_text(encoding="utf-8").strip() != ""

    # ── Step 3: Lint empty bundle (should find 0 files) ────────────────
    bundle_dir.mkdir(parents=True)
    report = lint_bundle(bundle_dir)
    assert report.files_checked == 0, (
        f"Expected 0 files in empty bundle, got {report.files_checked}"
    )
    assert report.passed is True

    # ── Step 4: Create concept files using okf_renderer ────────────────
    sources_dir = bundle_dir / "sources"
    concepts_dir = bundle_dir / "concepts"
    mocs_dir = bundle_dir / "mocs"
    for d in (sources_dir, concepts_dir, mocs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 4a. Render source pages
    for src in _SOURCES:
        content = render_source_page(
            title=src["title"],
            url=src["url"],
            content=src["content"],
            timestamp=_TS,
        )
        atomic_write(sources_dir / src["filename"], content)

    # 4b. Render concept pages
    for concept in _CONCEPTS:
        content = render_concept_page(
            title=concept["title"],
            summary=concept["summary"],
            body=concept["body"],
            tags=concept["tags"],
            source_ids=concept["source_ids"],
            timestamp=_TS,
        )
        atomic_write(concepts_dir / concept["filename"], content)

    # 4c. Render MOC page
    moc_content = render_moc_page(
        title=_MOC["title"],
        summary=_MOC["summary"],
        concept_links=_MOC["concept_links"],
        tags=_MOC["tags"],
        timestamp=_TS,
    )
    atomic_write(mocs_dir / _MOC["filename"], moc_content)

    # ── Step 5: Run okf_resolver.resolve_links on the bundle ───────────
    # Links from renderers are already absolute (start with /),
    # so resolver should not modify any files.
    modified = resolve_links(bundle_dir)
    assert modified == 0, f"Resolver modified {modified} files unexpectedly"

    # ── Step 6: Generate directory index for each subdirectory ────────
    for directory in (sources_dir, concepts_dir, mocs_dir):
        concept_files = sorted(
            f for f in directory.glob("*.md")
            if f.name not in ("index.md", "log.md")
        )
        index_content = generate_directory_index(directory, concept_files)
        atomic_write(directory / "index.md", index_content)

    # ── Step 7: Generate bundle root index ──────────────────────────────
    root_index_content = generate_bundle_index(bundle_dir, okf_version="0.1")
    atomic_write(bundle_dir / "index.md", root_index_content)

    # ── Step 8: Generate log.md with LogEntry objects ───────────────────
    log_entries: list[LogEntry] = []
    for src in _SOURCES:
        log_entries.append(LogEntry(
            date=_LOG_DATE,
            action="ingested",
            concept_id=f"sources/{Path(src['filename']).stem}",
            description=f"Ingested source: {src['title']}",
            timestamp=_TS,
        ))
    for concept in _CONCEPTS:
        log_entries.append(LogEntry(
            date=_LOG_DATE,
            action="created",
            concept_id=concept["concept_id"],
            description=f"Created concept: {concept['title']}",
            timestamp=_TS,
        ))
    log_entries.append(LogEntry(
        date=_LOG_DATE,
        action="created",
        concept_id=_MOC["concept_id"],
        description=f"Created MOC: {_MOC['title']}",
        timestamp=_TS,
    ))

    log_content = generate_log(log_entries)
    atomic_write(bundle_dir / "log.md", log_content)

    # ── Step 9: Lint the complete bundle — assert 0 errors ──────────────
    report = lint_bundle(bundle_dir)
    error_issues = [i for i in report.issues if i.severity == "error"]
    assert report.errors == 0, (
        f"Lint found {report.errors} errors: "
        + "; ".join(
            f"[{i.rule}] {i.file}: {i.message}" for i in error_issues
        )
    )
    assert report.passed is True

    # ── Step 10: Verify each concept has non-empty type field ──────────
    all_md_files = sorted(bundle_dir.rglob("*.md"))
    for md_file in all_md_files:
        if md_file.name in ("index.md", "log.md"):
            continue
        raw = md_file.read_text(encoding="utf-8")
        meta, _body = parse_frontmatter(raw)
        assert "type" in meta, f"Missing 'type' key in frontmatter of {md_file}"
        assert meta["type"], f"Empty 'type' in frontmatter of {md_file}"
        assert isinstance(meta["type"], str), (
            f"'type' is not a string in {md_file}: {type(meta['type'])}"
        )
        assert meta["type"].strip() != "", (
            f"Blank 'type' in frontmatter of {md_file}"
        )

    # ── Step 11: Verify no [[wikilinks]] remain in any file ─────────────
    for md_file in all_md_files:
        raw = md_file.read_text(encoding="utf-8")
        wikilinks = extract_wikilinks(raw)
        assert wikilinks == [], (
            f"Found wikilinks in {md_file}: {wikilinks}"
        )

    # ── Step 12: Verify index.md exists in each directory ──────────────
    for directory in (sources_dir, concepts_dir, mocs_dir):
        idx = directory / "index.md"
        assert idx.exists(), f"Missing index.md in {directory}"
    root_idx = bundle_dir / "index.md"
    assert root_idx.exists(), "Missing root index.md"

    # ── Step 13: Verify log.md exists at bundle root ───────────────────
    log_md = bundle_dir / "log.md"
    assert log_md.exists(), "Missing log.md at bundle root"

    # ── Step 14: Generate visualization ────────────────────────────────
    viz_path = generate_visualization(bundle_dir)
    assert viz_path.exists(), "viz.html was not created"
    assert viz_path.name == "viz.html"
    viz_html = viz_path.read_text(encoding="utf-8")
    assert "cytoscape" in viz_html.lower(), (
        "viz.html does not reference cytoscape"
    )

    # ── Step 15: Export bundle — assert .tar.gz created ────────────────
    tarball = export_bundle(
        bundle_dir, output_path=tmp_path / "okf-bundle.tar.gz"
    )
    assert tarball.exists(), "Tarball was not created"
    assert tarball.suffix == ".gz"
    assert tarball.name == "okf-bundle.tar.gz"

    # ── Step 16: Import bundle — assert conformance check passes ───────
    import_target = tmp_path / "imported"
    result = import_bundle(tarball, import_target, verify=True)
    assert "bundle_path" in result
    assert "lint_report" in result

    lint_report = result["lint_report"]
    assert lint_report["passed"] is True, (
        f"Imported bundle failed lint: {lint_report}"
    )
    assert lint_report["errors"] == 0, (
        f"Imported bundle has lint errors: {lint_report}"
    )

    # Verify imported bundle has the same structure
    imported_bundle = Path(result["bundle_path"])
    assert imported_bundle.is_dir()
    assert (imported_bundle / "index.md").exists()
    assert (imported_bundle / "log.md").exists()
    assert (imported_bundle / "concepts" / "neural-networks.md").exists()
    assert (imported_bundle / "sources" / "source1.md").exists()
    assert (imported_bundle / "mocs" / "machine-learning-overview.md").exists()


# ── pytest entry ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
