"""Tests for pipeline.okf_visualizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import okf_visualizer as viz

# ── Helpers ──────────────────────────────────────────────────────────────


def _write_concept(
    bundle: Path, name: str, *, title: str | None = None,
    ctype: str = "Concept", body: str = "", tags: list[str] | None = None,
) -> Path:
    """Write a minimal OKF concept markdown file into ``bundle``."""
    p = bundle / f"{name}.md"
    fm_lines = ["---", f"type: {ctype}"]
    if title:
        fm_lines.append(f"title: {title}")
    if tags:
        fm_lines.append("tags:")
        for t in tags:
            fm_lines.append(f"  - {t}")
    fm_lines.append("---")
    content = "\n".join(fm_lines) + "\n\n" + body + "\n"
    p.write_text(content, encoding="utf-8")
    return p


def _make_minimal_bundle(tmp_path: Path) -> Path:
    """Create a minimal OKF bundle with two concepts + a link between them."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_concept(
        bundle, "alpha",
        title="Alpha Concept",
        ctype="Concept",
        body="This is the alpha concept. See [Beta](beta.md) for more.",
        tags=["core"],
    )
    _write_concept(
        bundle, "beta",
        title="Beta Concept",
        ctype="Concept",
        body="Beta links back to [Alpha](alpha.md).",
        tags=["related"],
    )
    # index.md should be skipped by the visualizer.
    (bundle / "index.md").write_text("---\ntype: MOC\n---\n# Index\n", encoding="utf-8")
    return bundle


# ── generate_visualization ────────────────────────────────────────────────


def test_generate_visualization_creates_html_file(tmp_path: Path):
    """generate_visualization writes an HTML file to the default path."""
    bundle = _make_minimal_bundle(tmp_path)
    out = viz.generate_visualization(bundle)
    assert out.exists()
    assert out.suffix == ".html"
    assert out.name == "viz.html"


def test_generate_visualization_custom_output_path(tmp_path: Path):
    """generate_visualization honours output_path."""
    bundle = _make_minimal_bundle(tmp_path)
    custom = tmp_path / "custom" / "graph.html"
    out = viz.generate_visualization(bundle, output_path=custom)
    assert out == custom
    assert custom.exists()


def test_generate_visualization_custom_name_in_title(tmp_path: Path):
    """The ``name`` appears in the HTML <title>."""
    bundle = _make_minimal_bundle(tmp_path)
    out = viz.generate_visualization(bundle, name="MyWiki")
    html = out.read_text(encoding="utf-8")
    assert "MyWiki" in html


def test_generate_visualization_missing_bundle(tmp_path: Path):
    """A missing bundle dir raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        viz.generate_visualization(tmp_path / "does-not-exist")


# ── HTML content checks ──────────────────────────────────────────────────


def test_html_contains_cytoscape_reference(tmp_path: Path):
    """The HTML references the Cytoscape.js CDN."""
    bundle = _make_minimal_bundle(tmp_path)
    out = viz.generate_visualization(bundle)
    html = out.read_text(encoding="utf-8")
    assert "cytoscape" in html.lower()
    # The CDN URL should be present.
    assert (
        "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.4/cytoscape.min.js"
        in html
    )


def test_html_contains_embedded_concept_data(tmp_path: Path):
    """The HTML embeds concept data as JSON."""
    bundle = _make_minimal_bundle(tmp_path)
    out = viz.generate_visualization(bundle)
    html = out.read_text(encoding="utf-8")

    # The concept title should appear in the embedded JSON.
    assert "Alpha Concept" in html
    assert "Beta Concept" in html
    # Dark theme background colour.
    assert "#1a1a2e" in html
    # The layout selector options.
    assert "cose" in html
    assert "concentric" in html
    assert "breadthfirst" in html
    assert "circle" in html
    assert "grid" in html


def test_html_is_valid_structure(tmp_path: Path):
    """The generated HTML has DOCTYPE, html, head, body tags."""
    bundle = _make_minimal_bundle(tmp_path)
    out = viz.generate_visualization(bundle)
    html = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html
    assert "<head>" in html
    assert "<body>" in html


# ── _collect_concepts ─────────────────────────────────────────────────────


def test_collect_concepts_skips_index_and_log(tmp_path: Path):
    """_collect_concepts skips index.md, log.md, viz.html."""
    bundle = _make_minimal_bundle(tmp_path)
    (bundle / "log.md").write_text("---\ntype: Source\n---\n# Log\n", encoding="utf-8")
    (bundle / "viz.html").write_text("<html></html>", encoding="utf-8")

    concepts = viz._collect_concepts(bundle)
    ids = [c["id"] for c in concepts]
    assert "alpha" in ids
    assert "beta" in ids
    # index / log / viz should be excluded.
    assert "index" not in ids
    assert "log" not in ids


def test_collect_concepts_returns_expected_fields(tmp_path: Path):
    """Each concept dict has all expected keys."""
    bundle = _make_minimal_bundle(tmp_path)
    concepts = viz._collect_concepts(bundle)
    assert len(concepts) == 2
    for c in concepts:
        assert "id" in c
        assert "file" in c
        assert "title" in c
        assert "type" in c
        assert "description" in c
        assert "tags" in c
        assert "timestamp" in c
        assert "resource" in c
        assert "body" in c
        assert "links" in c


# ── _build_graph_data ──────────────────────────────────────────────────────


def test_build_graph_data_creates_nodes_and_edges(tmp_path: Path):
    """_build_graph_data returns nodes, edges, and backlinks."""
    bundle = _make_minimal_bundle(tmp_path)
    concepts = viz._collect_concepts(bundle)
    nodes, edges, backlinks = viz._build_graph_data(concepts, bundle)

    node_ids = {n["id"] for n in nodes}
    assert "alpha" in node_ids
    assert "beta" in node_ids

    # alpha links to beta, and beta links to alpha.
    assert len(edges) >= 2
    edge_pairs = {(e["source"], e["target"]) for e in edges}
    assert ("alpha", "beta") in edge_pairs
    assert ("beta", "alpha") in edge_pairs

    # Backlinks: alpha is cited by beta, beta is cited by alpha.
    assert "beta" in backlinks.get("alpha", [])
    assert "alpha" in backlinks.get("beta", [])


def test_node_data_has_required_keys(tmp_path: Path):
    """Nodes contain id, label, type, description, tags, timestamp, resource."""
    bundle = _make_minimal_bundle(tmp_path)
    concepts = viz._collect_concepts(bundle)
    nodes, _edges, _bl = viz._build_graph_data(concepts, bundle)
    for n in nodes:
        assert "id" in n
        assert "label" in n
        assert "type" in n
        assert "description" in n
        assert "tags" in n
        assert "timestamp" in n
        assert "resource" in n


# ── _resolve_target ────────────────────────────────────────────────────────


def test_resolve_target_absolute_link():
    """An absolute internal link /foo.md resolves to 'foo'."""
    result = viz._resolve_target("/foo.md", "concepts/bar", Path("/tmp"))
    assert result == "foo"


def test_resolve_target_relative_link():
    """A relative link foo.md resolves to 'foo' (same directory)."""
    result = viz._resolve_target("foo.md", "bar", Path("/tmp"))
    assert "foo" in result


def test_resolve_target_external_link():
    """External links resolve to empty string."""
    result = viz._resolve_target("https://example.com/page", "bar", Path("/tmp"))
    assert result == ""


def test_resolve_target_mailto():
    """mailto: links resolve to empty string."""
    result = viz._resolve_target("mailto:foo@bar.com", "bar", Path("/tmp"))
    assert result == ""


def test_resolve_target_empty():
    """Empty target resolves to empty string."""
    assert viz._resolve_target("", "bar", Path("/tmp")) == ""


# ── _render_html ───────────────────────────────────────────────────────────


def test_render_html_contains_all_json_data(tmp_path: Path):
    """_render_html embeds nodes, edges, backlinks, and concepts JSON."""
    bundle = _make_minimal_bundle(tmp_path)
    concepts = viz._collect_concepts(bundle)
    nodes, edges, backlinks = viz._build_graph_data(concepts, bundle)
    html = viz._render_html("TestWiki", nodes, edges, backlinks, concepts)

    # The JSON-serialised data should be parseable back from the HTML.
    # Find the script block and verify the data is present.
    assert "Alpha Concept" in html
    assert "TestWiki" in html


def test_render_html_no_format_placeholders_left(tmp_path: Path):
    """No un-substituted {placeholder} should remain after .format()."""
    bundle = _make_minimal_bundle(tmp_path)
    concepts = viz._collect_concepts(bundle)
    nodes, edges, backlinks = viz._build_graph_data(concepts, bundle)
    html = viz._render_html("TestWiki", nodes, edges, backlinks, concepts)
    # Un-substituted format fields look like {name} — but {{ }} is valid JS.
    # We check that no single-brace placeholders like {cytoscape_cdn} remain.
    import re
    singles = re.findall(r"(?<!\{)\{[a-z_]+\}(?!\})", html)
    assert singles == [], f"Un-substituted format placeholders: {singles}"


# ── pytest entry ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
