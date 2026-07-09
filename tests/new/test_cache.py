"""Tests for obsidian_llm_wiki.core.cache — synthesis cache round-trip."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.cache import (
    delete_cached_synthesis,
    load_all_cached_syntheses,
    load_synthesis,
    save_synthesis,
)
from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
)


def _make_synth() -> SourceSynthesis:
    """Build a minimal SourceSynthesis for testing."""
    return SourceSynthesis(
        source_title="Test Article",
        source_summary="A test summary.",
        source_tags=["test", "wiki"],
        key_points=["Point one", "Point two"],
        open_questions=["Why?"],
        language="en",
        concepts=[
            ConceptNote(
                title="Test Concept",
                slug="test-concept",
                summary="Concept summary.",
                tags=["test"],
                aliases=["tc"],
                sections=[BodySection(heading="Core", points=["p1", "p2"])],
                confidence=0.9,
                provenance="extracted",
                is_new=True,
            ),
        ],
        maps=[
            MapOfContent(
                title="Test MOC",
                slug="test-moc",
                summary="MOC summary.",
                tags=["test"],
                concept_slugs=["test-concept"],
            ),
        ],
    )


def test_save_and_load_round_trip(tmp_path: Path):
    """save_synthesis → load_synthesis preserves all fields."""
    synth = _make_synth()
    save_synthesis(synth, tmp_path, "article.md")

    loaded = load_synthesis(tmp_path, "article.md")
    assert loaded is not None
    assert loaded.source_title == "Test Article"
    assert loaded.source_summary == "A test summary."
    assert loaded.source_tags == ["test", "wiki"]
    assert loaded.key_points == ["Point one", "Point two"]
    assert loaded.language == "en"
    assert loaded.source_file == "article.md"
    assert len(loaded.concepts) == 1
    c = loaded.concepts[0]
    assert c.slug == "test-concept"
    assert c.title == "Test Concept"
    assert c.tags == ["test"]
    assert c.aliases == ["tc"]
    assert len(c.sections) == 1
    assert c.sections[0].heading == "Core"
    assert c.sections[0].points == ["p1", "p2"]
    assert c.confidence == 0.9
    assert len(loaded.maps) == 1
    assert loaded.maps[0].slug == "test-moc"
    assert loaded.maps[0].concept_slugs == ["test-concept"]


def test_load_synthesis_missing_file(tmp_path: Path):
    """load_synthesis returns None for missing files."""
    assert load_synthesis(tmp_path, "nonexistent.md") is None


def test_load_synthesis_corrupt_file(tmp_path: Path):
    """load_synthesis returns None for corrupt JSON."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "corrupt.md.json").write_text("not valid json {{{")
    assert load_synthesis(tmp_path, "corrupt.md") is None


def test_load_all_cached_syntheses(tmp_path: Path):
    """load_all_cached_syntheses loads multiple cache files."""
    save_synthesis(_make_synth(), tmp_path, "a.md")
    save_synthesis(_make_synth(), tmp_path, "b.md")

    all_cached = load_all_cached_syntheses(tmp_path)
    assert len(all_cached) == 2
    assert "a.md" in all_cached
    assert "b.md" in all_cached
    assert all_cached["a.md"].source_title == "Test Article"


def test_load_all_cached_syntheses_empty_dir(tmp_path: Path):
    """load_all_cached_syntheses returns empty dict when no cache exists."""
    assert load_all_cached_syntheses(tmp_path) == {}


def test_delete_cached_synthesis(tmp_path: Path):
    """delete_cached_synthesis removes the cache file."""
    save_synthesis(_make_synth(), tmp_path, "doomed.md")
    assert load_synthesis(tmp_path, "doomed.md") is not None

    delete_cached_synthesis(tmp_path, "doomed.md")
    assert load_synthesis(tmp_path, "doomed.md") is None


def test_delete_cached_synthesis_missing_file(tmp_path: Path):
    """delete_cached_synthesis on missing file is a no-op."""
    delete_cached_synthesis(tmp_path, "never-existed.md")
