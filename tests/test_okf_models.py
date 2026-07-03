"""Tests for OKF v0.1 frontmatter models and bundle conformance."""

import pytest

from pipeline.okf_models import (
    OKFBundle,
    OKFConcept,
    OKFConceptType,
    OKFFrontmatter,
)

# ── OKFFrontmatter ──────────────────────────────────────────────────────


class TestOKFFrontmatter:
    """Conformance, serialisation, and roundtrip behaviour."""

    def test_minimal_conformant_frontmatter(self):
        """A frontmatter with only type set is conformant."""
        fm = OKFFrontmatter(type="Concept")
        assert fm.is_conformant() is True

    def test_empty_type_not_conformant(self):
        """Empty or whitespace-only type fails conformance."""
        assert OKFFrontmatter(type="").is_conformant() is False
        assert OKFFrontmatter(type="   ").is_conformant() is False

    def test_none_type_not_conformant(self):
        """A None type (should not happen via constructor but defensive) fails."""
        fm = OKFFrontmatter(type=None)  # type: ignore[arg-type]
        assert fm.is_conformant() is False

    def test_full_frontmatter_to_dict(self):
        """All core fields appear in to_dict output."""
        fm = OKFFrontmatter(
            type="Entry",
            title="My Entry",
            description="A description",
            resource="https://example.com",
            tags=["a", "b"],
            timestamp="2025-01-01T00:00:00Z",
        )
        d = fm.to_dict()
        assert d["type"] == "Entry"
        assert d["title"] == "My Entry"
        assert d["description"] == "A description"
        assert d["resource"] == "https://example.com"
        assert d["tags"] == ["a", "b"]
        assert d["timestamp"] == "2025-01-01T00:00:00Z"

    def test_extensions_preserved_in_to_dict(self):
        """Extension keys beyond the core set survive to_dict."""
        fm = OKFFrontmatter(
            type="Concept",
            extensions={"custom_field": 42, "okf:version": "0.1"},
        )
        d = fm.to_dict()
        assert d["custom_field"] == 42
        assert d["okf:version"] == "0.1"
        # Core keys still present
        assert d["type"] == "Concept"

    def test_extensions_override_core_on_collision(self):
        """Extension keys take precedence over core keys on collision."""
        fm = OKFFrontmatter(type="Concept", extensions={"title": "Override"})
        d = fm.to_dict()
        assert d["title"] == "Override"

    def test_roundtrip_from_dict(self):
        """from_dict(to_dict()) preserves all fields including extensions."""
        original = OKFFrontmatter(
            type="Reference",
            title="RT",
            description="desc",
            resource="res",
            tags=["x", "y"],
            timestamp="2025-06-17",
            extensions={"extra": "val"},
        )
        d = original.to_dict()
        restored = OKFFrontmatter.from_dict(d)
        assert restored.type == original.type
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.resource == original.resource
        assert restored.tags == original.tags
        assert restored.timestamp == original.timestamp
        assert restored.extensions == original.extensions

    def test_from_dict_strips_unknown_keys_into_extensions(self):
        """Unknown keys in from_dict become extensions, not lost."""
        data = {"type": "Concept", "title": "T", "custom": "keep"}
        fm = OKFFrontmatter.from_dict(data)
        assert fm.type == "Concept"
        assert fm.title == "T"
        assert fm.extensions == {"custom": "keep"}

    def test_from_dict_none_tags_become_empty_list(self):
        """from_dict handles None tags gracefully."""
        data = {"type": "Concept", "tags": None}
        fm = OKFFrontmatter.from_dict(data)
        assert fm.tags == []


# ── OKFConceptType enum ─────────────────────────────────────────────────


class TestOKFConceptType:
    """Enum values match OKF v0.1 spec strings."""

    def test_enum_values(self):
        assert OKFConceptType.SOURCE == "Source"
        assert OKFConceptType.ENTRY == "Entry"
        assert OKFConceptType.CONCEPT == "Concept"
        assert OKFConceptType.MOC == "Map of Content"
        assert OKFConceptType.REFERENCE == "Reference"


# ── OKFConcept ─────────────────────────────────────────────────────────


class TestOKFConcept:
    """Concept file_path property and structure."""

    def test_file_path_property(self):
        fm = OKFFrontmatter(type="Concept", title="Test")
        c = OKFConcept(frontmatter=fm, body="Body text", concept_id="test-concept")
        assert c.file_path == "test-concept.md"


# ── OKFBundle ───────────────────────────────────────────────────────────


class TestOKFBundle:
    """Bundle conformance filtering."""

    def test_bundle_conformance_filter(self):
        """conformant_concepts / non_conformant partition correctly."""
        good_fm = OKFFrontmatter(type="Concept", title="Good")
        bad_fm = OKFFrontmatter(type="", title="Bad")

        bundle = OKFBundle(
            root="/vault/okf",
            concepts=[
                OKFConcept(frontmatter=good_fm, body="b1", concept_id="c1"),
                OKFConcept(frontmatter=bad_fm, body="b2", concept_id="c2"),
                OKFConcept(frontmatter=good_fm, body="b3", concept_id="c3"),
            ],
        )

        conformant = bundle.conformant_concepts()
        non_conformant = bundle.non_conformant()

        assert len(conformant) == 2
        assert len(non_conformant) == 1
        assert conformant[0].concept_id == "c1"
        assert conformant[1].concept_id == "c3"
        assert non_conformant[0].concept_id == "c2"

    def test_empty_bundle(self):
        """An empty bundle has zero conformant and non-conformant concepts."""
        bundle = OKFBundle(root="/empty")
        assert bundle.conformant_concepts() == []
        assert bundle.non_conformant() == []

    def test_all_conformant_bundle(self):
        """A fully-conformant bundle has no non-conformant concepts."""
        bundle = OKFBundle(
            root="/ok",
            concepts=[
                OKFConcept(
                    frontmatter=OKFFrontmatter(type="Entry"),
                    body="b",
                    concept_id="e1",
                ),
            ],
        )
        assert len(bundle.conformant_concepts()) == 1
        assert bundle.non_conformant() == []


# ── SourceStatus enum ──────────────────────────────────────────────────


class TestSourceStatus:
    """SourceStatus enum values."""

    def test_enum_values(self):
        from pipeline.okf_models import SourceStatus

        assert SourceStatus.NEW == "new"
        assert SourceStatus.CHANGED == "changed"
        assert SourceStatus.UNCHANGED == "unchanged"
        assert SourceStatus.DELETED == "deleted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
