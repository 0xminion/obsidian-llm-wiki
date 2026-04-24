"""Regression tests for template-mode and review workflow bugs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.cli import check_dependencies
from pipeline.config import Config
from pipeline.create.templates import create_file_templates
from pipeline.create.validate import validate_single_file
from pipeline.models import Language, Plan, Template
from pipeline.review import approve_reviews, stage_for_review
from pipeline.store import ContentStore


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    vault = tmp_path / "vault"
    for rel in [
        "01-Raw",
        "04-Wiki/sources",
        "04-Wiki/entries",
        "04-Wiki/concepts",
        "04-Wiki/mocs",
        "06-Config",
        "08-Archive-Raw",
        "Meta/Scripts",
        "Meta/prompts",
        "Meta/Templates",
    ]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    extract_dir = vault / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    return Config(vault_path=vault, extract_dir=extract_dir)


def _write_extract(cfg: Config, plan: Plan, *, url: str | None = None, title: str | None = None, content: str | None = None) -> None:
    payload = {
        "url": url or f"https://example.com/{plan.hash}",
        "title": title or plan.title,
        "type": "web",
        "author": "Author",
        "content": content or ("Useful extracted content. " * 120),
    }
    (cfg.resolved_extract_dir / f"{plan.hash}.json").write_text(json.dumps(payload), encoding="utf-8")


def _url_file_hash(url: str) -> str:
    return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]


@pytest.mark.parametrize(
    ("plan", "note_name"),
    [
        (
            Plan(
                hash="zh123abc456",
                title="测试文章",
                language=Language.ZH,
                template=Template.CHINESE,
                tags=["ai", "research"],
                concept_new=["新概念"],
            ),
            "chinese",
        ),
        (
            Plan(
                hash="tech123abc45",
                title="Technical Deep Dive",
                template=Template.TECHNICAL,
                tags=["systems", "testing"],
                concept_new=["Concurrency"],
            ),
            "technical",
        ),
    ],
)
def test_template_generated_notes_validate_for_nonstandard_templates(cfg: Config, plan: Plan, note_name: str):
    _write_extract(cfg, plan)

    stats = create_file_templates([plan], cfg, use_agent_insights=False)

    assert stats["failed"] == 0, note_name
    entry_path = next(cfg.entries_dir.glob("*.md"))
    source_path = next(cfg.sources_dir.glob("*.md"))
    entry_text = entry_path.read_text(encoding="utf-8")
    assert validate_single_file(entry_path, "entry") == []
    assert validate_single_file(source_path, "source") == []
    assert 'source_url: "https://example.com/' in entry_text
    assert 'author: "Author"' in entry_text
    assert "type: web" in entry_text


def test_template_entry_has_single_frontmatter_block(cfg: Config):
    plan = Plan(hash="frontmatter01", title="Frontmatter Test", tags=["ai"])
    _write_extract(cfg, plan)

    create_file_templates([plan], cfg, use_agent_insights=False)

    entry_path = next(cfg.entries_dir.glob("*.md"))
    text = entry_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert text.count("\n---\n") == 1


def test_template_entries_validate_when_tags_are_empty(cfg: Config):
    plan = Plan(hash="tagless01", title="Tagless Entry", tags=[])
    _write_extract(cfg, plan)

    stats = create_file_templates([plan], cfg, use_agent_insights=False)

    assert stats["failed"] == 0
    entry_path = next(cfg.entries_dir.glob("*.md"))
    assert validate_single_file(entry_path, "entry") == []


def test_template_creation_runs_post_processing(cfg: Config):
    url = "https://example.com/article"
    plan = Plan(hash=_url_file_hash(url), title="Template Pipeline", tags=["ai"])
    _write_extract(cfg, plan, url=url)
    (cfg.inbox_dir / "article.url").write_text(f"[InternetShortcut]\nURL={url}\n", encoding="utf-8")

    create_file_templates([plan], cfg, use_agent_insights=False)

    assert cfg.wiki_index.exists()
    assert (cfg.config_dir / "tag-registry.md").exists()
    assert cfg.log_md.exists()
    assert not (cfg.inbox_dir / "article.url").exists()
    assert (cfg.archive_dir / "article.url").exists()


def test_template_post_processing_does_not_touch_unrelated_older_notes(cfg: Config):
    stale_entry = cfg.entries_dir / "legacy.md"
    original = """---
title: \"Legacy\"
source: \"[[legacy]]\"
date_entry: 2026-04-01
status: draft
template: standard
tags:
  - legacy
---

# Legacy

This is a sufficiently long legacy paragraph that should be treated as real content during repair instead of being ignored by the validator.

- First useful bullet point that is definitely longer than twenty five characters.

What does this imply for the system?

## Linked concepts

- [[Legacy Concept]]
"""
    stale_entry.write_text(original, encoding="utf-8")
    old_time = 1_700_000_000
    os.utime(stale_entry, (old_time, old_time))

    plan = Plan(hash="postprocess01", title="Fresh Entry", tags=["ai"])
    _write_extract(cfg, plan)

    create_file_templates([plan], cfg, use_agent_insights=False)

    assert stale_entry.read_text(encoding="utf-8") == original


def test_template_post_processing_keeps_inbox_when_violations_remain(cfg: Config):
    url = "https://example.com/broken-template"
    plan = Plan(hash=_url_file_hash(url), title="Broken Template", tags=["ai"])
    _write_extract(cfg, plan, url=url, content="tiny")
    (cfg.inbox_dir / "broken.url").write_text(f"URL={url}\n", encoding="utf-8")

    create_file_templates([plan], cfg, use_agent_insights=False)

    assert (cfg.inbox_dir / "broken.url").exists()
    assert not (cfg.archive_dir / "broken.url").exists()


def test_template_source_creation_uses_collision_resolution(cfg: Config):
    plan_a = Plan(hash="same-title-a", title="Same Title", tags=["alpha"], moc_targets=["General"], concept_new=["Concept A"])
    plan_b = Plan(hash="same-title-b", title="Same Title", tags=["beta"], moc_targets=["General"], concept_new=["Concept B"])
    _write_extract(cfg, plan_a, url="https://example.com/a")
    _write_extract(cfg, plan_b, url="https://example.com/b")

    stats = create_file_templates([plan_a, plan_b], cfg, use_agent_insights=False)

    assert stats["sources"] == 2
    source_files = sorted(cfg.sources_dir.glob("*.md"))
    assert len(source_files) == 2
    contents = [p.read_text(encoding="utf-8") for p in source_files]
    assert any("https://example.com/a" in text for text in contents)
    assert any("https://example.com/b" in text for text in contents)

    entry_files = sorted(cfg.entries_dir.glob("*.md"))
    assert len(entry_files) == 2
    entry_texts = [p.read_text(encoding="utf-8") for p in entry_files]
    assert any('title: "Same Title"' in text and "# Same Title\n" in text for text in entry_texts)
    assert any('title: "Same Title-1"' in text and "# Same Title-1\n" in text for text in entry_texts)
    assert any('source: "[[same-title]]"' in text for text in entry_texts)
    assert any('source: "[[same-title-1]]"' in text for text in entry_texts)

    moc_text = (cfg.mocs_dir / "general.md").read_text(encoding="utf-8")
    assert "[[Same Title]]: Related to [[same-title]]" in moc_text
    assert "[[Same Title-1]]: Related to [[same-title-1]]" in moc_text

    concept_a = (cfg.concepts_dir / "concept-a.md").read_text(encoding="utf-8")
    concept_b = (cfg.concepts_dir / "concept-b.md").read_text(encoding="utf-8")
    assert '[[same-title]]' in concept_a
    assert '"Same Title"' in concept_a
    assert '[[same-title-1]]' in concept_b
    assert '"Same Title-1"' in concept_b


def test_template_mode_uses_shared_suffix_when_only_sources_collide(cfg: Config):
    (cfg.sources_dir / "same-title.md").write_text("existing", encoding="utf-8")
    plan = Plan(hash="shared-suffix-1", title="Same Title", tags=["alpha"], concept_new=["Shared Concept"])
    _write_extract(cfg, plan, url="https://example.com/shared")

    create_file_templates([plan], cfg, use_agent_insights=False)

    entry_text = (cfg.entries_dir / "same-title-1.md").read_text(encoding="utf-8")
    source_text = (cfg.sources_dir / "same-title-1.md").read_text(encoding="utf-8")
    concept_text = (cfg.concepts_dir / "shared-concept.md").read_text(encoding="utf-8")
    assert 'source: "[[same-title-1]]"' in entry_text
    assert 'title: "Same Title-1"' in entry_text
    assert '# Same Title-1\n' in entry_text
    assert 'title: "Same Title-1"' in source_text
    assert '[[same-title-1]]' in concept_text
    assert '"Same Title-1"' in concept_text


def test_approve_reviews_only_archives_fully_successful_plan_hashes(cfg: Config):
    url = "https://example.com/review-case"
    plan_hash = _url_file_hash(url)
    (cfg.inbox_dir / "review.url").write_text(f"URL={url}\n", encoding="utf-8")

    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        plan_data = {
            "hash": plan_hash,
            "title": "Review Plan",
            "language": "en",
            "template": "standard",
            "tags": ["ai"],
            "concept_updates": [],
            "concept_new": [],
            "moc_targets": [],
        }
        store.review_add(plan_hash, plan_data, "entry", str(cfg.entries_dir / "review-plan.md"), "ok")
        store.review_add(plan_hash, plan_data, "source", "/proc/1/forbidden.md", "bad")
    finally:
        store.close()

    stats = approve_reviews(cfg)

    assert stats["failed"] == 1
    assert (cfg.inbox_dir / "review.url").exists()
    assert not (cfg.archive_dir / "review.url").exists()


def test_approve_reviews_with_partial_review_ids_does_not_archive_plan(cfg: Config):
    url = "https://example.com/review-partial"
    plan_hash = _url_file_hash(url)
    (cfg.inbox_dir / "partial.url").write_text(f"URL={url}\n", encoding="utf-8")

    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        plan_data = {
            "hash": plan_hash,
            "title": "Partial Review Plan",
            "language": "en",
            "template": "standard",
            "tags": ["ai"],
            "concept_updates": [],
            "concept_new": [],
            "moc_targets": [],
        }
        first_id = store.review_add(plan_hash, plan_data, "entry", str(cfg.entries_dir / "partial-entry.md"), "ok")
        store.review_add(plan_hash, plan_data, "source", str(cfg.sources_dir / "partial-source.md"), "ok")
    finally:
        store.close()

    stats = approve_reviews(cfg, [first_id])

    assert stats["written"] == 1
    assert (cfg.inbox_dir / "partial.url").exists()
    assert not (cfg.archive_dir / "partial.url").exists()


def test_review_mode_duplicate_titles_get_distinct_paths(cfg: Config):
    plan_a = Plan(hash="review-same-a", title="Same Title", tags=["alpha"], concept_new=["Review Concept A"])
    plan_b = Plan(hash="review-same-b", title="Same Title", tags=["beta"], concept_new=["Review Concept B"])
    _write_extract(cfg, plan_a, url="https://example.com/review-a")
    _write_extract(cfg, plan_b, url="https://example.com/review-b")

    stats = stage_for_review(type("PlansObj", (), {"plans": [plan_a, plan_b]})(), cfg, use_agent_insights=False)

    assert stats == {"staged": 2, "failed": 0}
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        pending = store.review_get_pending()
    finally:
        store.close()

    entry_paths = sorted(r["file_path"] for r in pending if r["file_type"] == "entry")
    source_paths = sorted(r["file_path"] for r in pending if r["file_type"] == "source")
    concept_payloads = [r["file_content"] for r in pending if r["file_type"] == "concept"]
    assert len(set(entry_paths)) == 2
    assert len(set(source_paths)) == 2
    assert any('[[same-title]]' in text and '"Same Title"' in text for text in concept_payloads)
    assert any('[[same-title-1]]' in text and '"Same Title-1"' in text for text in concept_payloads)

    approval = approve_reviews(cfg)
    assert approval["written"] == 6
    assert len(list(cfg.entries_dir.glob("*.md"))) == 2
    assert len(list(cfg.sources_dir.glob("*.md"))) == 2


def test_check_dependencies_honors_configured_agent_command():
    available = {"curl", "jq", "python3", "claude"}

    def fake_which(cmd: str) -> str | None:
        return f"/usr/bin/{cmd}" if cmd in available else None

    with patch("pipeline.cli.shutil.which", side_effect=fake_which):
        assert check_dependencies(agent_cmd="claude") == []
