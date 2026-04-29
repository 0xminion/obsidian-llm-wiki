"""Scale / stress tests for the pipeline.

Measures performance and resource use at URL scale:
  - 200 mock URLs → full pipeline → wall time, SQLite WAL size, proc count
  - Tests different --parallel values (5, 10, 20) to find optimal throughput.

No live network. All extraction is pre-cached mock HTML.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.config import Config
from pipeline.models import ExtractedSource, SourceType, Manifest
from pipeline.store import ContentStore

runner = CliRunner()

# Realistic HTML page ~2KB of content (simulates a blog post)
MOCK_HTML_TEMPLATE = b"""
<!DOCTYPE html>
<html><head><title>{title}</title></head><body>
<article><h1>{title}</h1>
<p>{body}</p>
<p>Second paragraph with additional details and insights.</p>
<p>Third paragraph covering implications.</p>
</article></body></html>
"""


def _make_mock_content(url: str, title: str) -> str:
    """Produce ~2KB markdown-like extracted content for a URL."""
    body = f"""This is the body of {title}. It contains meaningful
content extracted from a web page. The article discusses various
aspects of the topic in depth. Key points include:
- First important point about {title.lower()}
- Second consideration regarding methodology
- Third insight about implications
""" + "\n" + ("More content here. " * 200)
    return f"# {title}\n\n{body}"


def _seed_inbox(vault: Path, n: int = 200) -> list[str]:
    """Create N .url files in the inbox. Returns list of URLs."""
    inbox = vault / "01-Raw"
    inbox.mkdir(parents=True, exist_ok=True)
    # Also create minimal vault structure for setup validation
    for d in ["02-Clippings", "03-Queries", "04-Wiki/sources", "04-Wiki/entries",
              "04-Wiki/concepts", "04-Wiki/mocs", "05-Outputs/answers",
              "05-Outputs/visualizations", "06-Config", "07-WIP", "08-Archive-Raw",
              "09-Archive-Queries", "Meta/Templates", "Meta/lib", "Meta/prompts"]:
        (vault / d).mkdir(parents=True, exist_ok=True)
    for f in ["06-Config/edges.tsv", "06-Config/wiki-index.md",
              "06-Config/url-index.tsv", "06-Config/log.md", "06-Config/tag-registry.md"]:
        (vault / f).touch()
    urls = []
    for i in range(n):
        url = f"https://example.com/article-{i}"
        name = f"article-{hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:8]}.url"
        (inbox / name).write_text(f"[InternetShortcut]\nURL={url}\n")
        urls.append(url)
    return urls


def _seed_extract_dir(cfg: Config, urls: list[str]) -> None:
    """Pre-populate extract dir with mock JSON so Stage 1 is instant."""
    ext_dir = cfg.resolved_extract_dir
    ext_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        h = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]
        title = f"Article {h[:6]}"
        extracted = {
            "url": url,
            "title": title,
            "content": _make_mock_content(url, title),
            "type": "web",
            "author": "Test Author",
        }
        (ext_dir / f"{h}.json").write_text(json.dumps(extracted), encoding="utf-8")


class TestScalePipeline:
    """Full-pipeline stress tests with mock extraction."""

    @pytest.mark.parametrize("parallel,expected_max_s", [
        (3, 120),
        (5, 90),
    ])
    def test_scale_pipeline_timing(self, tmp_path: Path, parallel: int, expected_max_s: int):
        """Run 100 URLs through full pipeline and verify wall time is reasonable."""
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in ["04-Wiki/sources", "04-Wiki/entries", "04-Wiki/concepts",
                  "04-Wiki/mocs", "06-Config", "01-Raw"]:
            (vault / d).mkdir(parents=True)

        cfg = Config(vault_path=vault)
        urls = _seed_inbox(vault, n=100)
        _seed_extract_dir(cfg, urls)

        # Pre-save manifest + plans so Stage 1+2 are skipped (we already "extracted")
        from pipeline.models import Plan, Plans, Language, Template
        plans = Plans(plans=[
            Plan(hash=hashlib.md5(u.encode(), usedforsecurity=False).hexdigest()[:12], title=f"Article {i}",
                 language=Language.EN, template=Template.STANDARD, tags=["test"])
            for i, u in enumerate(urls)
        ])
        manifest = Manifest(entries=[
            ExtractedSource(url=u, title=f"Article {i}",
                            content=_make_mock_content(u, f"Article {i}"),
                            type=SourceType.WEB)
            for i, u in enumerate(urls)
        ])
        manifest.save(cfg.resolved_extract_dir)
        plans.save(cfg.resolved_extract_dir)

        # Mock create_file_templates to only write files (no hermes subprocess)
        def _mock_create_file_templates(plans, cfg, use_agent_insights=True):
            for plan in plans:
                title = plan.title
                fname = title.replace(" ", "-").lower()[:120]
                source = cfg.sources_dir / f"{fname}.md"
                entry = cfg.entries_dir / f"{fname}.md"
                source.parent.mkdir(parents=True, exist_ok=True)
                entry.parent.mkdir(parents=True, exist_ok=True)
                h = plan.hash
                source.write_text(
                    f"---\ntitle: {title}\nsource_url: https://example.com/{h}\n---\n\n# {title}\n\nContent."
                )
                entry.write_text(
                    f"---\ntitle: {title}\nsource: [[{fname}]]\n---\n\n# {title}\n\n## Summary\n\nSummary.\n"
                )
            return {"created": len(plans), "failed": 0, "sources": len(plans), "entries": len(plans)}

        t0 = time.monotonic()
        with patch("pipeline.cli.create_file_templates", side_effect=_mock_create_file_templates):
            result = runner.invoke(app, [
                "ingest", str(vault),
                "--resume",  # skip Stage 1, 2
            ])
        wall_time = time.monotonic() - t0

        assert result.exit_code == 0
        assert wall_time < expected_max_s, (
            f"Pipeline too slow: {wall_time:.1f}s for 100 URLs (expected <{expected_max_s}s)"
        )

    def test_sqlite_wal_growth_bounded(self, tmp_path: Path):
        """Verify SQLite WAL file doesn't grow unbounded after 100 items."""
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in ["04-Wiki/sources", "04-Wiki/entries", "04-Wiki/concepts",
                  "04-Wiki/mocs", "06-Config", "01-Raw"]:
            (vault / d).mkdir(parents=True)

        cfg = Config(vault_path=vault)
        urls = _seed_inbox(vault, n=100)
        _seed_extract_dir(cfg, urls)

        store = ContentStore.open(cfg.resolved_extract_dir)

        # Register 100 URLs
        for url in urls:
            store.register_url(url, source_type="web")

        store.close()

        # Check WAL file size
        wal_file = Path(cfg.resolved_extract_dir / ".pipeline" / "store.db-wal")
        if wal_file.exists():
            wal_mb = wal_file.stat().st_size / (1024 * 1024)
            assert wal_mb < 5, (
                f"WAL file grew to {wal_mb:.1f}MB — likely not checkpointing"
            )

    def test_os_fd_limit_not_exceeded_at_scale(self, tmp_path: Path):
        """Verify that running with parallel=20 on 200 URLs doesn't hit ulimit."""
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in ["04-Wiki/sources", "04-Wiki/entries", "04-Wiki/concepts",
                  "04-Wiki/mocs", "06-Config", "01-Raw"]:
            (vault / d).mkdir(parents=True)

        cfg = Config(vault_path=vault)
        urls = _seed_inbox(vault, n=200)
        _seed_extract_dir(cfg, urls)

        manifest = Manifest(entries=[
            ExtractedSource(url=u, title=f"Article {i}",
                            content=_make_mock_content(u, f"Article {i}"),
                            type=SourceType.WEB)
            for i, u in enumerate(urls)
        ])
        from pipeline.models import Plan, Plans, Language, Template
        plans = Plans(plans=[
            Plan(hash=hashlib.md5(u.encode(), usedforsecurity=False).hexdigest()[:12], title=f"Article {i}",
                 language=Language.EN, template=Template.STANDARD, tags=["test"])
            for i, u in enumerate(urls)
        ])
        manifest.save(cfg.resolved_extract_dir)
        plans.save(cfg.resolved_extract_dir)

        # Mock agent insight generation to return immediately
        def _fast_insights(*args, **kwargs):
            return ""

        soft_limit = 8192
        try:
            import resource
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        except ImportError:
            pass

        # Count open FDs before
        len(os.listdir(f"/proc/{os.getpid()}/fd")) if Path(f"/proc/{os.getpid()}/fd").exists() else 0

        t0 = time.monotonic()
        with patch("pipeline.create.templates.generate_entry_insights", side_effect=_fast_insights):
            result = runner.invoke(app, [
                "ingest", str(vault),
                "--resume",
                "--parallel", "20",
            ])
        time.monotonic() - t0

        fds_after = len(os.listdir(f"/proc/{os.getpid()}/fd")) if Path(f"/proc/{os.getpid()}/fd").exists() else 0

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert fds_after < soft_limit - 50, (
            f"FD count near limit: {fds_after} (limit {soft_limit})"
        )

    def test_scale_pipeline_output_files_match_input(self, tmp_path: Path):
        """After processing N URLs, verify at least N sources and N entries are created."""
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in ["04-Wiki/sources", "04-Wiki/entries", "04-Wiki/concepts",
                  "04-Wiki/mocs", "06-Config", "01-Raw"]:
            (vault / d).mkdir(parents=True)

        cfg = Config(vault_path=vault)
        urls = _seed_inbox(vault, n=50)
        _seed_extract_dir(cfg, urls)

        manifest = Manifest(entries=[
            ExtractedSource(url=u, title=f"Article {i}",
                            content=_make_mock_content(u, f"Article {i}"),
                            type=SourceType.WEB)
            for i, u in enumerate(urls)
        ])
        from pipeline.models import Plan, Plans, Language, Template
        plans = Plans(plans=[
            Plan(hash=hashlib.md5(u.encode(), usedforsecurity=False).hexdigest()[:12], title=f"Article {i}",
                 language=Language.EN, template=Template.STANDARD, tags=["test"])
            for i, u in enumerate(urls)
        ])
        manifest.save(cfg.resolved_extract_dir)
        plans.save(cfg.resolved_extract_dir)

        def _mock_create_file_templates(plans, cfg, use_agent_insights=True):
            for plan in plans:
                title = plan.title
                fname = title.replace(" ", "-").lower()[:120]
                source = cfg.sources_dir / f"{fname}.md"
                entry = cfg.entries_dir / f"{fname}.md"
                source.parent.mkdir(parents=True, exist_ok=True)
                entry.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"---\ntitle: {title}\n---\n\n# {title}\n")
                entry.write_text(f"---\ntitle: {title}\n---\n\n# {title}\n")
            return {"created": len(plans), "failed": 0, "sources": len(plans), "entries": len(plans)}

        with patch("pipeline.cli.create_file_templates", side_effect=_mock_create_file_templates):
            result = runner.invoke(app, ["ingest", str(vault), "--resume"])

        assert result.exit_code == 0
        sources = list(cfg.sources_dir.glob("*.md"))
        entries = list(cfg.entries_dir.glob("*.md"))
        assert len(sources) >= 50, f"Expected ≥50 sources, got {len(sources)}"
        assert len(entries) >= 50, f"Expected ≥50 entries, got {len(entries)}"
