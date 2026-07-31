from __future__ import annotations

import threading
import time
from pathlib import Path

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli.ingest import _backfill_ingest_index, _BoundedURLExtractor
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.registry import VaultRegistry


def test_url_extraction_prefetch_is_bounded_by_one_worker_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_extract(url: str) -> SourceDoc:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return SourceDoc(title=url.rsplit("/", 1)[-1], content=f"content for {url}")

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", fake_extract)
    urls = [f"https://example.com/{index}" for index in range(8)]
    config = Config(vault_path=str(tmp_path))

    with _BoundedURLExtractor(urls, config, workers=3) as extractor:
        results = [extractor.extract(url) for url in urls]

    assert [result.title for result in results] == [str(index) for index in range(8)]
    assert 1 < peak <= 3


def test_indexed_url_duplicate_skips_extraction_entirely(
    tmp_path: Path, monkeypatch
) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module

    config = Config(vault_path=str(tmp_path))
    monkeypatch.setattr(
        ingest_module, "resolve_vault", lambda _vault: (tmp_path, config)
    )
    url = "https://example.test/scalable"
    calls = 0

    def first_extract(requested_url: str) -> SourceDoc:
        nonlocal calls
        calls += 1
        return SourceDoc(
            title="Scalable Source",
            content="A sufficiently long source body for indexed duplicate testing. " * 3,
            url=requested_url,
        )

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", first_extract)
    runner = CliRunner()
    first = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--url", url, "--skip-synthesis", "--json"],
    )
    assert first.exit_code == 0, first.output
    assert calls == 1

    def forbidden_extract(_requested_url: str) -> SourceDoc:
        raise AssertionError("indexed duplicate must not be extracted again")

    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract", forbidden_extract
    )
    second = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--url", url, "--skip-synthesis", "--json"],
    )
    assert second.exit_code == 0, second.output
    assert '"duplicate_of"' in second.output


def test_interrupted_ingest_backfill_resumes_until_marker_is_written(
    tmp_path: Path,
) -> None:
    sources_dir = tmp_path / "01-Sources"
    sources_dir.mkdir()
    for index in range(3):
        (sources_dir / f"source-{index}.md").write_text(
            "---\n"
            f"title: Source {index}\n"
            f"url: https://example.test/{index}\n"
            "---\n"
            f"# Source {index}\n\nBody {index}\n",
            encoding="utf-8",
        )

    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_ingest_hash(
            requested_url="https://example.test/0",
            canonical_url="https://example.test/0",
            content_hash="partial",
            output_file="source-0.md",
        )
        _backfill_ingest_index(registry, sources_dir)
        matches = [
            registry.find_ingest_duplicate(
                requested_url=f"https://example.test/{index}"
            )
            for index in range(3)
        ]
        complete = registry.ingest_backfill_complete()

    assert all(match is not None for match in matches)
    assert complete
