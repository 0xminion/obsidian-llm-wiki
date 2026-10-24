"""Reproducible synthetic pipeline benchmark (no network or LLM service).

Run each size in a fresh process for meaningful peak RSS:
  python scripts/benchmark_pipeline.py --sources 100 --output result.json
Exercises discovery, synthesis orchestration, merge, SQLite, note/graph writes,
unchanged build and one changed source. The synthesis function is deterministic;
this does NOT benchmark extraction, model latency, or real summary quality.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import resource
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import BodySection, ConceptNote, SourceSynthesis
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.ingest.sources import load_sources_incremental


async def benchmark(count: int, root: Path) -> dict:
    config = Config(vault_path=str(root), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    for index in range(count):
        (config.sources_dir / f"source-{index}.md").write_text(
            f"---\ntitle: Experiment {index}\n---\n"
            f"Experiment {index} measured an increase from 10 to 15 units under condition A. "
            "The control remained at 10 units. This isolates the effect of condition A.",
            encoding="utf-8",
        )
    calls = 0

    async def synthesize(_config, filename, source, _candidates, **_kwargs):
        nonlocal calls
        calls += 1
        slug = Path(filename).stem
        return SourceSynthesis(
            source.title, source.content, language="en", source_file=filename,
            key_points=[source.content],
            concepts=[ConceptNote(
                source.title, slug, source.content,
                sections=[BodySection("Observation", prose=source.content)],
            )],
        )

    async def run():
        started = time.perf_counter()
        with VaultRegistry(config.llmwiki_dir) as registry:
            loaded = load_sources_incremental(config.sources_dir, registry)
        discovered = time.perf_counter()
        before_calls = calls
        result = await run_pipeline(
            root, loaded.sources, config,
            unchanged_filenames=loaded.unchanged_filenames,
            source_file_signatures=loaded.signatures,
        )
        if result.errors:
            raise RuntimeError(result.errors)
        return {
            "seconds": time.perf_counter() - started,
            "discovery_seconds": discovered - started,
            "synthesis_calls": calls - before_calls,
            "compiled": result.compiled, "skipped": result.skipped,
            "written_paths": len(result.pages),
        }

    with patch("obsidian_llm_wiki.core.pipeline._synthesize_source", synthesize):
        initial = await run()
        unchanged = await run()
        changed = config.sources_dir / "source-0.md"
        changed.write_text(changed.read_text() + "\nThe follow-up measured 16 units.")
        update = await run()
    assert initial["compiled"] == count
    assert unchanged["synthesis_calls"] == 0 and unchanged["written_paths"] == 0
    assert update["synthesis_calls"] == 1 and update["compiled"] == 1
    notes = sum(1 for p in config.wiki_dir.rglob("*.md") if ".llmwiki" not in p.parts)
    return {
        "sources": count, "notes": notes,
        "initial": initial, "unchanged": unchanged, "one_changed": update,
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python": platform.python_version(), "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version, "embeddings": False,
        "synthesis": "deterministic fixture, no external LLM or extraction",
        "vault": str(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sources < 1:
        parser.error("--sources must be positive")
    root = Path(tempfile.mkdtemp(prefix=f"wiki-benchmark-{args.sources}-"))
    result = asyncio.run(benchmark(args.sources, root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
