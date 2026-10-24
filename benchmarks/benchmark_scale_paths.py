"""Offline synthetic path benchmark; not a production/million-note capacity claim.

Run with the project's Python. All fixtures are temporary under --scratch.
Query mode measures real Markdown I/O, graph construction, and retrieval (no LLM).
Registry mode measures production per-source upserts, not an end-to-end build.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import socket
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis  # noqa: E402
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry  # noqa: E402
from obsidian_llm_wiki.query.graph import build_graph_from_vault  # noqa: E402
from obsidian_llm_wiki.query.retrieval import retrieve  # noqa: E402


def no_network(*_args, **_kwargs):
    raise RuntimeError("Network forbidden during synthetic benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["query", "registry"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.scratch.is_dir() or any(n < 1 or n > 100000 for n in args.sizes):
        parser.error("Use an existing scratch directory and sizes between 1 and 100000")
    socket.create_connection = no_network
    socket.socket.connect = no_network
    rows = []
    with tempfile.TemporaryDirectory(prefix="wiki-scale-measured-", dir=args.scratch) as tmp:
        root = Path(tmp)
        previous = 0
        registry = VaultRegistry(root / "index") if args.mode == "registry" else None
        try:
            for n in sorted(set(args.sizes)):
                start = time.perf_counter()
                for i in range(previous, n):
                    title = f"Concept {i}"
                    body = (
                        f"needle evidence {i}. "
                        + ("[[p1]] " if i == 0 else "")
                        + "Local evidence. " * 20
                    )
                    if registry is None:
                        (root / f"p{i}.md").write_text(
                            f"---\ntitle: {title}\n---\n{body}",
                            encoding="utf8",
                        )
                    else:
                        registry.upsert_synthesis(
                            f"p{i}.md",
                            SourceDoc(title, body, source_file=f"p{i}.md"),
                            SourceSynthesis(
                                source_title=title,
                                source_summary=body,
                                source_file=f"p{i}.md",
                                concepts=[ConceptNote(title, f"p{i}", body)],
                            ),
                            ArtifactVersions("e", "p", "s", "m"),
                        )
                ingest_seconds = time.perf_counter() - start
                gc.collect()
                start = time.perf_counter()
                if registry is None:
                    graph = build_graph_from_vault(root)
                    graph_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    results = retrieve("needle", graph, max_results=10)
                    details = {
                        "markdown_pages_read": graph.page_count,
                        "edges": graph.edge_count,
                        "graph_seconds": graph_seconds,
                        "retrieval_seconds": time.perf_counter() - start,
                        "results": len(results.candidates),
                        "strategy": results.trace.strategy,
                    }
                    assert graph.page_count == n
                    del graph, results
                else:
                    results = registry.search_concepts("needle", limit=10)
                    details = {
                        "sources": registry.connection.execute(
                            "SELECT count(*) FROM source"
                        ).fetchone()[0],
                        "concepts": registry.connection.execute(
                            "SELECT count(*) FROM concept"
                        ).fetchone()[0],
                        "search_seconds": time.perf_counter() - start,
                        "results": len(results),
                        "registry_file_bytes": registry.path.stat().st_size,
                    }
                    assert details["sources"] == details["concepts"] == n
                row = {
                    "mode": args.mode,
                    "entities": n,
                    "new_entities": n - previous,
                    "fixture_or_upsert_seconds": ingest_seconds,
                    "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    **details,
                }
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2), encoding="utf8")
                print(json.dumps(row), flush=True)
                previous = n
        finally:
            if registry is not None:
                registry.close()


if __name__ == "__main__":
    main()
