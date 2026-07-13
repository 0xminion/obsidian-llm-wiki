"""Observability/metrics collection for the obsidian-llm-wiki pipeline.

Tracks per-run metrics for extraction, synthesis, rendering, and embedding
phases, then persists them to ``.llmwiki/metrics.json``.

Usage::

    from obsidian_llm_wiki.core.metrics import MetricsCollector

    metrics = MetricsCollector(vault_path)
    metrics.start_run()
    metrics.record_extraction(...)
    metrics.record_synthesis(...)
    metrics.record_rendering(...)
    metrics.record_embedding(...)
    metrics.finish_run()
    metrics.save()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("obswiki.core.metrics")

__all__ = [
    "MetricsCollector",
    "RunMetrics",
    "load_metrics",
    "load_all_metrics",
    "print_metrics_summary",
]


# ── Metric record dataclasses ──────────────────────────────────────────────


@dataclass
class ExtractionMetric:
    """Per-source extraction metric."""

    url: str = ""
    chars_extracted: int = 0
    extractor_used: str = ""
    time_seconds: float = 0.0
    success: bool = True


@dataclass
class SynthesisMetric:
    """Per-source synthesis metric."""

    source_file: str = ""
    pass1_time: float = 0.0
    pass2_time: float = 0.0
    concepts_extracted: int = 0
    success: bool = True
    error_type: str = ""


@dataclass
class RenderingMetric:
    """Rendering phase metric."""

    concepts_rendered: int = 0
    mocs_rendered: int = 0
    cross_lingual_links: int = 0
    backlinks_added: int = 0
    time_seconds: float = 0.0


@dataclass
class EmbeddingMetric:
    """Embedding phase metric."""

    model: str = ""
    concepts_embedded: int = 0
    cross_lingual_matches: int = 0
    time_seconds: float = 0.0


@dataclass
class RunMetrics:
    """Complete metrics for a single pipeline run."""

    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    total_time_seconds: float = 0.0
    extractions: list[ExtractionMetric] = field(default_factory=list)
    syntheses: list[SynthesisMetric] = field(default_factory=list)
    rendering: RenderingMetric = field(default_factory=RenderingMetric)
    embedding: EmbeddingMetric = field(default_factory=EmbeddingMetric)
    summary: dict[str, Any] = field(default_factory=dict)


# ── Collector ──────────────────────────────────────────────────────────────


class MetricsCollector:
    """Collects and persists pipeline run metrics.

    Attributes:
        vault_path: Path to the Obsidian vault root.
        metrics_file: Resolved path to ``.llmwiki/metrics.json``.
    """

    def __init__(self, vault_path: str | Path) -> None:
        self.vault_path = Path(vault_path).resolve()
        self.llmwiki_dir = self.vault_path / "04-Wiki" / ".llmwiki"
        self.metrics_file = self.llmwiki_dir / "metrics.json"
        self._metrics = RunMetrics()
        self._start_time: float = 0.0

    # ── Run lifecycle ───────────────────────────────────────────────────

    def start_run(self) -> None:
        """Begin a new run — records start timestamp."""
        self._metrics = RunMetrics()
        self._start_time = time.monotonic()
        now = datetime.now(UTC)
        self._metrics.run_id = now.strftime("%Y%m%d-%H%M%S")
        self._metrics.started_at = now.isoformat()

    def finish_run(self) -> None:
        """Finalise the run — records end timestamp and total time."""
        self._metrics.finished_at = datetime.now(UTC).isoformat()
        self._metrics.total_time_seconds = round(
            time.monotonic() - self._start_time, 3
        )
        self._compute_summary()

    # ── Recording individual phases ─────────────────────────────────────

    def record_extraction(
        self,
        url: str,
        chars_extracted: int,
        extractor_used: str,
        time_seconds: float,
        success: bool = True,
    ) -> None:
        """Record a single extraction event."""
        self._metrics.extractions.append(
            ExtractionMetric(
                url=url,
                chars_extracted=chars_extracted,
                extractor_used=extractor_used,
                time_seconds=round(time_seconds, 3),
                success=success,
            )
        )

    def record_synthesis(
        self,
        source_file: str,
        pass1_time: float = 0.0,
        pass2_time: float = 0.0,
        concepts_extracted: int = 0,
        success: bool = True,
        error_type: str = "",
    ) -> None:
        """Record a single synthesis event."""
        self._metrics.syntheses.append(
            SynthesisMetric(
                source_file=source_file,
                pass1_time=round(pass1_time, 3),
                pass2_time=round(pass2_time, 3),
                concepts_extracted=concepts_extracted,
                success=success,
                error_type=error_type,
            )
        )

    def record_rendering(
        self,
        concepts_rendered: int = 0,
        mocs_rendered: int = 0,
        cross_lingual_links: int = 0,
        backlinks_added: int = 0,
        time_seconds: float = 0.0,
    ) -> None:
        """Record rendering phase results."""
        self._metrics.rendering = RenderingMetric(
            concepts_rendered=concepts_rendered,
            mocs_rendered=mocs_rendered,
            cross_lingual_links=cross_lingual_links,
            backlinks_added=backlinks_added,
            time_seconds=round(time_seconds, 3),
        )

    def record_embedding(
        self,
        model: str = "",
        concepts_embedded: int = 0,
        cross_lingual_matches: int = 0,
        time_seconds: float = 0.0,
    ) -> None:
        """Record embedding phase results."""
        self._metrics.embedding = EmbeddingMetric(
            model=model,
            concepts_embedded=concepts_embedded,
            cross_lingual_matches=cross_lingual_matches,
            time_seconds=round(time_seconds, 3),
        )

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist metrics to ``metrics.json``.

        Creates the ``.llmwiki`` directory if it doesn't exist.
        Appends the current run to a ``runs`` list, keeping history.
        Also writes ``latest`` key for backward-compatible single-run access.
        """
        self.llmwiki_dir.mkdir(parents=True, exist_ok=True)
        current_run = self._to_dict()

        # Load existing metrics to preserve history.
        existing: dict[str, Any] = {}
        if self.metrics_file.exists():
            try:
                existing = json.loads(
                    self.metrics_file.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                existing = {}

        # Migrate legacy format (flat dict with run_id) to new format.
        if existing and "runs" not in existing:
            existing = {"runs": [existing], "latest": existing}

        # Append current run.
        runs = existing.get("runs", [])
        runs.append(current_run)
        # Keep last 50 runs to prevent unbounded growth.
        if len(runs) > 50:
            runs = runs[-50:]

        output = {"runs": runs, "latest": current_run}
        from obsidian_llm_wiki.render.obsidian import atomic_write

        atomic_write(
            self.metrics_file,
            json.dumps(output, indent=2, ensure_ascii=False),
        )
        logger.debug("Metrics saved to %s", self.metrics_file)

    def _to_dict(self) -> dict[str, Any]:
        """Serialise metrics to a JSON-compatible dict."""
        return asdict(self._metrics)

    def _compute_summary(self) -> None:
        """Compute aggregate summary statistics."""
        s = self._metrics.summary
        s["total_extractions"] = len(self._metrics.extractions)
        s["successful_extractions"] = sum(
            1 for e in self._metrics.extractions if e.success
        )
        s["total_syntheses"] = len(self._metrics.syntheses)
        s["successful_syntheses"] = sum(
            1 for syn in self._metrics.syntheses if syn.success
        )
        s["failed_syntheses"] = sum(
            1 for syn in self._metrics.syntheses if not syn.success
        )
        s["total_concepts_extracted"] = sum(
            syn.concepts_extracted for syn in self._metrics.syntheses
        )
        s["concepts_rendered"] = self._metrics.rendering.concepts_rendered
        s["mocs_rendered"] = self._metrics.rendering.mocs_rendered
        s["total_errors"] = (
            s["failed_syntheses"]
            + sum(1 for e in self._metrics.extractions if not e.success)
        )


# ── Standalone helpers ─────────────────────────────────────────────────────


def load_metrics(vault_path: str | Path) -> dict[str, Any] | None:
    """Load the latest metrics.json from a vault.

    Args:
        vault_path: Path to the Obsidian vault root.

    Returns:
        Parsed metrics dict for the latest run, or None if no metrics file exists.
        The dict is the latest run's data (backward-compatible with callers
        that expect a flat run dict). Use ``load_all_metrics`` for full history.
    """
    metrics_file = (
        Path(vault_path).resolve() / "04-Wiki" / ".llmwiki" / "metrics.json"
    )
    if not metrics_file.exists():
        return None
    try:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load metrics from %s: %s", metrics_file, exc)
        return None
    # New format: {"runs": [...], "latest": {...}}
    if isinstance(data, dict) and "latest" in data:
        return data["latest"]
    # Legacy format: flat dict
    return data


def load_all_metrics(vault_path: str | Path) -> list[dict[str, Any]]:
    """Load all historical metrics from a vault.

    Args:
        vault_path: Path to the Obsidian vault root.

    Returns:
        List of run metric dicts, oldest first. Empty list if no metrics.
    """
    metrics_file = (
        Path(vault_path).resolve() / "04-Wiki" / ".llmwiki" / "metrics.json"
    )
    if not metrics_file.exists():
        return []
    try:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load metrics from %s: %s", metrics_file, exc)
        return []
    if isinstance(data, dict) and "runs" in data:
        return data["runs"]
    # Legacy format: single flat dict
    if isinstance(data, dict) and data:
        return [data]
    return []


def print_metrics_summary(vault_path: str | Path) -> None:
    """Print a human-readable metrics summary from the latest run.

    Args:
        vault_path: Path to the Obsidian vault root.
    """
    data = load_metrics(vault_path)
    if data is None:
        print("No metrics found. Run 'olw build' first.")
        return

    print(f"📊 Metrics for run {data.get('run_id', '?')}")
    print(f"   Started:   {data.get('started_at', '?')}")
    print(f"   Finished:  {data.get('finished_at', '?')}")
    print(f"   Total time: {data.get('total_time_seconds', 0):.1f}s")

    summary = data.get("summary", {})
    print(f"\n   Extractions: {summary.get('total_extractions', 0)} "
          f"({summary.get('successful_extractions', 0)} successful)")
    print(f"   Syntheses:   {summary.get('total_syntheses', 0)} "
          f"({summary.get('successful_syntheses', 0)} ok, "
          f"{summary.get('failed_syntheses', 0)} failed)")
    print(f"   Concepts extracted: {summary.get('total_concepts_extracted', 0)}")
    print(f"   Concepts rendered:  {summary.get('concepts_rendered', 0)}")
    print(f"   MOCs rendered:      {summary.get('mocs_rendered', 0)}")

    # Per-source synthesis details
    syntheses = data.get("syntheses", [])
    if syntheses:
        print("\n   Per-source synthesis:")
        for syn in syntheses:
            status = "✅" if syn.get("success") else "❌"
            err = f" [{syn.get('error_type', '')}]" if syn.get("error_type") else ""
            print(
                f"     {status} {syn.get('source_file', '?')}: "
                f"{syn.get('concepts_extracted', 0)} concepts, "
                f"{syn.get('pass1_time', 0):.1f}s"
                f"{' + ' + str(syn.get('pass2_time', 0)) + 's' if syn.get('pass2_time') else ''}"
                f"{err}"
            )

    # Embedding info
    emb = data.get("embedding", {})
    if emb and emb.get("concepts_embedded", 0) > 0:
        print(f"\n   Embedding: {emb.get('model', '?')}, "
              f"{emb.get('concepts_embedded', 0)} concepts, "
              f"{emb.get('cross_lingual_matches', 0)} cross-lingual matches")
