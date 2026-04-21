"""Lightweight pipeline metrics — tracks agent calls, timing, and approximate token usage.

No external dependencies. Estimates tokens as chars/4 (standard heuristic).

Usage:
    from pipeline.metrics import get_metrics, start_stage, end_stage

    # In pipeline code:
    start_stage("plan")
    # ... agent calls happen ...
    end_stage("plan")

    # At the end:
    summary = get_metrics().summary()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    name: str
    agent_calls: int = 0
    prompt_chars: int = 0
    output_chars: int = 0
    duration_s: float = 0.0
    _start: Optional[float] = None

    @property
    def estimated_prompt_tokens(self) -> int:
        return self.prompt_chars // 4

    @property
    def estimated_output_tokens(self) -> int:
        return self.output_chars // 4

    @property
    def estimated_total_tokens(self) -> int:
        return self.estimated_prompt_tokens + self.estimated_output_tokens


@dataclass
class PipelineMetrics:
    """Aggregate metrics for a full pipeline run."""
    stages: dict[str, StageMetrics] = field(default_factory=dict)
    _active_stage: Optional[str] = None
    _run_start: Optional[float] = None

    def start_run(self) -> None:
        self._run_start = time.monotonic()

    def start_stage(self, name: str) -> None:
        if name not in self.stages:
            self.stages[name] = StageMetrics(name=name)
        stage = self.stages[name]
        stage._start = time.monotonic()
        self._active_stage = name

    def end_stage(self, name: str) -> None:
        if name in self.stages and self.stages[name]._start is not None:
            stage = self.stages[name]
            stage.duration_s += time.monotonic() - stage._start
            stage._start = None
        if self._active_stage == name:
            self._active_stage = None

    def record_agent_call(self, prompt_chars: int, output_chars: int = 0) -> None:
        """Record a single agent call in the active stage."""
        stage_name = self._active_stage or "unknown"
        if stage_name not in self.stages:
            self.stages[stage_name] = StageMetrics(name=stage_name)
        stage = self.stages[stage_name]
        stage.agent_calls += 1
        stage.prompt_chars += prompt_chars
        stage.output_chars += output_chars

    @property
    def total_duration_s(self) -> float:
        if self._run_start is None:
            return sum(s.duration_s for s in self.stages.values())
        return time.monotonic() - self._run_start

    @property
    def total_agent_calls(self) -> int:
        return sum(s.agent_calls for s in self.stages.values())

    @property
    def total_estimated_tokens(self) -> int:
        return sum(s.estimated_total_tokens for s in self.stages.values())

    def summary(self) -> str:
        """Format metrics as a human-readable summary."""
        lines = [
            f"Pipeline metrics: {self.total_agent_calls} agent calls, "
            f"~{self.total_estimated_tokens:,} tokens, "
            f"{self.total_duration_s:.1f}s total",
            "",
        ]
        for name, stage in self.stages.items():
            if stage.agent_calls == 0 and stage.duration_s < 0.1:
                continue
            lines.append(
                f"  {name}: {stage.agent_calls} calls, "
                f"~{stage.estimated_total_tokens:,} tokens, "
                f"{stage.duration_s:.1f}s"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "total_duration_s": round(self.total_duration_s, 1),
            "total_agent_calls": self.total_agent_calls,
            "total_estimated_tokens": self.total_estimated_tokens,
            "stages": {
                name: {
                    "agent_calls": s.agent_calls,
                    "estimated_tokens": s.estimated_total_tokens,
                    "duration_s": round(s.duration_s, 1),
                }
                for name, s in self.stages.items()
            },
        }


# ─── Global instance ────────────────────────────────────────────────────────

_metrics: Optional[PipelineMetrics] = None


def get_metrics() -> PipelineMetrics:
    """Get or create the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PipelineMetrics()
    return _metrics


def reset_metrics() -> PipelineMetrics:
    """Create a fresh metrics instance."""
    global _metrics
    _metrics = PipelineMetrics()
    _metrics.start_run()
    return _metrics


def start_stage(name: str) -> None:
    get_metrics().start_stage(name)


def end_stage(name: str) -> None:
    get_metrics().end_stage(name)


def record_agent_call(prompt_chars: int, output_chars: int = 0) -> None:
    get_metrics().record_agent_call(prompt_chars, output_chars)
