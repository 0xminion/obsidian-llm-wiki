"""Task-specific model selection with a backwards-compatible unified fallback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskModelConfig:
    """Configure a default model and optional model overrides per task."""

    model: str
    ingest_model: str | None = None
    maintenance_model: str | None = None
    query_model: str | None = None
    expand_model: str | None = None

    def __post_init__(self) -> None:
        """Reject explicitly configured task models that contain no model name."""
        for field_name in (
            "ingest_model",
            "maintenance_model",
            "query_model",
            "expand_model",
        ):
            override = getattr(self, field_name)
            if override is not None and not override.strip():
                raise ValueError(f"{field_name} must not be blank when provided")


_SUPPORTED_TASKS = frozenset({"ingest", "maintenance", "query", "expand"})


def resolve_task_model(config: TaskModelConfig, task: str) -> str:
    """Return the model configured for *task*, falling back to ``config.model``."""
    if task not in _SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task: {task!r}. Supported: ingest, maintenance, query, expand."
        )

    override = getattr(config, f"{task}_model")
    return override if override is not None else config.model
