"""Agent-native task/response bridge.

When cfg.agent_native=True, Stages 2-4 do NOT call external LLMs or use
deterministic heuristics.  Instead they emit structured task files under
``{extract_dir}/.agent-bridge/tasks/`` and wait for the running agent
(the human or AI agent driving this session) to write a response file
under ``{extract_dir}/.agent-bridge/responses/``.

This module is intentionally thin: it handles ONLY file I/O,
serialization, and idempotency.  All semantic reasoning lives in the
running agent, not in this module.

Design rules
------------
1. A task file is immutable once emitted (identified by ``task_id``).
2. A response overwrites any previous response for the same ``task_id``.
3. The pipeline never deletes tasks or responses — they are an audit trail.
4. If a response is partially invalid, the pipeline logs a warning and falls
   back to an empty result (never to deterministic heuristics when
   ``agent_native`` is active).

Flow
----
.. code-block:: text

   Stage 1 (Extract) → deterministic Python
   Stage 2 (Plan)    → emit plan task
                        ↓ agent processes task
                        ↓ agent writes response
   Stage 3 (Create)  → consume plan response → emit create tasks
                        ↓ agent processes tasks
                        ↓ agent writes responses
   Stage 4 (Compile) → consume create responses → emit compile task
                        ↓ agent processes task
                        ↓ agent writes response
   Pipeline finishes → consume compile response
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TASKS_SUBDIR = ".agent-bridge/tasks"
RESPONSES_SUBDIR = ".agent-bridge/responses"


@dataclass
class Task:
    """A single agent task."""

    task_type: str
    task_id: str
    payload: dict[str, Any]
    created: str


@dataclass
class Response:
    """A single agent response."""

    task_type: str
    task_id: str
    result: dict[str, Any]
    created: str


class AgentBridge:
    """Task/response file bridge."""

    def __init__(self, base_dir: Path) -> None:
        self.tasks_dir = base_dir / TASKS_SUBDIR
        self.responses_dir = base_dir / RESPONSES_SUBDIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    # ─── Task I/O ────────────────────────────────────────────────────────

    def emit_task(self, task_type: str, task_id: str, payload: dict[str, Any]) -> Path:
        """Emit a task file.  Idempotent: if the file already exists, return the
        existing path without overwriting it."""
        path = self.tasks_dir / f"{task_id}.json"
        if path.exists():
            log.debug("Task %s already emitted", task_id)
            return path
        data = {
            "version": 1,
            "type": task_type,
            "id": task_id,
            "payload": payload,
            "created": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Agent-native: emitted %s task %s", task_type, task_id)
        return path

    def get_task(self, task_id: str) -> Task | None:
        path = self.tasks_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Task(
                task_type=data["type"],
                task_id=data["id"],
                payload=data.get("payload", {}),
                created=data.get("created", ""),
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning("Corrupt task file %s: %s", path, e)
            return None

    # ─── Response I/O ────────────────────────────────────────────────────

    def write_response(self, task_type: str, task_id: str, result: dict[str, Any]) -> Path:
        """Write a response file.  Overwrites any previous response."""
        path = self.responses_dir / f"{task_id}.json"
        data = {
            "version": 1,
            "type": task_type,
            "id": task_id,
            "result": result,
            "created": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Agent-native: wrote response for %s task %s", task_type, task_id)
        return path

    def has_response(self, task_id: str) -> bool:
        return (self.responses_dir / f"{task_id}.json").exists()

    def consume_response(self, task_id: str) -> Response | None:
        """Read a response file.  Returns None if missing or corrupt.
        The file is left on disk for audit (not deleted)."""
        path = self.responses_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Response(
                task_type=data["type"],
                task_id=data["id"],
                result=data.get("result", {}),
                created=data.get("created", ""),
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning("Corrupt response file %s: %s", path, e)
            return None

    # ─── Batch helpers ───────────────────────────────────────────────────

    def get_pending(self, task_type: str = "") -> list[Task]:
        """Return tasks that do not yet have a response.

        Optionally filter by ``task_type``."""
        pending: list[Task] = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            task = self.get_task(path.stem)
            if task is None:
                continue
            if task_type and task.task_type != task_type:
                continue
            if not self.has_response(task.task_id):
                pending.append(task)
        return pending

    def get_all_responses(self, task_type: str = "") -> list[Response]:
        """Return all responses, optionally filtered by type."""
        responses: list[Response] = []
        for path in sorted(self.responses_dir.glob("*.json")):
            resp = self.consume_response(path.stem)
            if resp is None:
                continue
            if task_type and resp.task_type != task_type:
                continue
            responses.append(resp)
        return responses

    # ─── High-level state machine helpers ────────────────────────────────

    def waiting_message(self, pending: list[Task]) -> str:
        """Return a user-friendly message listing pending tasks."""
        lines = [
            "=" * 60,
            "  AGENT-NATIVE MODE — pending tasks",
            "=" * 60,
            "",
        ]
        for t in pending:
            lines.append(f"  [{t.task_type}] {t.task_id}")
            lines.append(f"    File: {self.tasks_dir / f'{t.task_id}.json'}")
            lines.append("")
        lines.extend([
            "Process each task, then write responses to:",
            f"  {self.responses_dir}",
            "",
            "Then re-run the pipeline to continue.",
            "=" * 60,
        ])
        return "\n".join(lines)


def get_bridge(cfg: Any) -> AgentBridge:
    """Factory: create an AgentBridge rooted in the configured extract dir."""
    return AgentBridge(cfg.resolved_extract_dir)
