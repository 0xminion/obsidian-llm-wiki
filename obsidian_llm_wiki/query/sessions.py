"""Durable, local JSON storage for provenance-rich query sessions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["QuerySession", "QuerySessionStore", "create_session"]


@dataclass(frozen=True, slots=True)
class QuerySession:
    """One query interaction plus the retrieval evidence used to answer it."""

    session_id: str
    query: str
    retrieved_paths: tuple[str, ...]
    retrieval_trace: Mapping[str, Any]
    profile: str = ""
    instructions: str = ""
    answer: str = ""
    citation_paths: tuple[str, ...] = ()
    created_at: str = ""


def create_session(
    session_id: str,
    query: str,
    retrieved_paths: tuple[str, ...],
    retrieval_trace: Mapping[str, Any],
    *,
    profile: str = "",
    instructions: str = "",
    answer: str = "",
    citation_paths: tuple[str, ...] = (),
    created_at: str | None = None,
) -> QuerySession:
    """Create a session using the render timestamp helper only when needed."""
    if created_at is None:
        from obsidian_llm_wiki.render.frontmatter import timestamp

        created_at = timestamp()
    return QuerySession(
        session_id=session_id,
        query=query,
        retrieved_paths=tuple(retrieved_paths),
        retrieval_trace=dict(retrieval_trace),
        profile=profile,
        instructions=instructions,
        answer=answer,
        citation_paths=tuple(citation_paths),
        created_at=created_at,
    )


class QuerySessionStore:
    """A small atomic JSON-file store keyed by explicit session IDs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, session: QuerySession) -> QuerySession:
        """Insert or replace one session, retaining all other stored sessions."""
        sessions = {existing.session_id: existing for existing in self.list()}
        sessions[session.session_id] = session
        ordered = sorted(
            sessions.values(), key=lambda item: (item.created_at, item.session_id)
        )
        payload = {
            "version": 1,
            "sessions": [_session_to_dict(item) for item in ordered],
        }
        from obsidian_llm_wiki.render.frontmatter import atomic_write

        atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return session

    def load(self, session_id: str) -> QuerySession | None:
        """Load a session by ID, returning ``None`` if the ID is absent."""
        return next((session for session in self.list() if session.session_id == session_id), None)

    def list(self) -> tuple[QuerySession, ...]:
        """Return validated stored sessions in deterministic chronological order."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ()
        if not isinstance(payload, dict) or not isinstance(payload.get("sessions"), list):
            return ()
        sessions = [
            _session_from_dict(item)
            for item in payload["sessions"]
            if isinstance(item, dict)
        ]
        valid = [session for session in sessions if session is not None]
        return tuple(sorted(valid, key=lambda item: (item.created_at, item.session_id)))


def _session_to_dict(session: QuerySession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "query": session.query,
        "retrieved_paths": list(session.retrieved_paths),
        "retrieval_trace": dict(session.retrieval_trace),
        "profile": session.profile,
        "instructions": session.instructions,
        "answer": session.answer,
        "citation_paths": list(session.citation_paths),
        "created_at": session.created_at,
    }


def _session_from_dict(data: Mapping[str, Any]) -> QuerySession | None:
    session_id = data.get("session_id")
    query = data.get("query")
    trace = data.get("retrieval_trace")
    if (
        not isinstance(session_id, str)
        or not isinstance(query, str)
        or not isinstance(trace, Mapping)
    ):
        return None
    return QuerySession(
        session_id=session_id,
        query=query,
        retrieved_paths=_strings(data.get("retrieved_paths")),
        retrieval_trace=dict(trace),
        profile=_string(data.get("profile")),
        instructions=_string(data.get("instructions")),
        answer=_string(data.get("answer")),
        citation_paths=_strings(data.get("citation_paths")),
        created_at=_string(data.get("created_at")),
    )


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))
