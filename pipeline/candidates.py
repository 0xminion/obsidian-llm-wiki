"""Review candidate system for pending wiki pages.

Ported from obsidian-llm-wiki/src/compiler/candidates.ts.

Manages draft pages that need human review:
  - Writing candidates as JSON to .llmwiki/candidates/
  - Reading, listing, approving, and rejecting candidates
"""

from __future__ import annotations

import contextlib
import json
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pipeline.okf_markdown import atomic_write, build_frontmatter, safe_read_file, slugify
from pipeline.okf_models import ReviewCandidate, SourceState
from pipeline.state import read_state, write_state

# ── Candidate I/O ───────────────────────────────────────────────────────


def write_candidate(root_dir: str | Path, draft: dict) -> ReviewCandidate:
    """Write a candidate page as JSON to .llmwiki/candidates/.

    Args:
        root_dir: Wiki root directory.
        draft: Dict with keys: title, slug, summary, sources, body, source_states,
               schema_violations, provenance_violations.

    Returns:
        The ReviewCandidate that was persisted.
    """
    root = Path(root_dir)
    candidates_dir = root / ".llmwiki" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique id: slug + random 4-byte hex suffix
    title = draft.get("title", "Untitled")
    slug = draft.get("slug") or slugify(title)
    suffix = secrets.token_hex(4)
    candidate_id = f"{slug}-{suffix}"

    now = datetime.now(UTC).isoformat()

    # Build source states from dict
    source_states_raw = draft.get("source_states", {})
    source_states: dict[str, SourceState] = {}
    for filename, entry in source_states_raw.items():
        if isinstance(entry, SourceState):
            source_states[filename] = entry
        elif isinstance(entry, dict):
            source_states[filename] = SourceState(
                hash=entry.get("hash", ""),
                concepts=entry.get("concepts", []),
                compiled_at=entry.get("compiled_at") or entry.get("compiledAt"),
            )

    candidate = ReviewCandidate(
        id=candidate_id,
        title=title,
        slug=slug,
        summary=draft.get("summary", ""),
        sources=draft.get("sources", []),
        body=draft.get("body", ""),
        generated_at=now,
        source_states=source_states,
        schema_violations=draft.get("schema_violations"),
        provenance_violations=draft.get("provenance_violations"),
    )

    # Serialize
    data = _serialize_candidate(candidate)

    candidate_path = candidates_dir / f"{candidate_id}.json"
    atomic_write(candidate_path, json.dumps(data, indent=2, ensure_ascii=False))

    return candidate


def read_candidate(root_dir: str | Path, candidate_id: str) -> ReviewCandidate | None:
    """Read a candidate by its id.

    Args:
        root_dir: Wiki root directory.
        candidate_id: The candidate's unique id (slug-hexsuffix).

    Returns:
        ReviewCandidate if found, None otherwise.
    """
    root = Path(root_dir)
    candidate_path = root / ".llmwiki" / "candidates" / f"{candidate_id}.json"

    raw = safe_read_file(candidate_path)
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return _deserialize_candidate(data)


def list_candidates(root_dir: str | Path) -> list[ReviewCandidate]:
    """List all pending review candidates.

    Args:
        root_dir: Wiki root directory.

    Returns:
        List of ReviewCandidate objects sorted by generation time (newest first).
    """
    root = Path(root_dir)
    candidates_dir = root / ".llmwiki" / "candidates"

    if not candidates_dir.is_dir():
        return []

    results: list[ReviewCandidate] = []

    for json_file in sorted(candidates_dir.glob("*.json")):
        if json_file.name.startswith("."):
            continue
        raw = safe_read_file(json_file)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        candidate = _deserialize_candidate(data)
        if candidate:
            results.append(candidate)

    # Newest first
    results.sort(key=lambda c: c.generated_at, reverse=True)
    return results


# ── Approval / Rejection ────────────────────────────────────────────────


def approve_candidate(root_dir: str | Path, candidate_id: str,
                      wiki_dir: str | Path) -> bool:
    """Approve a candidate: write its page to wiki/concepts/ and update state.

    Args:
        root_dir: Wiki root directory.
        candidate_id: The candidate's unique id.
        wiki_dir: Path to the wiki directory (e.g., 04-Wiki).

    Returns:
        True on success, False if candidate not found.
    """
    candidate = read_candidate(root_dir, candidate_id)
    if candidate is None:
        return False

    root = Path(root_dir)
    wiki = Path(wiki_dir)
    concepts_dir = wiki / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    # Write the page
    fm = {
        "title": candidate.title,
        "slug": candidate.slug,
        "summary": candidate.summary,
    }
    page_content = build_frontmatter(fm) + "\n" + candidate.body

    page_path = concepts_dir / f"{candidate.slug}.md"
    atomic_write(page_path, page_content)

    # Update state with source states from candidate
    state_path = wiki / ".llmwiki" / "state.json"
    state = read_state(state_path)

    for filename, source_state in candidate.source_states.items():
        if filename not in state.sources:
            state.sources[filename] = source_state
        else:
            existing = state.sources[filename]
            combined_concepts = list(set(existing.concepts + source_state.concepts))
            state.sources[filename] = SourceState(
                hash=existing.hash,
                concepts=combined_concepts,
                compiled_at=existing.compiled_at,
            )

    write_state(state_path, state)

    # Remove the candidate file
    candidate_path = root / ".llmwiki" / "candidates" / f"{candidate_id}.json"
    with contextlib.suppress(FileNotFoundError):
        candidate_path.unlink()

    return True


def reject_candidate(root_dir: str | Path, candidate_id: str) -> bool:
    """Reject a candidate: move its JSON to .llmwiki/candidates/archive/.

    Args:
        root_dir: Wiki root directory.
        candidate_id: The candidate's unique id.

    Returns:
        True on success, False if candidate not found.
    """
    root = Path(root_dir)
    candidates_dir = root / ".llmwiki" / "candidates"
    archive_dir = candidates_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = candidates_dir / f"{candidate_id}.json"

    if not candidate_path.exists():
        return False

    # Move to archive
    archive_path = archive_dir / f"{candidate_id}.json"
    shutil.move(str(candidate_path), str(archive_path))

    return True


# ── Serialization helpers ───────────────────────────────────────────────


def _serialize_candidate(c: ReviewCandidate) -> dict:
    """Convert ReviewCandidate to a JSON-serializable dict."""
    return {
        "id": c.id,
        "title": c.title,
        "slug": c.slug,
        "summary": c.summary,
        "sources": c.sources,
        "body": c.body,
        "generated_at": c.generated_at,
        "source_states": {
            filename: {
                "hash": ss.hash,
                "concepts": ss.concepts,
                "compiled_at": ss.compiled_at,
            }
            for filename, ss in c.source_states.items()
        },
        "schema_violations": c.schema_violations,
        "provenance_violations": c.provenance_violations,
    }


def _deserialize_candidate(data: dict) -> ReviewCandidate | None:
    """Convert a deserialized dict to a ReviewCandidate."""
    try:
        source_states: dict[str, SourceState] = {}
        for filename, entry in data.get("source_states", {}).items():
            source_states[filename] = SourceState(
                hash=entry.get("hash", ""),
                concepts=entry.get("concepts", []),
                compiled_at=entry.get("compiled_at"),
            )

        return ReviewCandidate(
            id=data["id"],
            title=data["title"],
            slug=data["slug"],
            summary=data.get("summary", ""),
            sources=data.get("sources", []),
            body=data.get("body", ""),
            generated_at=data.get("generated_at", ""),
            source_states=source_states,
            schema_violations=data.get("schema_violations"),
            provenance_violations=data.get("provenance_violations"),
        )
    except (KeyError, TypeError):
        return None
