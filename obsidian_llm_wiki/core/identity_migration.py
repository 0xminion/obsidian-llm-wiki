"""Read-only legacy identity inventory for a future identity-plan migration.

This module answers one question without touching the vault: *which legacy
identity bindings exist, what evidence backs each one, and which of them can be
carried into a new plan without re-assessment?* It never opens the registry
through :class:`~obsidian_llm_wiki.core.registry.VaultRegistry` (the ordinary
constructor negotiates WAL, runs schema migrations and writes ``meta``), never
contacts a provider, never acquires the compile lock and never creates a
missing vault, metadata directory, cache, sidecar or metrics file.

Registry access strategy
------------------------
``registry.db`` is read as plain bytes and loaded into an in-memory SQLite
connection (:meth:`sqlite3.Connection.deserialize`). That is only a consistent
view of the committed state when the main file *is* the whole database:

* a ``registry.db-wal`` with any frames may hold committed rows that the main
  file does not (``immutable=1`` readers miss them too, so it is never used as
  a substitute);
* a live connection (detected through the ``-shm`` DMS lock byte, a compile
  lock PID that is alive, or a hot rollback journal) means a writer may be in
  the middle of a transaction.

In those states the preview **refuses** with an actionable explanation rather
than guessing. Snapshot consistency is additionally guarded by re-hashing the
database bytes and re-stat'ing every vault entry after the inventory; a change
during the read makes the result ``inconsistent``.

Schema 7 is checked against the archived immutable publication plan and its
committed marker. Any remaining publication-stage journal blocks certification,
even if the marker says committed. This module never executes/reconstructs a
plan, recovers a journal, estimates unbounded model calls or authorizes apply.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import struct
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from obsidian_llm_wiki.core.entity_identity import (
    IDENTITY_RECORD_VERSION,
    assessment_records,
    definition_snapshot,
)
from obsidian_llm_wiki.core.generated_paths import (
    checked_generated_path,
    checked_metadata_directory,
)
from obsidian_llm_wiki.core.registry import _SCHEMA_VERSION, VaultRegistry
from obsidian_llm_wiki.render.frontmatter import extract_wikilinks, parse_frontmatter
from obsidian_llm_wiki.synth.review import KNOWN_REVIEW_RECEIPT_VERSIONS
from obsidian_llm_wiki.synth.review import REVIEW_RECEIPT_VERSION as _REVIEW_RECEIPT_VERSION

__all__ = [
    "PREVIEW_VERSION",
    "PreviewRefusedError",
    "preview_identity_migration",
]

PREVIEW_VERSION = "identity-migration-preview-v1"

_SQLITE_MAGIC = b"SQLite format 3\x00"
_SHM_DMS_OFFSET = 128
_SHM_WRITE_OFFSET = 120
_PAGE_DIRS = ("concepts", "entries", "mocs")
_NON_SOURCE_FILENAMES = frozenset({"failed_urls.md", "index.md"})
_MAX_REGISTRY_BYTES = 2**31


class PreviewRefusedError(Exception):
    """The preview cannot establish a consistent, write-free view."""

    def __init__(self, reasons: list[str], action: str) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons
        self.action = action


@dataclass
class _Layout:
    vault: Path
    wiki: Path
    sources: Path
    llmwiki: Path
    db: Path
    immutable_manifest: dict | None = None


@dataclass
class _Inventory:
    blockers: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def block(self, kind: str, path: str, detail: str) -> None:
        self.blockers.append({"kind": kind, "path": path, "detail": detail})


# ── filesystem observation (no follow, no writes) ────────────────────────


def _tree_stat(root: Path) -> dict[str, tuple[int, int, int, int]]:
    """Entry → (type, size, mtime_ns, inode) for the whole tree; symlinks unfollowed."""
    result: dict[str, tuple[int, int, int, int]] = {}
    if not root.is_dir():
        return result
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(dirs + files):
            path = Path(directory) / name
            try:
                info = path.lstat()
            except OSError:
                continue
            result[path.relative_to(root).as_posix()] = (
                stat.S_IFMT(info.st_mode), info.st_size, info.st_mtime_ns, info.st_ino,
            )
    return result


def _symlinks_in(root: Path) -> list[str]:
    links: list[str] = []
    if root.is_symlink():
        return [root.as_posix()]
    if not root.is_dir():
        return links
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                links.append(path.relative_to(root).as_posix())
    return sorted(links)


def _stat_or_none(path: Path) -> dict[str, Any] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return {
        "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "inode": info.st_ino,
        "symlink": stat.S_ISLNK(info.st_mode),
    }


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _posix_lock_holder(path: Path, offset: int, length: int) -> dict[str, Any] | None:
    """Who holds a POSIX record lock on ``path[offset:offset+length]``? (None = nobody)

    ``F_GETLK`` is a pure query: it never takes a lock and never modifies the
    file. SQLite's unix VFS uses exactly these byte-range locks on ``-shm``.
    """
    if os.name == "nt":
        raise OSError("record-lock inspection is not available on this platform")
    import fcntl

    fd = os.open(path, os.O_RDONLY)
    try:
        request = struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, offset, length, 0)
        kind, _whence, start, span, pid = struct.unpack(
            "hhqqi", fcntl.fcntl(fd, fcntl.F_GETLK, request)
        )
    finally:
        os.close(fd)
    if kind == fcntl.F_UNLCK:
        return None
    return {
        "type": "read" if kind == fcntl.F_RDLCK else "write", "start": start, "length": span,
        "pid": pid,
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ── registry snapshot ───────────────────────────────────────────────────


def _registry_state(layout: _Layout) -> tuple[dict[str, Any], list[str]]:
    """Observe the registry family without opening it; return (facts, refusal reasons)."""
    db = layout.db
    sidecars = {
        "wal": _stat_or_none(db.with_name(db.name + "-wal")),
        "shm": _stat_or_none(db.with_name(db.name + "-shm")),
        "journal": _stat_or_none(db.with_name(db.name + "-journal")),
    }
    facts: dict[str, Any] = {
        "path": db.as_posix(), "main": _stat_or_none(db), "sidecars": sidecars,
        "locks": {}, "compile_lock": None,
    }
    reasons: list[str] = []
    if facts["main"] is None:
        facts["strategy"] = "absent"
        return facts, reasons
    if sidecars["journal"] is not None:
        reasons.append(
            "a hot rollback journal (registry.db-journal) exists: a transaction may be "
            "in flight or was interrupted"
        )
    if sidecars["shm"] is not None:
        shm = db.with_name(db.name + "-shm")
        try:
            dms = _posix_lock_holder(shm, _SHM_DMS_OFFSET, 1)
            writer = _posix_lock_holder(shm, _SHM_WRITE_OFFSET, 8)
        except OSError as exc:
            reasons.append(
                "registry.db-shm exists but its lock state cannot be inspected on this "
                f"platform ({exc}); a live connection cannot be ruled out"
            )
        else:
            facts["locks"] = {"dms": dms, "write": writer}
            if dms is not None:
                holder = f" (pid {dms['pid']})" if dms.get("pid") else ""
                reasons.append(
                    f"a live SQLite connection holds registry.db-shm{holder}: the main "
                    "file alone is not a consistent snapshot"
                )
    wal = sidecars["wal"]
    if wal is not None and wal["bytes"] > 0:
        reasons.append(
            f"registry.db-wal holds {wal['bytes']} bytes of frames not checkpointed into "
            "registry.db; committed rows may live only in the WAL, and an immutable/"
            "in-memory read of the main file would silently miss them"
        )
    lock_file = layout.llmwiki / "lock"
    if lock_file.is_file():
        try:
            pid = int(lock_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            facts["compile_lock"] = {"pid": None, "alive": None}
        else:
            alive = _pid_alive(pid)
            facts["compile_lock"] = {"pid": pid, "alive": alive}
            if alive:
                reasons.append(f"the compile lock names a live process (pid {pid})")
    facts["strategy"] = "in-memory-deserialize"
    return facts, reasons


def _load_registry_copy(db: Path, facts: dict[str, Any]) -> tuple[sqlite3.Connection, str]:
    """Deserialize ``registry.db`` bytes into memory; return (connection, sha256)."""
    size = facts["main"]["bytes"]
    if size > _MAX_REGISTRY_BYTES:
        raise PreviewRefusedError(
            [f"registry.db is {size} bytes; the in-memory preview is bounded at "
             f"{_MAX_REGISTRY_BYTES}"],
            "preview a disposable copy with a smaller registry or extend the strategy",
        )
    raw = db.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not raw.startswith(_SQLITE_MAGIC):
        raise PreviewRefusedError(
            ["registry.db is not an SQLite database (bad header)"],
            "restore registry.db from a verified snapshot before previewing",
        )
    buffer = bytearray(raw)
    if buffer[18] == 2 or buffer[19] == 2:
        # In-memory copy only: with no WAL frames the main file is complete,
        # and deserialize() rejects a WAL-mode header outright.
        buffer[18] = buffer[19] = 1
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(bytes(buffer))
        check = connection.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError as exc:
        connection.close()
        raise PreviewRefusedError(
            [f"registry.db could not be loaded as SQLite: {exc}"],
            "restore registry.db from a verified snapshot before previewing",
        ) from exc
    if check is None or check[0] != "ok":
        connection.close()
        raise PreviewRefusedError(
            [f"registry.db failed quick_check: {check[0] if check else 'no result'}"],
            "restore registry.db from a verified snapshot before previewing",
        )
    connection.row_factory = sqlite3.Row
    return connection, digest


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _schema_version(connection: sqlite3.Connection, inventory: _Inventory) -> int | None:
    from obsidian_llm_wiki.core.immutable_publication import (
        registry_format_version,
        validate_registry_format,
    )

    try:
        version = registry_format_version(connection)
    except RuntimeError as exc:
        inventory.block("unknown_schema_version", "registry.db", str(exc))
        return None
    if version == _SCHEMA_VERSION:
        try:
            validate_registry_format(connection)
        except RuntimeError as exc:
            inventory.block("invalid_registry_shape", "registry.db", str(exc))
    if version > _SCHEMA_VERSION:
        inventory.block(
            "unknown_schema_version", "registry.db",
            f"schema {version} is newer than this binary ({_SCHEMA_VERSION}); table "
            "meanings are unknown",
        )
    elif version == 6:
        inventory.notes.append("schema 6 is a legacy inventory, not a committed schema-7 plan")
    elif version < _SCHEMA_VERSION:
        inventory.block(
            "legacy_schema_version", "registry.db",
            f"schema {version} < {_SCHEMA_VERSION}: a normal open would migrate it; the "
            "inventory below reads the tables as they are",
        )
    return version


# ── source / cache / page observation ───────────────────────────────────


def _source_bodies(layout: _Layout, inventory: _Inventory) -> dict[str, dict[str, Any]]:
    bodies: dict[str, dict[str, Any]] = {}
    if not layout.sources.is_dir():
        return bodies
    for path in sorted(layout.sources.iterdir()):
        if path.suffix != ".md" or not path.is_file() or path.name in _NON_SOURCE_FILENAMES:
            continue
        try:
            raw = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            inventory.block("unreadable_source", f"sources/{path.name}", str(exc))
            continue
        _meta, body = parse_frontmatter(raw)
        content = body.strip()
        bodies[path.name] = {
            "body_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "file_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "bytes": path.stat().st_size,
        }
    return bodies


def _cache_artifacts(layout: _Layout, inventory: _Inventory) -> dict[str, dict[str, Any]]:
    cache_dir = layout.llmwiki / "cache"
    artifacts: dict[str, dict[str, Any]] = {}
    if not cache_dir.is_dir():
        return artifacts
    for path in sorted(cache_dir.glob("*.json")):
        rel = f".llmwiki/cache/{path.name}"
        if path.name == "_resynthesis_overlay.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            inventory.block("malformed_cache", rel, f"{type(exc).__name__}: {exc}")
            continue
        if not isinstance(data, dict):
            inventory.block("malformed_cache", rel, "top-level value is not an object")
            continue
        artifact = data.get("_artifact")
        payload = data.get("payload")
        if isinstance(artifact, dict) and isinstance(payload, dict):
            header = {
                "kind": artifact.get("kind"), "schema_version": artifact.get("schema_version"),
            }
            raw_versions = artifact.get("versions")
            versions = raw_versions if isinstance(raw_versions, dict) else {}
        else:
            header, payload, versions = {"kind": None, "schema_version": None}, data, {}
        source_file = payload.get("source_file") if isinstance(payload, dict) else None
        key = source_file if isinstance(source_file, str) and source_file else path.stem
        artifacts[key] = {
            "path": rel, "artifact": header, "versions": versions, "payload": payload,
            "sha256": _sha256_file(path),
        }
    return artifacts


def _pages(layout: _Layout, inventory: _Inventory) -> list[dict[str, Any]]:
    from obsidian_llm_wiki.core.render_integrity import load_render_hashes, rendered_page_problem

    hashes = load_render_hashes(layout.wiki)
    pages: list[dict[str, Any]] = []
    # Legacy source inputs are human files, not generated render projections.
    directories = (*_PAGE_DIRS, "sources") if layout.immutable_manifest else _PAGE_DIRS
    for subdir in directories:
        directory = layout.wiki / subdir
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.md")):
            if not path.is_file():
                continue
            rel = path.relative_to(layout.wiki).as_posix()
            try:
                raw_bytes = path.read_bytes()
                raw = raw_bytes.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                inventory.block("unreadable_page", rel, str(exc))
                continue
            meta, body = parse_frontmatter(raw)
            digest = hashlib.sha256(raw_bytes).hexdigest()
            render_problem = rendered_page_problem(rel, digest, hashes)
            if layout.immutable_manifest and render_problem:
                inventory.block("rendered_page_unverified", rel, render_problem)
            if meta.get("reviewed") is True:
                basis = "reviewed: true"
            elif rel not in hashes:
                basis = "no render hash (unknown ownership)"
            elif hashes[rel] != digest:
                basis = "render hash mismatch (edited after generation)"
            else:
                basis = ""
            pages.append({
                "page": rel, "stem": path.stem, "sha256": digest, "human_edited": bool(basis),
                "basis": basis, "links": extract_wikilinks(body),
                "render_problem": render_problem,
            })
    return pages


# ── inventory ───────────────────────────────────────────────────────────


def _payload_concepts(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    return [c for c in payload.get("concepts", []) if isinstance(c, dict)]


def _receipt_facts(receipt: Any, current_body_sha: str | None) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        return {
            "receipt_version": None, "receipt_status": None, "receipt_source_sha256": None,
            "source_revision_matches_receipt": None, "identity_record_version": None,
        }
    block = receipt.get("identity")
    return {
        "receipt_version": receipt.get("version"),
        "receipt_status": receipt.get("status"),
        "receipt_source_sha256": receipt.get("source_sha256"),
        "source_revision_matches_receipt": (
            None if current_body_sha is None
            else receipt.get("source_sha256") == current_body_sha
        ),
        "identity_record_version": block.get("version") if isinstance(block, dict) else None,
    }


def _claim_facts(concept: dict[str, Any], current_body_sha: str | None) -> dict[str, Any]:
    claims = [c for c in concept.get("claims", []) if isinstance(c, dict)]
    spans = quotes = hash_matches = hashed = 0
    for claim in claims:
        evidence = claim.get("evidence")
        if not isinstance(evidence, dict):
            if isinstance(claim.get("quote"), str) and claim["quote"]:
                quotes += 1
            continue
        fragments = evidence.get("fragments", [evidence])
        # Model conversion/receipt admission enforces the versioned shape; this
        # preview summary counts every separate anchor rather than skipping them.
        if not isinstance(fragments, list):
            continue
        for evidence in fragments:
            if not isinstance(evidence, dict):
                continue
            if evidence.get("span_ids"):
                spans += 1
            elif evidence.get("quote"):
                quotes += 1
            source_hash = evidence.get("source_hash")
            if isinstance(source_hash, str) and source_hash:
                hashed += 1
                if source_hash == current_body_sha:
                    hash_matches += 1
    return {
        "claims": len(claims), "catalog_span_citations": spans, "quote_citations": quotes,
        "cited_source_hash_matches_current": (
            None if hashed == 0 else hash_matches == hashed
        ),
    }


def _classify(
    *, owners: list[str], canonical_id: str, source_on_disk: bool, revision_match: bool | None,
    registry_hash_match: bool | None, receipt: dict[str, Any], cache_match: bool | None,
) -> tuple[str, str]:
    if not canonical_id:
        return "unresolved", "legacy payload has no canonical_id; identity cannot be carried"
    if len(owners) > 1:
        return "needs_assessment", (
            f"entity is co-owned by {len(owners)} sources ({', '.join(owners)}); a new plan "
            "must re-assess each owner bilaterally against its historical referent. Legacy "
            "assessment records are historical evidence only, not a verdict under the new "
            "plan"
        )
    if not source_on_disk:
        return "unresolved", (
            "source file is missing from sources/; the reviewed revision cannot be re-read"
        )
    if revision_match is False or registry_hash_match is False:
        return "unresolved", (
            "source revision on disk differs from the reviewed/registered revision; the "
            "legacy binding and its citations are historical and must be re-reviewed"
        )
    if receipt.get("receipt_version") != _REVIEW_RECEIPT_VERSION:
        return "unresolved", (
            f"review receipt version {receipt.get('receipt_version')!r} is not the current "
            f"{_REVIEW_RECEIPT_VERSION!r}; no current approved review backs the binding"
        )
    if receipt.get("receipt_status") != "approved":
        return "unresolved", (
            f"review receipt status is {receipt.get('receipt_status')!r}, not approved"
        )
    if cache_match is False:
        return "unresolved", "synthesis cache payload disagrees with the registry payload"
    return "resolved", (
        "sole legacy owner with current approved review; bookkeeping continuity only, "
        "not authorization to reuse identity in a new publication plan"
    )


def _inventory_bindings(
    connection: sqlite3.Connection, bodies: dict[str, dict[str, Any]],
    caches: dict[str, dict[str, Any]], inventory: _Inventory,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tables = _tables(connection)
    required = {"source", "synthesis", "concept", "concept_source"}
    missing = sorted(required - tables)
    if missing:
        inventory.block(
            "missing_tables", "registry.db", f"cannot inventory bindings without {missing}",
        )
        return [], {"registered": [], "missing_on_disk": [], "unregistered": sorted(bodies)}
    sources = {
        row["id"]: dict(row) for row in connection.execute(
            "SELECT id, filename, content_hash, status, created_at FROM source "
            "ORDER BY created_at, filename"
        )
    }
    owners_by_concept: dict[str, list[str]] = {}
    for row in connection.execute(
        "SELECT cs.concept_id, s.filename FROM concept_source cs JOIN source s ON s.id="
        "cs.source_id ORDER BY s.created_at, s.filename"
    ):
        owners_by_concept.setdefault(str(row[0]), []).append(str(row[1]))
    concept_rows = {
        str(row["id"]): dict(row)
        for row in connection.execute("SELECT id, slug, title, summary FROM concept")
    }
    bindings: list[dict[str, Any]] = []
    for row in connection.execute(
        "SELECT source_id, payload_json, input_hash, prompt_version, schema_version, "
        "model_id, status FROM synthesis"
    ):
        source = sources.get(row["source_id"])
        if source is None:
            inventory.block(
                "orphan_synthesis", "registry.db", f"synthesis for unknown source {row[0]}",
            )
            continue
        filename = str(source["filename"])
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError) as exc:
            inventory.block("malformed_registry_payload", f"synthesis:{filename}", str(exc))
            continue
        if not isinstance(payload, dict):
            inventory.block(
                "malformed_registry_payload", f"synthesis:{filename}", "not an object",
            )
            continue
        body = bodies.get(filename)
        current_sha = body["body_sha256"] if body else None
        receipt = payload.get("review_receipt")
        receipt_facts = _receipt_facts(receipt, current_sha)
        records = assessment_records(receipt)
        if isinstance(receipt, dict) and isinstance(receipt.get("identity"), dict):
            legacy = receipt["identity"]
            if legacy.get("version") == "identity-assessment-v1":
                # Inventory only: legacy unilateral evidence is never merge authority.
                records = [r for r in legacy.get("assessments", []) if isinstance(r, dict)
                           and all(isinstance(r.get(k), str) for k in
                                   ("candidate_id", "subject_sha256", "verdict"))]
        cache = caches.get(filename)
        cache_match = None if cache is None else cache["payload"] == payload
        for concept in _payload_concepts(payload):
            slug = str(concept.get("slug") or "")
            canonical_id = str(concept.get("canonical_id") or "")
            effective = canonical_id or VaultRegistry._stable_id("concept", slug)
            owners = owners_by_concept.get(effective, [])
            own_sha = definition_snapshot(concept.get("title"), concept.get("summary"))
            legacy_assessments = [
                r for r in records
                if r["candidate_id"] == effective and r["subject_sha256"] == own_sha
            ]
            status, reason = _classify(
                owners=owners, canonical_id=canonical_id, source_on_disk=body is not None,
                revision_match=receipt_facts["source_revision_matches_receipt"],
                registry_hash_match=(
                    None if current_sha is None else str(row["input_hash"]) == current_sha
                ),
                receipt=receipt_facts, cache_match=cache_match,
            )
            if effective not in concept_rows:
                status, reason = "unresolved", (
                    f"payload names entity {effective} but no concept row exists"
                )
            bindings.append({
                "source_file": filename, "slug": slug, "canonical_id": effective,
                "title": concept.get("title"), "status": status, "reason": reason,
                "owners": owners, "incumbent": owners[0] if owners else None,
                "definition_sha256": own_sha,
                "evidence": {
                    **receipt_facts,
                    "legacy_assessments": len(legacy_assessments),
                    "legacy_assessment_verdicts": sorted(
                        {str(r["verdict"]) for r in legacy_assessments}
                    ),
                    "registry_input_hash_matches_current": (
                        None if current_sha is None else str(row["input_hash"]) == current_sha
                    ),
                    **_claim_facts(concept, current_sha),
                    "cache_matches_registry": cache_match,
                },
                "provenance": {
                    "registry_source_status": source["status"],
                    "registry_source_created_at": source["created_at"],
                    "registry_synthesis_status": row["status"],
                    "registry_versions": {
                        "prompt": row["prompt_version"], "schema": row["schema_version"],
                        "model": row["model_id"],
                    },
                    "cache_artifact": None if cache is None else cache["artifact"],
                    "cache_versions": None if cache is None else cache["versions"],
                    "cache_path": None if cache is None else cache["path"],
                    "current_source_body_sha256": current_sha,
                },
                "missing_context": [
                    item for item, absent in (
                        ("review receipt", receipt_facts["receipt_version"] is None),
                        ("identity assessment records",
                         receipt_facts["identity_record_version"] != IDENTITY_RECORD_VERSION),
                        ("synthesis cache", cache is None),
                        ("source file", body is None),
                    ) if absent
                ],
            })
    registered = sorted(str(s["filename"]) for s in sources.values())
    for name in caches:
        if name not in set(registered):
            inventory.block("orphan_cache", caches[name]["path"], "cache without registry row")
    return bindings, {
        "registered": registered,
        "missing_on_disk": sorted(set(registered) - set(bodies)),
        "unregistered": sorted(set(bodies) - set(registered)),
    }


def _references(pages: list[dict[str, Any]], slugs: set[str]) -> list[dict[str, Any]]:
    stems = {page["stem"] for page in pages} | slugs
    references: list[dict[str, Any]] = []
    for page in pages:
        for target, alias in page["links"]:
            normalized = target.split("#", 1)[0].strip().rsplit("/", 1)[-1]
            normalized = normalized.removesuffix(".md").strip()
            resolved = bool(normalized) and normalized in stems
            references.append({
                "page": page["page"], "target": target, "display": alias,
                "normalized": normalized, "status": "resolved" if resolved else "unresolved",
                "page_human_edited": page["human_edited"],
                "note": "" if resolved else "no page or registry slug with this name; keep text",
            })
    return references


# ── committed publication observation (pure reads only) ─────────────────


def _publication_inventory(layout, connection, inventory):
    """Compare the archived contract and every marker checkpoint to this byte copy.

    The lightweight adapters expose only reads. Never construct VaultRegistry,
    Config, a planner, or a recovery transaction while previewing.
    """
    from obsidian_llm_wiki.core.identity_plan import (
        PLAN_VERSION,
        canonical,
        checkpoint_files,
        digest,
        output_snapshot,
        registry_generation,
    )

    path = layout.llmwiki / "identity-publication.json"
    stage = layout.llmwiki / "publication-stage"
    facts = {"status": "absent", "verified": False, "plan_id": None,
             "recovery_required": stage.exists(), "journal_status": None,
             "verification_scope": "historical committed container and policy binding; "
                                   "admission/review-policy checks are per binding",
             "current_policy_verified": False}
    if stage.exists():
        try:
            journal = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
            facts["journal_status"] = journal.get("status", "unknown")
        except (OSError, ValueError, AttributeError):
            facts["journal_status"] = "interrupted-or-corrupt"
        inventory.block(
            "pending_publication_recovery", "publication-stage",
            "journal remains; preview never restores, cleans up or certifies recovery",
        )
    payloads = {}
    if not path.exists():
        inventory.block("publication_absent", path.name, "no committed publication marker")
        return facts, payloads
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        facts.update(status=marker["status"], plan_id=marker["plan_id"])
        if marker["status"] == "pending":
            facts["recovery_required"] = True
            inventory.block("pending_publication_recovery", path.name,
                            "pending marker; recover through the writer, never preview")
        if marker["version"] != PLAN_VERSION:
            raise ValueError("unknown publication plan version")
        plan_id = marker["plan_id"]
        if not isinstance(plan_id, str) or len(plan_id) != 64 or any(
            ch not in "0123456789abcdef" for ch in plan_id
        ):
            raise ValueError("invalid publication plan ID")
        archive = layout.llmwiki / "publications" / (plan_id + ".json")
        plan = json.loads(archive.read_text(encoding="utf-8"))
        if plan["version"] != PLAN_VERSION or digest(plan) != plan_id:
            raise ValueError("unknown or corrupt archived publication plan")
        payloads = {name: json.loads(raw) for name, raw in plan["sources"]}
        registry = SimpleNamespace(
            connection=connection,
            source_filenames=lambda: [r[0] for r in connection.execute(
                "SELECT filename FROM source ORDER BY filename")],
        )
        config = SimpleNamespace(
            vault=layout.vault, wiki_dir=layout.wiki, sources_dir=layout.sources,
            llmwiki_dir=layout.llmwiki, concepts_dir=layout.wiki / "concepts",
            state_file=layout.llmwiki / "state.json",
        )
        if marker["generation"] != registry_generation(registry):
            raise ValueError("registry/review/context generation drift")
        if marker["policy"] != plan["policy"]:
            raise ValueError("publication policy mismatch")
        if layout.immutable_manifest is None:
            if marker["source_files"] != list(map(list, checkpoint_files(registry, config))):
                raise ValueError("source or policy file drift")
            if marker["output_files"] != list(map(list, output_snapshot(config))):
                raise ValueError("generated output/cache drift")
        else:
            # Archived stage paths are historical evidence, not live locations.
            # The immutable envelope verifies actual installed bytes/consumers;
            # current human inputs still must match before certifying continuity.
            from obsidian_llm_wiki.core.immutable_publication import (
                _validate_publication_inputs,
                validate_generation,
            )
            from obsidian_llm_wiki.core.publication_inputs import input_freshness

            validate_generation(layout.wiki)
            _validate_publication_inputs(layout.immutable_manifest, marker)
            freshness = input_freshness(layout.vault, layout.immutable_manifest)
            facts["input_freshness"] = freshness
            if freshness["status"] != "current":
                raise ValueError("; ".join(freshness["reasons"]))
        if marker["source_payloads"] != {name: digest(p) for name, p in payloads.items()}:
            raise ValueError("publication source payload digest mismatch")
        assignments = {name: [c["canonical_id"] for c in p["concepts"]]
                       for name, p in payloads.items()}
        if marker["assignments"] != assignments:
            raise ValueError("publication assignment mismatch")
        rows = connection.execute(
            "SELECT s.filename,sy.payload_json FROM source s "
            "JOIN synthesis sy ON s.id=sy.source_id"
        )
        if {r[0]: canonical(json.loads(r[1])) for r in rows} != {
            name: canonical(p) for name, p in payloads.items()
        }:
            raise ValueError("archived plan differs from registry payloads")
        if marker["status"] != "committed" or marker.get("errors"):
            raise ValueError("publication is not committed without errors")
        facts["verified"] = not facts["recovery_required"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            RuntimeError, sqlite3.DatabaseError) as exc:
        inventory.block("publication_unverified", path.name, str(exc))
    return facts, payloads


def _committed_bindings(connection, bindings, payloads, publication, inventory, layout):
    from obsidian_llm_wiki.config import load_review_policy
    from obsidian_llm_wiki.core.identity_plan import digest
    from obsidian_llm_wiki.core.models import source_synthesis_from_dict
    from obsidian_llm_wiki.ingest.sources import source_body_for_receipt
    from obsidian_llm_wiki.synth.review import (
        verify_review_receipt,
        verify_review_receipt_integrity,
    )

    # Resolve policy from the same independent input layers as ordinary CLI
    # configuration, never from the recorded fingerprint. No Config/provider is
    # constructed, and only review-policy compatibility is claimed here.
    review_config = None
    config_problem = ""
    try:
        env_path = checked_generated_path(layout.vault / ".env", layout.vault)
        review_config = load_review_policy(
            str(env_path) if env_path.exists() else None, VAULT_PATH=str(layout.vault),
        )
        if review_config.grounding_review != "required":
            config_problem = "current review policy does not require admission review"
    except (OSError, ValueError, TypeError) as exc:
        config_problem = f"current review policy unavailable: {exc}"

    tables = _tables(connection)
    contexts = {
        (row[0], row[1]): row[2] for row in connection.execute(
            "SELECT filename,revision,content FROM identity_source_context")
    } if "identity_source_context" in tables else {}
    versions = dict(connection.execute("SELECT filename,extractor_version FROM source"))
    for binding in bindings:
        name, entity = binding["source_file"], binding["canonical_id"]
        revision = binding["evidence"]["receipt_source_sha256"]
        content = contexts.get((name, revision))
        has_context = isinstance(content, str) and (
            hashlib.sha256(content.encode()).hexdigest() == revision)
        binding["evidence"]["historical_source_context"] = has_context
        interpretation_problem = False
        if layout.immutable_manifest is not None:
            recorded_version = layout.immutable_manifest.get("compiler_versions", {}).get(
                "artifacts", {}).get("extractor", "")
            version = recorded_version if versions.get(name) == recorded_version else ""
            captured = checked_generated_path(layout.llmwiki / "source-inputs" / name, layout.wiki)
            try:
                captured_body, interpretation = source_body_for_receipt(
                    captured.read_bytes().decode("utf-8"), revision,
                    extractor_version=version, historical_content=content,
                )
            except (OSError, UnicodeDecodeError):
                captured_body, interpretation = None, "captured-source-unavailable"
            binding["evidence"]["source_interpretation"] = interpretation
            binding["evidence"]["source_interpretation_version"] = version
            interpretation_problem = captured_body is None or (
                interpretation == "historical-universal-newlines")
            if interpretation_problem:
                inventory.block(
                    "historical_source_interpretation" if captured_body is not None
                    else "source_interpretation_unverified", name,
                    f"{interpretation}: not current exact-source interpretation; rebuild required",
                )
        payload = payloads.get(name, {})
        receipt = payload.get("review_receipt") or {}
        accepted = receipt.get("publication_binding", {}).get(entity, {})
        evidence_problem = _evidence_problem(receipt, accepted, entity, contexts)
        if evidence_problem:
            inventory.block("unsupported_identity_evidence", name, evidence_problem)
        # Historical content must match the reviewed revision. Do not substitute
        # the current body, the rendered source page, or a caller-supplied hash.
        admission_problem = "reviewed historical source revision unavailable"
        policy_problem = config_problem
        if has_context:
            synthesis = source_synthesis_from_dict(payload)
            admission_problem = verify_review_receipt_integrity(synthesis, content, name)
            if not admission_problem and not policy_problem:
                policy_problem = verify_review_receipt(synthesis, content, name, review_config)
        binding["evidence"]["historical_admission_integrity"] = not admission_problem
        binding["evidence"]["current_review_policy_compatible"] = (
            not policy_problem if not admission_problem else None)
        if admission_problem:
            inventory.block("invalid_admission_receipt", name, admission_problem)
        if policy_problem:
            inventory.block("review_policy_unverified", name, policy_problem)
        binding["provenance"]["publication_plan_id"] = publication["plan_id"]
        binding["provenance"]["publication_basis"] = accepted.get("basis")
        binding["provenance"]["source_payload_sha256"] = digest(payload)
        if not has_context:
            binding["missing_context"].append("historical source revision")
            inventory.block("missing_historical_context", name, "reviewed revision unavailable")
        if publication["verified"] and has_context and not (
            evidence_problem or admission_problem or policy_problem or interpretation_problem
        ) and (
            accepted.get("subject_sha256") == binding["definition_sha256"]
            and accepted.get("source_sha256") == revision
            and receipt.get("version") == _REVIEW_RECEIPT_VERSION
            and receipt.get("status") == "approved"
            and binding["evidence"]["source_revision_matches_receipt"] is True
            and binding["evidence"]["cache_matches_registry"] is True
        ):
            binding["status"] = "resolved"
            binding["reason"] = (
                "effective historical binding with verified admission and compatible "
                "current review policy (not full publication-policy revalidation or apply)")
        else:
            binding["status"] = "unresolved"
            binding["reason"] = (
                "committed publication, admission, current review policy or historical "
                "context unverified")


def _evidence_problem(receipt, binding, entity, historical):
    """Unknown versions are not equivalent to absent/valid evidence."""
    from obsidian_llm_wiki.core.identity_context import CONTEXT_VERSION

    if receipt.get("version") not in KNOWN_REVIEW_RECEIPT_VERSIONS:
        return "unknown or absent review receipt version"
    if receipt.get("status") != "approved":
        return "review not approved"
    if binding.get("basis") not in {
        "new entity", "assessed same entity", "unchanged revision binding", "own reservation",
    }:
        return "unknown or absent publication binding basis"
    identity = receipt.get("identity")
    if identity is not None and (
        not isinstance(identity, dict) or identity.get("version") != IDENTITY_RECORD_VERSION
    ):
        return "unknown identity assessment version"
    if binding.get("basis") == "assessed same entity":
        from obsidian_llm_wiki.core.entity_identity import find_assessment

        if find_assessment(receipt, subject_sha=binding.get("subject_sha256"),
                           candidate_id=entity) is None:
            return "bilateral same proof missing, incomplete or invalid"

    def contexts(value):
        if isinstance(value, dict):
            if value.get("version") == CONTEXT_VERSION:
                revision = value.get("source_sha256")
                content = historical.get((value.get("source_file"), revision))
                if not isinstance(content, str) or hashlib.sha256(
                    content.encode()).hexdigest() != revision:
                    return "historical bilateral source revision unavailable"
            for key, child in value.items():
                if key == "subject_context" and (
                    not isinstance(child, dict) or child.get("version") != CONTEXT_VERSION
                ):
                    return "unknown bilateral context version"
                # v1 candidate aggregates have no version; their owner contexts do.
                if key == "candidate_context" and (
                    not isinstance(child, dict) or "version" in child
                ):
                    return "unknown candidate aggregate context version"
                if key == "owners" and isinstance(child, list) and any(
                    not isinstance(owner, dict) or owner.get("version") != CONTEXT_VERSION
                    for owner in child
                ):
                    return "unknown historical owner context version"
                problem = contexts(child)
                if problem:
                    return problem
        elif isinstance(value, list):
            for child in value:
                problem = contexts(child)
                if problem:
                    return problem
        return None

    return contexts(receipt)


def _identity_references(payloads, bindings, publication, inventory):
    """Per-occurrence plan records, kept separate from name-only page links."""
    from obsidian_llm_wiki.core.entity_identity import find_assessment
    from obsidian_llm_wiki.core.identity_plan import digest

    result = []
    for name, payload in sorted(payloads.items()):
        receipt = payload.get("review_receipt") or {}
        local = {c["canonical_id"] for c in payload.get("concepts", [])}
        source_bindings = [b for b in bindings if b["source_file"] == name]
        source_ok = publication["verified"] and all(
            b["status"] == "resolved" for b in source_bindings)
        review_hash = digest({k: v for k, v in receipt.items()
                              if k not in {"identity", "publication_binding", "references"}})
        for key, record in sorted(receipt.get("references", {}).items()):
            target = record.get("target_id", "")
            proof_ok = False
            if record.get("basis") == "reviewed local occurrence":
                proof_ok = target in local
            elif record.get("proof"):
                proof_ok = find_assessment(
                    {**receipt, "identity": {"version": IDENTITY_RECORD_VERSION,
                                              "assessments": [record["proof"]]}},
                    subject_sha=definition_snapshot(record.get("original_text"), key),
                    candidate_id=target,
                ) is not None
            resolved = (source_ok and record.get("status") == "bound" and proof_ok
                        and record.get("source_revision") == receipt.get("source_sha256")
                        and record.get("review_sha256") == review_hash)
            if record.get("status") == "bound" and not proof_ok:
                inventory.block("unproved_reference", f"{name}:{key}",
                                "bound occurrence has no valid local/bilateral proof")
            result.append({
                "source_file": name, "occurrence": key,
                "original_text": record.get("original_text"),
                "original_slug": record.get("original_slug"),
                "target_id": target, "recorded_status": record.get("status"),
                "status": "resolved" if resolved else "unresolved",
                "source_revision": record.get("source_revision"),
                "review_sha256": record.get("review_sha256"), "proof_available": proof_ok,
            })
    return result


def _cache_history(layout):
    return [{"path": p.relative_to(layout.wiki).as_posix(), "sha256": _sha256_file(p),
             "bytes": p.stat().st_size}
            for p in sorted((layout.llmwiki / "cache-history").glob("*.json")) if p.is_file()]


# ── entry point ─────────────────────────────────────────────────────────


def _layout(vault: Path) -> _Layout:
    wiki = checked_generated_path(vault / "04-Wiki", vault)
    return _Layout(
        vault=vault, wiki=wiki, sources=wiki / "sources", llmwiki=wiki / ".llmwiki",
        db=wiki / ".llmwiki" / "registry.db",
    )


def preview_identity_migration(vault_path: Path) -> dict[str, Any]:
    """Build the read-only preview report; raise :class:`PreviewRefusedError` to fail closed."""
    generated_at = datetime.now(UTC).isoformat()
    vault = Path(vault_path).expanduser().absolute()
    if vault.is_symlink():
        raise PreviewRefusedError(
            [f"vault path is a symlink: {vault}"],
            "preview the real directory; symlinked vaults need an explicit snapshot strategy",
        )
    # Same normalization as ``resolve_vault``: a spelled ``..`` component would
    # otherwise be rejected by the containment check as parent traversal.
    vault = vault.resolve()
    if not vault.is_dir():
        raise PreviewRefusedError(
            [f"vault directory does not exist: {vault}"],
            "check the path; the preview never creates a vault",
        )
    try:
        layout = _layout(vault)
    except ValueError as exc:
        raise PreviewRefusedError(
            [str(exc)], "remove the symlink/traversal before previewing",
        ) from exc
    links = _symlinks_in(layout.wiki)
    if links:
        raise PreviewRefusedError(
            [f"symlink inside 04-Wiki: {name}" for name in links[:20]],
            "symlinked entries are not owned vault files; snapshot them separately",
        )
    try:
        if layout.llmwiki.exists():
            checked_metadata_directory(layout.llmwiki, layout.wiki)
    except ValueError as exc:
        raise PreviewRefusedError([str(exc)], "remove the unsafe metadata entry") from exc

    observed_wiki = layout.wiki
    tree_before = _tree_stat(observed_wiki)
    from obsidian_llm_wiki.core.immutable_publication import _preview_publication

    try:
        committed = _preview_publication(layout.wiki)
        if committed:
            directory, manifest = committed
            layout = _Layout(vault, directory, layout.sources, directory / ".llmwiki",
                             directory / ".llmwiki/registry.db", manifest)
    except (OSError, ValueError, RuntimeError) as exc:
        raise PreviewRefusedError(
            [str(exc)], "inspect the committed generation; preview never repairs",
        ) from exc
    registry_facts, refusal = _registry_state(layout)
    if refusal:
        raise _busy(refusal, registry_facts)

    inventory = _Inventory()
    bindings: list[dict[str, Any]] = []
    schema_version: int | None = None
    publication = {"status": "absent", "verified": False, "plan_id": None}
    payloads = {}
    sources_report: dict[str, Any]
    publication_artifacts = (layout.llmwiki / "identity-publication.json").exists() or (
        layout.llmwiki / "publication-stage").exists()
    bodies = _source_bodies(layout, inventory)
    caches = _cache_artifacts(layout, inventory) if layout.llmwiki.is_dir() else {}
    slugs: set[str] = set()
    digest_before = digest_after = None
    if registry_facts["strategy"] == "absent":
        inventory.block(
            "registry_absent", layout.db.relative_to(layout.vault).as_posix(),
            "no registry.db: nothing has been compiled, or the vault layout differs",
        )
        sources_report = {"registered": [], "missing_on_disk": [], "unregistered": sorted(bodies)}
        if publication_artifacts:
            publication, payloads = _publication_inventory(layout, None, inventory)
    else:
        connection, digest_before = _load_registry_copy(layout.db, registry_facts)
        try:
            schema_version = _schema_version(connection, inventory)
            bindings, sources_report = _inventory_bindings(connection, bodies, caches, inventory)
            if schema_version == 7 or publication_artifacts:
                publication, payloads = _publication_inventory(layout, connection, inventory)
            if schema_version == 7:
                _committed_bindings(connection, bindings, payloads, publication, inventory, layout)
            elif schema_version != 6 or publication_artifacts:
                for binding in bindings:
                    binding.update(status="unresolved", reason=(
                        "registry schema/publication evidence unverified; "
                        "recovery may be required"))
            if "concept" in _tables(connection):
                slugs = {str(r[0]) for r in connection.execute("SELECT slug FROM concept")}
        except (sqlite3.DatabaseError, TypeError, KeyError, AttributeError, ValueError) as exc:
            raise PreviewRefusedError(
                [f"registry schema/payload cannot be inventoried: {exc}"],
                "inspect a disposable snapshot with the matching writer version; "
                "preview never repairs",
            ) from exc
        finally:
            connection.close()
    pages = _pages(layout, inventory)
    # Identity proof continuity is not certification of the consumed projection.
    # Keep its historical result separate; never repair or adopt altered bytes.
    by_page = {page["page"]: page for page in pages}
    for binding in bindings:
        relative = f"concepts/{binding['slug']}.md"
        page = by_page.get(relative)
        binding["evidence"]["rendered_page_certified"] = (
            not page["render_problem"] if page is not None and layout.immutable_manifest else False
        )
        if layout.immutable_manifest and page is None:
            inventory.block("rendered_page_unverified", relative, "Rendered page unavailable")
    references = _references(pages, slugs)
    identity_references = _identity_references(payloads, bindings, publication, inventory)
    cache_history = _cache_history(layout)

    # Consistency guard: the registry bytes, sidecar family and every vault
    # entry must be exactly what they were when the read started.
    inconsistent: list[str] = []
    if digest_before is not None:
        digest_after = _sha256_file(layout.db)
        if digest_after != digest_before:
            inconsistent.append("registry.db bytes changed during the preview")
    facts_after, _ = _registry_state(layout)
    if facts_after["sidecars"] != registry_facts["sidecars"]:
        inconsistent.append("registry sidecar family changed during the preview")
    tree_after = _tree_stat(observed_wiki)
    if tree_after != tree_before:
        changed = sorted(
            k for k in set(tree_before) | set(tree_after)
            if tree_before.get(k) != tree_after.get(k)
        )
        inconsistent.append(f"vault entries changed during the preview: {changed[:20]}")

    counts = {s: sum(1 for b in bindings if b["status"] == s)
              for s in ("resolved", "unresolved", "needs_assessment")}
    counts["total"] = len(bindings)
    needs = counts["needs_assessment"]
    report: dict[str, Any] = {
        "preview_version": PREVIEW_VERSION,
        "generated_at": generated_at,
        "mode": "dry-run",
        "status": "inconsistent" if inconsistent else "preview",
        "vault": vault.as_posix(),
        "vault_unchanged": not inconsistent,
        "inconsistencies": inconsistent,
        "apply": {"implemented": False, "handled_by": "future plan build"},
        "registry": {
            "strategy": registry_facts["strategy"],
            "path": registry_facts["path"],
            "bytes": None if registry_facts["main"] is None else registry_facts["main"]["bytes"],
            "sha256_before": digest_before, "sha256_after": digest_after,
            "sidecars": registry_facts["sidecars"],
            "locks": registry_facts["locks"],
            "compile_lock": registry_facts["compile_lock"],
            "schema_version": schema_version,
            "binary_schema_version": _SCHEMA_VERSION,
        },
        "sources": {"present": sorted(bodies), **sources_report},
        "bindings": bindings,
        "publication": publication,
        "rendered_pages": {
            "scope": "generated concepts, entries, maps and captured sources; "
                     "render inventory equality, not semantic review or atomic race immunity",
            "certified": bool(layout.immutable_manifest) and bool(pages) and not any(
                b["kind"] in {"rendered_page_unverified", "unreadable_page"}
                for b in inventory.blockers
            ),
            "unverified": [{"page": p["page"], "reason": p["render_problem"]}
                           for p in pages if p["render_problem"]],
        },
        "identity_references": identity_references,
        "cache_history": cache_history,
        "references": references,
        "human_edited_pages": [
            {"page": p["page"], "basis": p["basis"], "sha256": p["sha256"]}
            for p in pages if p["human_edited"]
        ],
        "blockers": inventory.blockers,
        "notes": inventory.notes,
        "summary": {
            "source_count": len(sources_report["registered"]),
            "claims": sum(b["evidence"]["claims"] for b in bindings),
            "catalog_span_citations": sum(
                b["evidence"]["catalog_span_citations"] for b in bindings),
            "quote_citations": sum(b["evidence"]["quote_citations"] for b in bindings),
            "bindings": counts,
            "references": {
                s: sum(1 for r in references if r["status"] == s)
                for s in ("resolved", "unresolved")
            },
            "human_edited_pages": sum(1 for p in pages if p["human_edited"]),
            "blockers": len(inventory.blockers),
        },
        "assessment_estimate": {
            "status": "unavailable",
            "lower": None, "upper": None,
            "needs_assessment_bindings": needs,
            "model_calls_made": 0,
            "method": (
                "This read-only inventory does not enumerate a prospective assessment request "
                "graph or provider retry policy. Binding counts are not call counts, even "
                "when an existing publication plan is verified; no bounded estimate is available."
            ),
        },
        "classification_policy": {
            "version": PREVIEW_VERSION,
            "resolved": "schema 7: verified historical committed plan, no stage, verified "
                        "admission integrity and independently compatible current review policy, "
                        "known binding/evidence and available historical contexts; "
                        "not full current publication-policy revalidation; schema 6: "
                        "legacy-only sole-owner bookkeeping continuity, not apply authority; "
                        "neither certifies rendered page bytes (see rendered_pages)",
            "unresolved": "missing canonical_id, source or concept row; revision drift; "
                          "stale/unapproved receipt; cache/registry disagreement",
            "needs_assessment": "legacy entity co-owned by more than one source",
        },
        "environment": {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version},
    }
    from obsidian_llm_wiki.core.immutable_publication import preview_publication_workflow
    try:
        report['publication_workflow'] = preview_publication_workflow(vault)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreviewRefusedError(
            [str(exc)], 'inspect the committed generation; preview never repairs',
        ) from exc
    return report


def _busy(reasons: list[str], facts: dict[str, Any]) -> PreviewRefusedError:
    exc = PreviewRefusedError(
        reasons,
        "Stop every olw process, scheduled job, desktop bridge and sync client for this "
        "vault, let the application close the registry normally (a clean close checkpoints "
        "and removes the WAL/SHM), then re-run the preview. Do not delete sidecars by hand "
        "and do not read with immutable=1: both discard committed WAL frames.",
    )
    exc.registry = facts  # type: ignore[attr-defined]
    return exc
