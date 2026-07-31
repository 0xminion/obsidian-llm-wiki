"""SQLite registry for scalable vault state, search, queues, and vectors.

The registry is the authoritative incremental index for v2 vaults. Markdown and
JSON cache files remain portable artifacts, but normal builds use indexed rows
rather than reconstructing state by scanning the filesystem.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path
from types import TracebackType
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from obsidian_llm_wiki.core.models import (
    SourceDoc,
    SourceSynthesis,
    concept_note_to_dict,
    moc_to_dict,
    source_synthesis_from_dict,
    source_synthesis_to_dict,
)

__all__ = [
    "ArtifactVersions",
    "AnnCandidate",
    "AnnSearchResult",
    "RenderDiff",
    "ConceptCandidate",
    "DirtyRecord",
    "IngestMatch",
    "VaultRegistry",
]

_SCHEMA_VERSION = 3
_LSH_TABLES = 8
_LSH_BITS = 12
_LSH_SPARSITY = 8
_COMPLETED_QUEUE_HISTORY = 10_000
_DIRTY_LEASE_SECONDS = 600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    normalized_url TEXT NOT NULL DEFAULT '',
    canonical_url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    extractor_version TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS source_content_hash_idx ON source(content_hash);
CREATE INDEX IF NOT EXISTS source_canonical_url_idx ON source(canonical_url);

CREATE TABLE IF NOT EXISTS source_version (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    extractor_version TEXT NOT NULL DEFAULT '',
    requested_url TEXT NOT NULL DEFAULT '',
    canonical_url TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(source_id, content_hash, extractor_version)
);

CREATE TABLE IF NOT EXISTS synthesis (
    source_id INTEGER PRIMARY KEY REFERENCES source(id) ON DELETE CASCADE,
    payload_json TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    parse_error_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS synthesis_versions_idx
    ON synthesis(prompt_version, schema_version, model_id);

CREATE TABLE IF NOT EXISTS concept (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS concept_slug_idx ON concept(slug);

CREATE TABLE IF NOT EXISTS concept_source (
    source_id INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concept(id) ON DELETE CASCADE,
    PRIMARY KEY(source_id, concept_id)
);
CREATE INDEX IF NOT EXISTS concept_source_concept_idx
    ON concept_source(concept_id);

CREATE TABLE IF NOT EXISTS concept_alias (
    alias TEXT PRIMARY KEY,
    concept_id TEXT NOT NULL REFERENCES concept(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS concept_alias_id_idx ON concept_alias(concept_id);

CREATE VIRTUAL TABLE IF NOT EXISTS concept_fts USING fts5(
    concept_id UNINDEXED,
    slug,
    title,
    aliases,
    tags,
    summary,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS moc (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS moc_slug_idx ON moc(slug);

CREATE TABLE IF NOT EXISTS moc_source (
    source_id INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    moc_id TEXT NOT NULL REFERENCES moc(id) ON DELETE CASCADE,
    PRIMARY KEY(source_id, moc_id)
);
CREATE INDEX IF NOT EXISTS moc_source_moc_idx ON moc_source(moc_id);

CREATE TABLE IF NOT EXISTS membership_source (
    source_id INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    moc_id TEXT NOT NULL REFERENCES moc(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concept(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(source_id, moc_id, concept_id)
);
CREATE INDEX IF NOT EXISTS membership_source_moc_idx
    ON membership_source(moc_id);

CREATE TABLE IF NOT EXISTS membership (
    moc_id TEXT NOT NULL REFERENCES moc(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concept(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(moc_id, concept_id)
);
CREATE INDEX IF NOT EXISTS membership_concept_idx ON membership(concept_id);

CREATE TABLE IF NOT EXISTS embedding (
    entity_kind TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    input_hash TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(entity_kind, entity_id, model_id)
);
CREATE INDEX IF NOT EXISTS embedding_lookup_idx
    ON embedding(entity_kind, model_id, dimensions);

CREATE TABLE IF NOT EXISTS ann_bucket (
    entity_kind TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    bucket INTEGER NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY(entity_kind, model_id, dimensions, table_id, bucket, entity_id)
);
CREATE INDEX IF NOT EXISTS ann_bucket_lookup_idx
    ON ann_bucket(entity_kind, model_id, dimensions, table_id, bucket);

CREATE TABLE IF NOT EXISTS dirty_queue (
    id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    claimed_by TEXT NOT NULL DEFAULT '',
    lease_expires_at TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS dirty_queue_claim_idx
    ON dirty_queue(state, next_attempt_at, id);
CREATE UNIQUE INDEX IF NOT EXISTS dirty_queue_pending_unique
    ON dirty_queue(entity_type, entity_id, reason)
    WHERE state IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS operation (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS operation_source_idx ON operation(source, updated_at);

CREATE TABLE IF NOT EXISTS ingest_hash (
    normalized_url TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    output_file TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ingest_hash_content_idx ON ingest_hash(content_hash);
CREATE INDEX IF NOT EXISTS ingest_hash_canonical_idx ON ingest_hash(canonical_url);

CREATE TABLE IF NOT EXISTS render_state (
    entity_kind TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(entity_kind, entity_key)
);
CREATE INDEX IF NOT EXISTS render_state_path_idx ON render_state(path);

CREATE TABLE IF NOT EXISTS health_finding (
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    reviewed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(path, kind, message)
);
CREATE INDEX IF NOT EXISTS health_finding_kind_idx ON health_finding(kind, severity);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _canonical_vector(vector: list[float]) -> list[float]:
    if not vector:
        raise ValueError("embedding vector must not be empty")
    result: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("embedding vector must contain finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding vector must contain finite numbers")
        result.append(number)
    return result


def _lsh_bucket(vector: list[float], model_id: str, table_id: int) -> int:
    """Compute one deterministic sparse random-hyperplane signature."""
    dimensions = len(vector)
    bucket = 0
    for bit in range(_LSH_BITS):
        projection = 0.0
        for sample in range(_LSH_SPARSITY):
            seed = f"{model_id}\x00{dimensions}\x00{table_id}\x00{bit}\x00{sample}".encode()
            digest = hashlib.blake2b(seed, digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            projection += sign * vector[index]
        if projection >= 0:
            bucket |= 1 << bit
    return bucket


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _normalize_url(url: str) -> str:
    """Canonicalize a public URL for exact indexed identity matching."""
    if not url.strip():
        return ""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.casefold()
    host = (parts.hostname or "").casefold()
    if not scheme or not host:
        return url.strip()
    port = parts.port
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)), doseq=True)
    return urlunsplit((scheme, host, parts.path or "/", query, ""))


@dataclass(frozen=True)
class ArtifactVersions:
    """Versions that determine whether persisted synthesis is reusable."""

    extractor: str
    prompt: str
    schema: str
    synthesis_model: str

    def as_dict(self) -> dict[str, str]:
        return {
            "extractor": self.extractor,
            "prompt": self.prompt,
            "schema": self.schema,
            "synthesis_model": self.synthesis_model,
        }


@dataclass(frozen=True)
class DirtyRecord:
    """One durable unit of invalidation or retry work."""

    id: int
    entity_type: str
    entity_id: str
    reason: str
    state: str
    attempts: int
    next_attempt_at: str
    payload: dict[str, object]
    error: str


@dataclass(frozen=True)
class IngestMatch:
    """Indexed duplicate match for an already persisted source."""

    normalized_url: str
    canonical_url: str
    content_hash: str
    output_file: str


@dataclass(frozen=True)
class ConceptCandidate:
    """One bounded lexical/semantic prompt candidate."""

    id: str
    slug: str
    title: str
    summary: str
    aliases: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class AnnCandidate:
    """One exact-reranked candidate from bounded ANN buckets."""

    entity_id: str
    score: float


@dataclass(frozen=True)
class AnnSearchResult:
    """ANN results plus the candidate-evaluation budget used."""

    candidates: list[AnnCandidate]
    evaluated: int
    bucket_hits: int


@dataclass(frozen=True)
class RenderDiff:
    changed_keys: frozenset[str]
    stale_paths: tuple[str, ...]
    stale_keys: frozenset[str]


class VaultRegistry:
    """Own the vault-local SQLite registry and its transactional schema."""

    def __init__(self, llmwiki_dir: str | Path) -> None:
        self.llmwiki_dir = Path(llmwiki_dir)
        self.llmwiki_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.llmwiki_dir / "registry.db"
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.executescript(_SCHEMA)
        dirty_columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(dirty_queue)")
        }
        if "lease_expires_at" not in dirty_columns:
            self.connection.execute(
                "ALTER TABLE dirty_queue "
                "ADD COLUMN lease_expires_at TEXT NOT NULL DEFAULT ''"
            )
        version_row = self.connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        try:
            previous_version = int(version_row[0]) if version_row is not None else 0
        except (TypeError, ValueError):
            previous_version = 0
        if previous_version < 3:
            self._backfill_entity_sources()
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(_SCHEMA_VERSION),),
        )
        self.connection.commit()

    def _backfill_entity_sources(self) -> None:
        """Build v3 source/entity ownership from existing synthesis payloads."""
        rows = self.connection.execute(
            "SELECT source_id, payload_json FROM synthesis"
        ).fetchall()
        with self.connection:
            self.connection.execute("DELETE FROM membership_source")
            self.connection.execute("DELETE FROM concept_source")
            self.connection.execute("DELETE FROM moc_source")
            self.connection.execute("DELETE FROM membership")
            for row in rows:
                try:
                    payload = json.loads(row[1])
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                synthesis = source_synthesis_from_dict(payload)
                source_id = int(row[0])
                for concept in synthesis.concepts:
                    concept_row = self.connection.execute(
                        "SELECT id FROM concept WHERE slug=?", (concept.slug,)
                    ).fetchone()
                    if concept_row is not None:
                        self.connection.execute(
                            "INSERT OR IGNORE INTO concept_source(source_id, concept_id) "
                            "VALUES(?, ?)",
                            (source_id, str(concept_row[0])),
                        )
                for moc in synthesis.maps:
                    moc_row = self.connection.execute(
                        "SELECT id FROM moc WHERE slug=?", (moc.slug,)
                    ).fetchone()
                    if moc_row is not None:
                        moc_id = str(moc_row[0])
                        self.connection.execute(
                            "INSERT OR IGNORE INTO moc_source(source_id, moc_id) VALUES(?, ?)",
                            (source_id, moc_id),
                        )
                        for position, slug in enumerate(moc.concept_slugs):
                            concept_row = self.connection.execute(
                                "SELECT id FROM concept WHERE slug=?", (slug,)
                            ).fetchone()
                            if concept_row is not None:
                                self.connection.execute(
                                    """
                                    INSERT OR IGNORE INTO membership_source(
                                        source_id, moc_id, concept_id, position
                                    ) VALUES(?, ?, ?, ?)
                                    """,
                                    (source_id, moc_id, str(concept_row[0]), position),
                                )
            moc_ids = {
                str(row[0])
                for row in self.connection.execute("SELECT id FROM moc")
            }
            self._rebuild_moc_memberships(moc_ids)
            self._cleanup_unreferenced_entities()

    def upsert_synthesis(
        self,
        filename: str,
        source: SourceDoc,
        synthesis: SourceSynthesis,
        versions: ArtifactVersions,
    ) -> None:
        """Atomically persist one source revision and its synthesis payload."""
        now = _now()
        input_hash = _content_hash(source.content)
        self._assign_canonical_ids(synthesis)
        requested_url = source.provenance.requested_url or source.url or ""
        canonical_url = (
            source.provenance.extracted_url
            or source.provenance.resolved_url
            or source.url
            or ""
        )
        synthesis.source_file = filename
        payload = json.dumps(
            source_synthesis_to_dict(synthesis),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        metadata = json.dumps(
            {
                "content_type": source.provenance.content_type,
                "diagnostics": list(source.provenance.diagnostics),
                "document_format": source.provenance.document_format,
                "extractor_chain": list(source.provenance.extractor_chain),
                "source_type": source.source_type,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO source(
                    filename, normalized_url, canonical_url, content_hash,
                    extractor_version, title, status, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    normalized_url=excluded.normalized_url,
                    canonical_url=excluded.canonical_url,
                    content_hash=excluded.content_hash,
                    extractor_version=excluded.extractor_version,
                    title=excluded.title,
                    status='active',
                    updated_at=excluded.updated_at
                """,
                (
                    filename,
                    requested_url,
                    canonical_url,
                    input_hash,
                    versions.extractor,
                    source.title,
                    now,
                    now,
                ),
            )
            source_id = int(
                self.connection.execute(
                    "SELECT id FROM source WHERE filename = ?", (filename,)
                ).fetchone()[0]
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO source_version(
                    source_id, content_hash, extractor_version, requested_url,
                    canonical_url, metadata_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    input_hash,
                    versions.extractor,
                    requested_url,
                    canonical_url,
                    metadata,
                    now,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO synthesis(
                    source_id, payload_json, input_hash, prompt_version,
                    schema_version, model_id, status, parse_error_json, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 'ready', '{}', ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    input_hash=excluded.input_hash,
                    prompt_version=excluded.prompt_version,
                    schema_version=excluded.schema_version,
                    model_id=excluded.model_id,
                    status='ready',
                    parse_error_json='{}',
                    updated_at=excluded.updated_at
                """,
                (
                    source_id,
                    payload,
                    input_hash,
                    versions.prompt,
                    versions.schema,
                    versions.synthesis_model,
                    now,
                ),
            )
            self.connection.execute(
                """
                UPDATE dirty_queue
                SET state='completed', claimed_by='', lease_expires_at='',
                    updated_at=?, error=''
                WHERE entity_type='source' AND entity_id=?
                  AND state IN ('pending', 'running')
                """,
                (now, filename),
            )
            self._prune_completed_dirty_unlocked()
            touched_moc_ids = {
                str(row[0])
                for row in self.connection.execute(
                    "SELECT moc_id FROM moc_source WHERE source_id=?", (source_id,)
                )
            }
            touched_concept_ids = {
                str(row[0])
                for row in self.connection.execute(
                    "SELECT concept_id FROM concept_source WHERE source_id=?",
                    (source_id,),
                )
            }
            self.connection.execute(
                "DELETE FROM membership_source WHERE source_id=?", (source_id,)
            )
            self.connection.execute(
                "DELETE FROM concept_source WHERE source_id=?", (source_id,)
            )
            self.connection.execute(
                "DELETE FROM moc_source WHERE source_id=?", (source_id,)
            )
            self._upsert_synthesis_entities(
                source_id, synthesis, versions, now
            )
            touched_moc_ids.update(
                str(row[0])
                for row in self.connection.execute(
                    "SELECT moc_id FROM moc_source WHERE source_id=?", (source_id,)
                )
            )
            self._rebuild_moc_memberships(touched_moc_ids)
            self._cleanup_unreferenced_entities(
                concept_ids=touched_concept_ids,
                moc_ids=touched_moc_ids,
            )

    def _assign_canonical_ids(self, synthesis: SourceSynthesis) -> None:
        """Resolve model-provided IDs only when they already exist locally."""
        for concept in synthesis.concepts:
            row = None
            if concept.canonical_id:
                row = self.connection.execute(
                    "SELECT id, slug FROM concept WHERE id=?", (concept.canonical_id,)
                ).fetchone()
            if row is None:
                row = self.connection.execute(
                    "SELECT id, slug FROM concept WHERE slug=?", (concept.slug,)
                ).fetchone()
            if row is not None:
                concept.canonical_id = str(row[0])
                concept.slug = str(row[1])
            else:
                concept.canonical_id = self._stable_id("concept", concept.slug)

    def assign_canonical_ids(self, synthesis: SourceSynthesis) -> None:
        """Populate trusted stable IDs without persisting synthesis state."""
        self._assign_canonical_ids(synthesis)

    @staticmethod
    def _stable_id(kind: str, slug: str) -> str:
        digest = hashlib.sha256(f"{kind}\x00{slug}".encode()).hexdigest()[:24]
        return f"{kind}-{digest}"

    def _upsert_synthesis_entities(
        self,
        source_id: int,
        synthesis: SourceSynthesis,
        versions: ArtifactVersions,
        now: str,
    ) -> None:
        """Update canonical concept/MoC rows and FTS documents."""
        for concept in synthesis.concepts:
            payload = concept_note_to_dict(concept)
            payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            tags_json = json.dumps(concept.tags, ensure_ascii=False, separators=(",", ":"))
            self.connection.execute(
                """
                INSERT INTO concept(
                    id, slug, title, summary, tags_json, payload_json,
                    content_hash, prompt_version, schema_version, model_id, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    slug=excluded.slug,
                    title=excluded.title,
                    summary=excluded.summary,
                    tags_json=excluded.tags_json,
                    payload_json=excluded.payload_json,
                    content_hash=excluded.content_hash,
                    prompt_version=excluded.prompt_version,
                    schema_version=excluded.schema_version,
                    model_id=excluded.model_id,
                    updated_at=excluded.updated_at
                """,
                (
                    concept.canonical_id,
                    concept.slug,
                    concept.title,
                    concept.summary,
                    tags_json,
                    payload_json,
                    _content_hash(payload_json),
                    versions.prompt,
                    versions.schema,
                    versions.synthesis_model,
                    now,
                ),
            )
            self.connection.execute(
                "DELETE FROM concept_alias WHERE concept_id=?", (concept.canonical_id,)
            )
            for alias in dict.fromkeys([concept.slug, concept.title, *concept.aliases]):
                normalized = " ".join(alias.casefold().split())
                if normalized:
                    self.connection.execute(
                        """
                        INSERT INTO concept_alias(alias, concept_id) VALUES(?, ?)
                        ON CONFLICT(alias) DO UPDATE SET concept_id=excluded.concept_id
                        """,
                        (normalized, concept.canonical_id),
                    )
            self.connection.execute(
                "DELETE FROM concept_fts WHERE concept_id=?", (concept.canonical_id,)
            )
            self.connection.execute(
                """
                INSERT INTO concept_fts(concept_id, slug, title, aliases, tags, summary)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    concept.canonical_id,
                    concept.slug,
                    concept.title,
                    " ".join(concept.aliases),
                    " ".join(concept.tags),
                    concept.summary,
                ),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO concept_source(source_id, concept_id) VALUES(?, ?)",
                (source_id, concept.canonical_id),
            )

        for moc in synthesis.maps:
            moc_id = self._stable_id("moc", moc.slug)
            payload_json = json.dumps(moc_to_dict(moc), ensure_ascii=False, separators=(",", ":"))
            self.connection.execute(
                """
                INSERT INTO moc(
                    id, slug, title, summary, tags_json, payload_json,
                    content_hash, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    slug=excluded.slug,
                    title=excluded.title,
                    summary=excluded.summary,
                    tags_json=excluded.tags_json,
                    payload_json=excluded.payload_json,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    moc_id,
                    moc.slug,
                    moc.title,
                    moc.summary,
                    json.dumps(moc.tags, ensure_ascii=False, separators=(",", ":")),
                    payload_json,
                    _content_hash(payload_json),
                    now,
                ),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO moc_source(source_id, moc_id) VALUES(?, ?)",
                (source_id, moc_id),
            )
            for position, slug in enumerate(moc.concept_slugs):
                row = self.connection.execute(
                    "SELECT id FROM concept WHERE slug=?", (slug,)
                ).fetchone()
                if row is not None:
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO membership_source(
                            source_id, moc_id, concept_id, position
                        ) VALUES(?, ?, ?, ?)
                        """,
                        (source_id, moc_id, str(row[0]), position),
                    )

    def _rebuild_moc_memberships(self, moc_ids: set[str]) -> None:
        ordered = sorted(moc_ids)
        for offset in range(0, len(ordered), 500):
            chunk = ordered[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            self.connection.execute(
                f"DELETE FROM membership WHERE moc_id IN ({placeholders})", chunk
            )
            self.connection.execute(
                f"""
                INSERT INTO membership(moc_id, concept_id, position)
                SELECT moc_id, concept_id, MIN(position)
                FROM membership_source
                WHERE moc_id IN ({placeholders})
                GROUP BY moc_id, concept_id
                """,
                chunk,
            )

    def _cleanup_unreferenced_entities(
        self,
        *,
        concept_ids: set[str] | None = None,
        moc_ids: set[str] | None = None,
    ) -> None:
        """Remove canonical rows that no live synthesis still references."""
        if concept_ids is not None or moc_ids is not None:
            for moc_id in sorted(moc_ids or set()):
                if self.connection.execute(
                    "SELECT 1 FROM moc_source WHERE moc_id=? LIMIT 1", (moc_id,)
                ).fetchone() is not None:
                    continue
                row = self.connection.execute(
                    "SELECT slug FROM moc WHERE id=?", (moc_id,)
                ).fetchone()
                if row is not None:
                    moc_slug = str(row[0])
                    self.connection.execute(
                        "DELETE FROM ann_bucket WHERE entity_kind='moc' AND entity_id=?",
                        (moc_slug,),
                    )
                    self.connection.execute(
                        "DELETE FROM embedding WHERE entity_kind='moc' AND entity_id=?",
                        (moc_slug,),
                    )
                    self.connection.execute("DELETE FROM moc WHERE id=?", (moc_id,))
            for concept_id in sorted(concept_ids or set()):
                if self.connection.execute(
                    "SELECT 1 FROM concept_source WHERE concept_id=? LIMIT 1",
                    (concept_id,),
                ).fetchone() is not None:
                    continue
                self.connection.execute(
                    "DELETE FROM ann_bucket WHERE entity_kind='concept' AND entity_id=?",
                    (concept_id,),
                )
                self.connection.execute(
                    "DELETE FROM embedding WHERE entity_kind='concept' AND entity_id=?",
                    (concept_id,),
                )
                self.connection.execute(
                    "DELETE FROM concept_fts WHERE concept_id=?", (concept_id,)
                )
                self.connection.execute("DELETE FROM concept WHERE id=?", (concept_id,))
            return

        stale_mocs = "SELECT id FROM moc WHERE id NOT IN (SELECT moc_id FROM moc_source)"
        stale_moc_slugs = (
            "SELECT slug FROM moc WHERE id NOT IN (SELECT moc_id FROM moc_source)"
        )
        self.connection.execute(
            f"DELETE FROM ann_bucket WHERE entity_kind='moc' "
            f"AND entity_id IN ({stale_moc_slugs})"
        )
        self.connection.execute(
            f"DELETE FROM embedding WHERE entity_kind='moc' "
            f"AND entity_id IN ({stale_moc_slugs})"
        )
        self.connection.execute(f"DELETE FROM moc WHERE id IN ({stale_mocs})")

        stale_concepts = (
            "SELECT id FROM concept "
            "WHERE id NOT IN (SELECT concept_id FROM concept_source)"
        )
        self.connection.execute(
            f"DELETE FROM ann_bucket WHERE entity_kind='concept' "
            f"AND entity_id IN ({stale_concepts})"
        )
        self.connection.execute(
            f"DELETE FROM embedding WHERE entity_kind='concept' "
            f"AND entity_id IN ({stale_concepts})"
        )
        self.connection.execute(
            f"DELETE FROM concept_fts WHERE concept_id IN ({stale_concepts})"
        )
        self.connection.execute(
            f"DELETE FROM concept WHERE id IN ({stale_concepts})"
        )

    def load_synthesis(self, filename: str) -> SourceSynthesis | None:
        """Load one synthesis by indexed source filename."""
        row = self.connection.execute(
            """
            SELECT synthesis.payload_json
            FROM synthesis
            JOIN source ON source.id = synthesis.source_id
            WHERE source.filename = ? AND synthesis.status = 'ready'
            """,
            (filename,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        synthesis = source_synthesis_from_dict(payload)
        synthesis.source_file = filename
        return synthesis

    def load_syntheses(
        self, filenames: set[str] | None = None
    ) -> dict[str, SourceSynthesis]:
        """Load ready syntheses from SQLite, optionally for an indexed subset."""
        if filenames is not None and not filenames:
            return {}
        if filenames is None:
            rows = self.connection.execute(
                """
                SELECT source.filename, synthesis.payload_json
                FROM synthesis JOIN source ON source.id=synthesis.source_id
                WHERE synthesis.status='ready' ORDER BY source.filename
                """
            ).fetchall()
        else:
            rows = []
            ordered = sorted(filenames)
            for offset in range(0, len(ordered), 500):
                chunk = ordered[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    self.connection.execute(
                        f"""
                        SELECT source.filename, synthesis.payload_json
                        FROM synthesis JOIN source ON source.id=synthesis.source_id
                        WHERE synthesis.status='ready'
                          AND source.filename IN ({placeholders})
                        ORDER BY source.filename
                        """,
                        chunk,
                    ).fetchall()
                )
        syntheses: dict[str, SourceSynthesis] = {}
        for row in rows:
            try:
                payload = json.loads(row[1])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            filename = str(row[0])
            synthesis = source_synthesis_from_dict(payload)
            synthesis.source_file = filename
            syntheses[filename] = synthesis
        return syntheses

    def source_filenames(self) -> set[str]:
        """Return active source identities without scanning source Markdown."""
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT filename FROM source WHERE status='active'"
            )
        }

    def delete_source(self, filename: str) -> None:
        """Delete one source and its versioned synthesis rows transactionally."""
        with self.connection:
            moc_ids = {
                str(row[0])
                for row in self.connection.execute(
                    """
                    SELECT ms.moc_id
                    FROM moc_source AS ms
                    JOIN source AS s ON s.id=ms.source_id
                    WHERE s.filename=?
                    """,
                    (filename,),
                )
            }
            concept_ids = {
                str(row[0])
                for row in self.connection.execute(
                    """
                    SELECT cs.concept_id
                    FROM concept_source AS cs
                    JOIN source AS s ON s.id=cs.source_id
                    WHERE s.filename=?
                    """,
                    (filename,),
                )
            }
            self.connection.execute("DELETE FROM source WHERE filename=?", (filename,))
            self._rebuild_moc_memberships(moc_ids)
            self._cleanup_unreferenced_entities(
                concept_ids=concept_ids,
                moc_ids=moc_ids,
            )

    def render_diff(
        self,
        entity_kind: str,
        manifest: dict[str, tuple[str, str]],
    ) -> RenderDiff:
        """Compare entity render fingerprints without reading generated files."""
        rows = self.connection.execute(
            "SELECT entity_key, path, content_hash FROM render_state WHERE entity_kind=?",
            (entity_kind,),
        ).fetchall()
        existing = {str(row[0]): (str(row[1]), str(row[2])) for row in rows}
        changed = {
            key
            for key, value in manifest.items()
            if existing.get(key) != value
        }
        stale_keys = set(existing) - set(manifest)
        return RenderDiff(
            changed_keys=frozenset(changed),
            stale_paths=tuple(existing[key][0] for key in sorted(stale_keys)),
            stale_keys=frozenset(stale_keys),
        )

    def commit_render_manifest(
        self,
        entity_kind: str,
        manifest: dict[str, tuple[str, str]],
        diff: RenderDiff,
    ) -> None:
        """Commit only changed/stale render fingerprints after render success."""
        now = _now()
        with self.connection:
            for key in diff.changed_keys:
                path, content_hash = manifest[key]
                self.connection.execute(
                    """
                    INSERT INTO render_state(
                        entity_kind, entity_key, path, content_hash, updated_at
                    ) VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(entity_kind, entity_key) DO UPDATE SET
                        path=excluded.path,
                        content_hash=excluded.content_hash,
                        updated_at=excluded.updated_at
                    """,
                    (entity_kind, key, path, content_hash, now),
                )
            if diff.stale_keys:
                self.connection.executemany(
                    "DELETE FROM render_state WHERE entity_kind=? AND entity_key=?",
                    [(entity_kind, key) for key in diff.stale_keys],
                )

    def next_maintenance_batch(self, *, limit: int) -> tuple[list[str], bool]:
        """Return a bounded, persistent round-robin batch of rendered pages."""
        if limit < 1:
            raise ValueError("maintenance batch limit must be positive")
        cursor_row = self.connection.execute(
            "SELECT value FROM meta WHERE key='maintenance_cursor'"
        ).fetchone()
        cursor = str(cursor_row[0]) if cursor_row is not None else ""
        rows = self.connection.execute(
            """
            SELECT path FROM render_state
            WHERE path > ?
            ORDER BY path
            LIMIT ?
            """,
            (cursor, limit + 1),
        ).fetchall()
        cycle_complete = len(rows) <= limit
        paths = [str(row[0]) for row in rows[:limit]]
        next_cursor = "" if cycle_complete else paths[-1]
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO meta(key, value) VALUES('maintenance_cursor', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (next_cursor,),
            )
        return paths, cycle_complete

    def maintenance_graph_sets(self) -> tuple[set[str], set[str]]:
        """Return current concept and MoC-membership slugs from indexed state."""
        concepts = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT entity_key FROM render_state WHERE entity_kind='concept-page'"
            )
        }
        moced = {
            str(row[0])
            for row in self.connection.execute(
                """
                SELECT c.slug
                FROM membership AS ms
                JOIN concept AS c ON c.id=ms.concept_id
                JOIN moc AS m ON m.id=ms.moc_id
                JOIN render_state AS rs
                  ON rs.entity_kind='moc-page' AND rs.entity_key=m.slug
                """
            )
        }
        return concepts, moced

    def search_concepts(self, query: str, *, limit: int) -> list[ConceptCandidate]:
        """Return a bounded FTS5 candidate set without scanning concept rows."""
        if limit <= 0:
            return []
        tokens = list(
            dict.fromkeys(
                token.casefold()
                for token in re.findall(r"[^\W_]{2,}", query, flags=re.UNICODE)
            )
        )[:32]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        rows = self.connection.execute(
            """
            SELECT concept.id, concept.slug, concept.title, concept.summary,
                   COALESCE((
                       SELECT group_concat(alias, char(31))
                       FROM concept_alias WHERE concept_id=concept.id
                   ), ''),
                   bm25(concept_fts)
            FROM concept_fts
            JOIN concept ON concept.id=concept_fts.concept_id
            WHERE concept_fts MATCH ?
            ORDER BY bm25(concept_fts), concept.slug
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [
            ConceptCandidate(
                id=str(row[0]),
                slug=str(row[1]),
                title=str(row[2]),
                summary=str(row[3]),
                aliases=tuple(alias for alias in str(row[4]).split("\x1f") if alias),
                score=-float(row[5]),
            )
            for row in rows
        ]

    def concept_candidates_by_id(
        self, entity_ids: list[str]
    ) -> list[ConceptCandidate]:
        """Load a bounded ANN result set without scanning canonical concepts."""
        if not entity_ids:
            return []
        result_by_id: dict[str, ConceptCandidate] = {}
        for offset in range(0, len(entity_ids), 500):
            chunk = entity_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"""
                SELECT id, slug, title, summary,
                       COALESCE((
                           SELECT group_concat(alias, char(31))
                           FROM concept_alias WHERE concept_id=concept.id
                       ), '')
                FROM concept WHERE id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                result_by_id[str(row[0])] = ConceptCandidate(
                    id=str(row[0]),
                    slug=str(row[1]),
                    title=str(row[2]),
                    summary=str(row[3]),
                    aliases=tuple(alias for alias in str(row[4]).split("\x1f") if alias),
                    score=0.0,
                )
        return [result_by_id[entity_id] for entity_id in entity_ids if entity_id in result_by_id]

    def upsert_embedding(
        self,
        entity_kind: str,
        entity_id: str,
        vector: list[float],
        *,
        model_id: str,
        input_hash: str,
        schema_version: str,
    ) -> None:
        """Persist one float32 vector and replace its ANN bucket entries."""
        self.upsert_embeddings(
            entity_kind,
            [(entity_id, vector, input_hash)],
            model_id=model_id,
            schema_version=schema_version,
            batch_size=1,
        )

    def upsert_embeddings(
        self,
        entity_kind: str,
        records: Iterable[tuple[str, list[float], str]],
        *,
        model_id: str,
        schema_version: str,
        batch_size: int = 256,
    ) -> int:
        """Persist vectors in bounded transactions instead of one commit per row."""
        if batch_size < 1:
            raise ValueError("embedding batch_size must be positive")
        iterator = iter(records)
        written = 0
        while batch := list(islice(iterator, batch_size)):
            now = _now()
            embedding_rows: list[tuple[object, ...]] = []
            bucket_rows: list[tuple[object, ...]] = []
            delete_rows: list[tuple[str, str, str]] = []
            for entity_id, vector, input_hash in batch:
                canonical = _canonical_vector(vector)
                dimensions = len(canonical)
                embedding_rows.append(
                    (
                        entity_kind,
                        entity_id,
                        model_id,
                        dimensions,
                        input_hash,
                        schema_version,
                        struct.pack(f"<{dimensions}f", *canonical),
                        now,
                    )
                )
                delete_rows.append((entity_kind, entity_id, model_id))
                bucket_rows.extend(
                    (
                        entity_kind,
                        model_id,
                        dimensions,
                        table_id,
                        _lsh_bucket(canonical, model_id, table_id),
                        entity_id,
                    )
                    for table_id in range(_LSH_TABLES)
                )
            with self.connection:
                self.connection.executemany(
                    """
                    INSERT INTO embedding(
                        entity_kind, entity_id, model_id, dimensions, input_hash,
                        schema_version, vector, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_kind, entity_id, model_id) DO UPDATE SET
                        dimensions=excluded.dimensions,
                        input_hash=excluded.input_hash,
                        schema_version=excluded.schema_version,
                        vector=excluded.vector,
                        updated_at=excluded.updated_at
                    """,
                    embedding_rows,
                )
                self.connection.executemany(
                    """
                    DELETE FROM ann_bucket
                    WHERE entity_kind=? AND entity_id=? AND model_id=?
                    """,
                    delete_rows,
                )
                self.connection.executemany(
                    """
                    INSERT INTO ann_bucket(
                        entity_kind, model_id, dimensions, table_id, bucket, entity_id
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    bucket_rows,
                )
            written += len(batch)
        return written

    def load_embedding(
        self,
        entity_kind: str,
        entity_id: str,
        *,
        model_id: str,
    ) -> list[float] | None:
        """Load one authoritative float32 vector by its versioned key."""
        row = self.connection.execute(
            """
            SELECT dimensions, vector FROM embedding
            WHERE entity_kind=? AND entity_id=? AND model_id=?
            """,
            (entity_kind, entity_id, model_id),
        ).fetchone()
        if row is None:
            return None
        dimensions = int(row[0])
        raw = bytes(row[1])
        if dimensions <= 0 or len(raw) != dimensions * 4:
            return None
        return list(struct.unpack(f"<{dimensions}f", raw))

    def load_embeddings(
        self,
        entity_kind: str,
        entity_ids: Iterable[str],
        *,
        model_id: str,
    ) -> dict[str, list[float]]:
        """Load a bounded ID subset without materializing the vector table."""
        result: dict[str, list[float]] = {}
        ordered = list(dict.fromkeys(entity_ids))
        for offset in range(0, len(ordered), 500):
            chunk = ordered[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"""
                SELECT entity_id, dimensions, vector FROM embedding
                WHERE entity_kind=? AND model_id=?
                  AND entity_id IN ({placeholders})
                """,
                [entity_kind, model_id, *chunk],
            ).fetchall()
            for row in rows:
                dimensions = int(row[1])
                raw = bytes(row[2])
                if dimensions > 0 and len(raw) == dimensions * 4:
                    result[str(row[0])] = list(
                        struct.unpack(f"<{dimensions}f", raw)
                    )
        return result

    def has_embeddings(
        self,
        entity_kind: str,
        *,
        model_id: str,
        schema_version: str | None = None,
    ) -> bool:
        """Check whether a versioned ANN namespace has any persisted vectors."""
        if schema_version is None:
            row = self.connection.execute(
                """
                SELECT 1 FROM embedding
                WHERE entity_kind=? AND model_id=? LIMIT 1
                """,
                (entity_kind, model_id),
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                SELECT 1 FROM embedding
                WHERE entity_kind=? AND model_id=? AND schema_version=? LIMIT 1
                """,
                (entity_kind, model_id, schema_version),
            ).fetchone()
        return row is not None

    def search_ann(
        self,
        entity_kind: str,
        vector: list[float],
        *,
        model_id: str,
        limit: int,
        candidate_pool: int = 1_000,
        schema_version: str | None = None,
    ) -> AnnSearchResult:
        """Search bounded LSH buckets, then exact-rerank only that pool."""
        if limit <= 0 or candidate_pool <= 0:
            return AnnSearchResult([], 0, 0)
        canonical = _canonical_vector(vector)
        dimensions = len(canonical)
        collision_counts: dict[str, int] = {}
        bucket_hits = 0
        for table_id in range(_LSH_TABLES):
            bucket = _lsh_bucket(canonical, model_id, table_id)
            probes = [bucket, *(bucket ^ (1 << bit) for bit in range(_LSH_BITS))]
            placeholders = ",".join("?" for _ in probes)
            rows = self.connection.execute(
                f"""
                SELECT entity_id FROM ann_bucket
                WHERE entity_kind=? AND model_id=? AND dimensions=? AND table_id=?
                  AND bucket IN ({placeholders})
                ORDER BY rowid
                LIMIT ?
                """,
                (entity_kind, model_id, dimensions, table_id, *probes, candidate_pool),
            ).fetchall()
            bucket_hits += len(rows)
            for row in rows:
                entity_id = str(row[0])
                collision_counts[entity_id] = collision_counts.get(entity_id, 0) + 1

        selected_ids = [
            entity_id
            for entity_id, _hits in sorted(
                collision_counts.items(), key=lambda item: (-item[1], item[0])
            )[:candidate_pool]
        ]
        scored: list[AnnCandidate] = []
        for offset in range(0, len(selected_ids), 500):
            chunk = selected_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            schema_filter = "" if schema_version is None else " AND schema_version=?"
            params: list[object] = [entity_kind, model_id, dimensions]
            if schema_version is not None:
                params.append(schema_version)
            params.extend(chunk)
            rows = self.connection.execute(
                f"""
                SELECT entity_id, vector FROM embedding
                WHERE entity_kind=? AND model_id=? AND dimensions=?
                  {schema_filter}
                  AND entity_id IN ({placeholders})
                """,
                params,
            ).fetchall()
            for row in rows:
                candidate = list(struct.unpack(f"<{dimensions}f", bytes(row[1])))
                scored.append(AnnCandidate(str(row[0]), _cosine(canonical, candidate)))
        scored.sort(key=lambda item: (-item.score, item.entity_id))
        return AnnSearchResult(scored[:limit], len(scored), bucket_hits)

    def migrate_legacy(self, versions: ArtifactVersions) -> bool:
        """Import legacy JSON state/cache exactly once.

        The migration marker is written last. Partial imports after an external
        interruption are safe to retry because every row uses deterministic
        keys and upsert semantics.
        """
        marker = self.connection.execute(
            "SELECT value FROM meta WHERE key='legacy_migration_complete'"
        ).fetchone()
        if marker is not None and str(marker[0]) == "1":
            return False

        state_hashes: dict[str, str] = {}
        state_path = self.llmwiki_dir / "state.json"
        try:
            raw_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_state = {}
        raw_sources = raw_state.get("sources", {}) if isinstance(raw_state, dict) else {}
        if isinstance(raw_sources, dict):
            for filename, raw in raw_sources.items():
                if isinstance(filename, str) and isinstance(raw, dict):
                    value = raw.get("hash", "")
                    if isinstance(value, str):
                        state_hashes[filename] = value

        cache_dir = self.llmwiki_dir / "cache"
        if cache_dir.is_dir():
            for path in sorted(cache_dir.glob("*.json")):
                if path.name == "_resynthesis_overlay.json":
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("_artifact"), dict) and isinstance(
                    payload.get("payload"), dict
                ):
                    payload = payload["payload"]
                try:
                    synthesis = source_synthesis_from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    continue
                filename = synthesis.source_file or path.name.removesuffix(".json")
                synthesis.source_file = filename
                source = SourceDoc(
                    title=synthesis.source_title or Path(filename).stem,
                    content=synthesis.source_summary,
                    source_file=filename,
                )
                self.upsert_synthesis(filename, source, synthesis, versions)
                legacy_hash = state_hashes.get(filename)
                if legacy_hash:
                    with self.connection:
                        self.connection.execute(
                            "UPDATE source SET content_hash=? WHERE filename=?",
                            (legacy_hash, filename),
                        )
                        self.connection.execute(
                            """
                            UPDATE synthesis SET input_hash=?
                            WHERE source_id=(SELECT id FROM source WHERE filename=?)
                            """,
                            (legacy_hash, filename),
                        )

        now = _now()
        with self.connection:
            for filename, legacy_hash in state_hashes.items():
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO source(
                        filename, content_hash, extractor_version, status,
                        created_at, updated_at
                    ) VALUES(?, ?, ?, 'active', ?, ?)
                    """,
                    (filename, legacy_hash, versions.extractor, now, now),
                )
            self.connection.execute(
                """
                INSERT INTO meta(key, value) VALUES('legacy_migration_complete', '1')
                ON CONFLICT(key) DO UPDATE SET value='1'
                """
            )
        return True

    def synthesis_is_current(
        self,
        filename: str,
        content: str,
        versions: ArtifactVersions,
    ) -> bool:
        """Check content and version compatibility, recording invalidation."""
        row = self.connection.execute(
            """
            SELECT synthesis.input_hash, source.extractor_version,
                   synthesis.prompt_version, synthesis.schema_version,
                   synthesis.model_id
            FROM synthesis
            JOIN source ON source.id = synthesis.source_id
            WHERE source.filename = ? AND synthesis.status = 'ready'
            """,
            (filename,),
        ).fetchone()
        if row is None:
            return False
        expected = (
            _content_hash(content),
            versions.extractor,
            versions.prompt,
            versions.schema,
            versions.synthesis_model,
        )
        actual = tuple(row)
        if actual == expected:
            return True
        reasons = (
            "content_hash_changed",
            "extractor_version_changed",
            "prompt_version_changed",
            "schema_version_changed",
            "synthesis_model_changed",
        )
        reason = next(
            reason
            for reason, current, wanted in zip(reasons, actual, expected, strict=True)
            if current != wanted
        )
        self.enqueue_dirty("source", filename, reason)
        return False

    def enqueue_dirty(
        self,
        entity_type: str,
        entity_id: str,
        reason: str,
        payload: dict[str, object] | None = None,
        *,
        next_attempt_at: str | None = None,
    ) -> int:
        """Insert one deduplicated pending work item and return its ID."""
        now = _now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO dirty_queue(
                    entity_type, entity_id, reason, payload_json, state,
                    next_attempt_at, created_at, updated_at
                ) VALUES(?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(entity_type, entity_id, reason)
                    WHERE state IN ('pending', 'running')
                DO UPDATE SET
                    payload_json=excluded.payload_json,
                    next_attempt_at=excluded.next_attempt_at,
                    updated_at=excluded.updated_at
                """,
                (
                    entity_type,
                    entity_id,
                    reason,
                    json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":")),
                    next_attempt_at or now,
                    now,
                    now,
                ),
            )
            row = self.connection.execute(
                """
                SELECT id FROM dirty_queue
                WHERE entity_type=? AND entity_id=? AND reason=?
                  AND state IN ('pending', 'running')
                """,
                (entity_type, entity_id, reason),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def claim_dirty(self, *, limit: int, worker_id: str) -> list[DirtyRecord]:
        """Atomically claim at most ``limit`` due items for one worker."""
        if limit <= 0:
            return []
        now = _now()
        lease_expires_at = (datetime.now(UTC) + timedelta(
            seconds=_DIRTY_LEASE_SECONDS
        )).isoformat()
        with self.connection:
            rows = self.connection.execute(
                """
                UPDATE dirty_queue
                SET state='running', attempts=attempts+1, claimed_by=?,
                    lease_expires_at=?, updated_at=?
                WHERE id IN (
                    SELECT id FROM dirty_queue
                    WHERE (state='pending' AND next_attempt_at <= ?)
                       OR (state='running' AND lease_expires_at != ''
                           AND lease_expires_at <= ?)
                    ORDER BY next_attempt_at, id
                    LIMIT ?
                )
                RETURNING id, entity_type, entity_id, reason, state, attempts,
                          next_attempt_at, payload_json, error
                """,
                (worker_id, lease_expires_at, now, now, now, limit),
            ).fetchall()
        return sorted(self._dirty_records(rows), key=lambda record: record.id)

    def start_dirty(self, record_id: int, *, worker_id: str) -> None:
        """Claim a known item owned by an external sequential work plan."""
        now = datetime.now(UTC)
        with self.connection:
            self.connection.execute(
                """
                UPDATE dirty_queue
                SET state='running', attempts=attempts+1, claimed_by=?,
                    lease_expires_at=?, updated_at=?
                WHERE id=? AND state IN ('pending', 'running')
                """,
                (
                    worker_id,
                    (now + timedelta(seconds=_DIRTY_LEASE_SECONDS)).isoformat(),
                    now.isoformat(),
                    record_id,
                ),
            )

    def complete_dirty(self, record_id: int) -> None:
        """Mark a claimed queue item complete."""
        with self.connection:
            self.connection.execute(
                """
                UPDATE dirty_queue
                SET state='completed', claimed_by='', lease_expires_at='',
                    error='', updated_at=?
                WHERE id=?
                """,
                (_now(), record_id),
            )
            self._prune_completed_dirty_unlocked()

    def _prune_completed_dirty_unlocked(
        self, keep: int = _COMPLETED_QUEUE_HISTORY
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM dirty_queue
            WHERE state='completed'
              AND id NOT IN (
                  SELECT id FROM dirty_queue
                  WHERE state='completed'
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (keep,),
        )

    def fail_dirty(
        self,
        record_id: int,
        error: str,
        *,
        next_attempt_at: str | None = None,
    ) -> None:
        """Return a failed claim to the retry queue with bounded diagnostics."""
        if next_attempt_at is None:
            row = self.connection.execute(
                "SELECT attempts FROM dirty_queue WHERE id=?", (record_id,)
            ).fetchone()
            attempts = int(row[0]) if row is not None else 1
            delay_seconds = min(86_400, 60 * (2 ** max(0, attempts - 1)))
            next_attempt_at = datetime.fromtimestamp(
                datetime.now(UTC).timestamp() + delay_seconds,
                tz=UTC,
            ).isoformat()
        with self.connection:
            self.connection.execute(
                """
                UPDATE dirty_queue
                SET state='pending', claimed_by='', lease_expires_at='',
                    error=?, next_attempt_at=?, updated_at=?
                WHERE id=?
                """,
                (error[:1_000], next_attempt_at, _now(), record_id),
            )

    def upsert_ingest_hash(
        self,
        *,
        requested_url: str,
        canonical_url: str,
        content_hash: str,
        output_file: str,
    ) -> None:
        """Atomically index URL and content identities for a written source."""
        normalized = _normalize_url(requested_url or canonical_url)
        canonical = _normalize_url(canonical_url or requested_url)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO ingest_hash(
                    normalized_url, canonical_url, content_hash, output_file, updated_at
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(normalized_url) DO UPDATE SET
                    canonical_url=excluded.canonical_url,
                    content_hash=excluded.content_hash,
                    output_file=excluded.output_file,
                    updated_at=excluded.updated_at
                """,
                (normalized, canonical, content_hash, output_file, _now()),
            )

    def ingest_index_empty(self) -> bool:
        """Return whether legacy source hashes still need one-time backfill."""
        return self.connection.execute(
            "SELECT 1 FROM ingest_hash LIMIT 1"
        ).fetchone() is None

    def ingest_backfill_complete(self) -> bool:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key='ingest_backfill_complete'"
        ).fetchone()
        return row is not None and str(row[0]) == "1"

    def mark_ingest_backfill_complete(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO meta(key, value) VALUES('ingest_backfill_complete', '1')
                ON CONFLICT(key) DO UPDATE SET value='1'
                """
            )

    def find_ingest_duplicate(
        self,
        *,
        requested_url: str = "",
        content_hash: str = "",
    ) -> IngestMatch | None:
        """Resolve a duplicate by canonicalized URL or exact body hash."""
        row = None
        normalized = _normalize_url(requested_url) if requested_url else ""
        if normalized:
            row = self.connection.execute(
                """
                SELECT normalized_url, canonical_url, content_hash, output_file
                FROM ingest_hash
                WHERE normalized_url=? OR canonical_url=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (normalized, normalized),
            ).fetchone()
        if row is None and content_hash:
            row = self.connection.execute(
                """
                SELECT normalized_url, canonical_url, content_hash, output_file
                FROM ingest_hash WHERE content_hash=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
        if row is None:
            return None
        return IngestMatch(*(str(value) for value in row))

    def list_dirty(self, *, limit: int = 100, state: str = "pending") -> list[DirtyRecord]:
        """Return bounded queue records in deterministic claim order."""
        rows = self.connection.execute(
            """
            SELECT id, entity_type, entity_id, reason, state, attempts,
                   next_attempt_at, payload_json, error
            FROM dirty_queue
            WHERE state=?
            ORDER BY next_attempt_at, id
            LIMIT ?
            """,
            (state, max(0, limit)),
        ).fetchall()
        return self._dirty_records(rows)

    @staticmethod
    def _dirty_records(rows: list[sqlite3.Row]) -> list[DirtyRecord]:
        result: list[DirtyRecord] = []
        for row in rows:
            try:
                payload = json.loads(row[7])
            except json.JSONDecodeError:
                payload = {}
            result.append(
                DirtyRecord(
                    id=int(row[0]),
                    entity_type=str(row[1]),
                    entity_id=str(row[2]),
                    reason=str(row[3]),
                    state=str(row[4]),
                    attempts=int(row[5]),
                    next_attempt_at=str(row[6]),
                    payload=payload if isinstance(payload, dict) else {},
                    error=str(row[8]),
                )
            )
        return result

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> VaultRegistry:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.connection.rollback()
        self.close()
