"""``olw ingest`` — extract URLs + collect clippings → write source files."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import signal
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import print_result_summary, resolve_vault
from obsidian_llm_wiki.core.lock import acquire_lock, release_lock
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.operations import OperationRecord, OperationStatus, OperationStore
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.ingest.runtime import extraction_policy
from obsidian_llm_wiki.ingest.sources import (
    _SOURCE_INDEX_SHARD,
    _SOURCE_METADATA_FILES,
    _discover_source_paths,
    load_sources_incremental,
)
from obsidian_llm_wiki.render.frontmatter import parse_frontmatter
from obsidian_llm_wiki.render.obsidian import atomic_write, slugify

_PlannedSource = tuple[str, str, SourceDoc | None, bool]
logger = logging.getLogger("obswiki.cli.ingest")

LEDGER_TEMPLATE = """\
---
type: ledger
title: Failed URL Ingestion Ledger
timestamp: {timestamp}
---

# Failed URL Ingestion Ledger

This file records URLs that permanently failed extraction after all fallback
strategies were exhausted. Each entry shows the URL, the error, and the date.

To retry: manually remove the entry and re-run ``olw ingest``.

| Date | URL | Error |
|------|-----|-------|
{rows}
"""


class _CancellationToken:
    """A SIGINT-aware flag checked between source operations."""

    def __init__(self) -> None:
        self.cancelled = False

    def request(self, _signum: int, _frame: object) -> None:
        self.cancelled = True


def _extract_direct(url: str) -> SourceDoc:
    from obsidian_llm_wiki.ingest.extractors import extract

    return extract(url)


class _BoundedURLExtractor:
    """Prefetch URL extraction with a fixed in-memory and worker budget."""

    def __init__(self, urls: list[str], config: Any, workers: int) -> None:
        self._urls = iter(urls)
        self._config = config
        self._workers = max(1, min(workers, 8))
        self._executor: ThreadPoolExecutor | None = None
        self._environment: Any = None
        self._active: dict[str, Future[SourceDoc]] = {}
        self._completed: dict[str, SourceDoc | BaseException] = {}

    def __enter__(self) -> _BoundedURLExtractor:
        from obsidian_llm_wiki.config import extraction_environment

        self._environment = extraction_environment(
            self._config, use_residential_proxy=False
        )
        self._environment.__enter__()
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers,
            thread_name_prefix="olw-extract",
        )
        self._refill()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if self._environment is not None:
            self._environment.__exit__(exc_type, exc, traceback)

    def extract(self, url: str) -> SourceDoc:
        outcome: SourceDoc | BaseException
        if url in self._completed:
            outcome = self._completed.pop(url)
        else:
            future = self._active.pop(url, None)
            if future is None:
                raise RuntimeError(f"URL extraction was not scheduled: {url}")
            try:
                outcome = future.result()
            except BaseException as exc:
                outcome = exc

        if isinstance(outcome, BaseException):
            # Proxy fallback mutates a compatibility environment bridge. Drain
            # the bounded direct window first so no worker observes that change.
            self._drain_active()
            if (
                self._config.residential_proxy_url
                and _can_retry_with_residential_proxy(url)
            ):
                from obsidian_llm_wiki.config import extraction_environment

                logger.info("Retrying extraction through residential proxy: %s", url)
                try:
                    with extraction_environment(self._config):
                        outcome = _extract_direct(url)
                except BaseException as exc:
                    outcome = exc
        self._refill()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def _drain_active(self) -> None:
        for url, future in list(self._active.items()):
            try:
                self._completed[url] = future.result()
            except BaseException as exc:
                self._completed[url] = exc
        self._active.clear()

    def _refill(self) -> None:
        assert self._executor is not None
        while len(self._active) + len(self._completed) < self._workers:
            try:
                url = next(self._urls)
            except StopIteration:
                return
            self._active[url] = self._executor.submit(copy_context().run, _extract_direct, url)


@contextmanager
def _cooperative_sigint(token: _CancellationToken) -> Iterator[None]:
    """Turn the first Ctrl-C into a safe stop after the current source."""
    previous = signal.getsignal(signal.SIGINT)
    installed = False
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, token.request)
        installed = True
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous)


def _run_id() -> str:
    return f"ingest-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _bounded_source(source: SourceDoc, maximum_bytes: int) -> tuple[SourceDoc, bool]:
    """Backward-compatible CLI wrapper for canonical source normalization."""
    from obsidian_llm_wiki.core.source_content import bound_source_content

    return bound_source_content(source, maximum_bytes)


def _source_event(
    source: str,
    source_kind: str,
    status: OperationStatus | str,
    **details: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "source",
        "source": source,
        "source_kind": source_kind,
        "status": str(status),
    }
    event.update(
        {
            key: value
            for key, value in details.items()
            if value is not None and value != "" and value is not False
        }
    )
    return event


def _emit(json_output: bool, event: dict[str, Any], text: str = "") -> None:
    """Emit NDJSON in machine mode and concise text otherwise."""
    if json_output:
        typer.echo(json.dumps(event, ensure_ascii=False, sort_keys=True))
    elif text:
        typer.echo(text)


@dataclass
class _IngestOutcome:
    """Keep source and pipeline failures visible at every terminal return."""

    run_id: str
    mode: str
    json_output: bool
    succeeded: int = 0
    extraction_errors: list[str] = field(default_factory=list)
    synthesis_errors: list[str] = field(default_factory=list)
    inventory_errors: list[str] = field(default_factory=list)

    def finish(self, sources: int, text: str = "", **details: Any) -> None:
        errors = self.extraction_errors + self.synthesis_errors + self.inventory_errors
        has_success = self.succeeded or details.get("compiled", 0)
        event = "partial_failure" if has_success and errors else "failed" if errors else "completed"
        if errors:
            text = (
                f"\n❌ {event.replace('_', ' ').capitalize()}: {len(errors)} failure(s) "
                f"({len(self.extraction_errors)} extraction, "
                f"{len(self.synthesis_errors)} synthesis, "
                f"{len(self.inventory_errors)} inventory)."
                + "".join(f"\n   - {error}" for error in errors)
            )
            if self.mode == "preview":
                text += "\n   Preview — no files written."
        _emit(
            self.json_output,
            {
                "type": "run", "event": event, "run_id": self.run_id, "mode": self.mode,
                "sources": sources, "succeeded": self.succeeded,
                "extraction_failures": len(self.extraction_errors),
                "synthesis_failures": len(self.synthesis_errors),
                "inventory_failures": len(self.inventory_errors),
                "failure_count": len(errors), "errors": errors,
                **details,
            },
            text,
        )
        if errors:
            raise typer.Exit(code=1)


def _can_retry_with_residential_proxy(url: str) -> bool:
    """Return whether proxy fallback is permitted for this public source."""
    hostname = (urlparse(url).hostname or "").casefold()
    # SSRN is deliberately public-only: use its public metadata fallback, not
    # a proxy or other access-control bypass.
    return not (hostname == "ssrn.com" or hostname.endswith(".ssrn.com"))


def _reserve_collision_safe_path(
    sources_dir: Path,
    source: SourceDoc,
    *,
    preferred_stem: str | None = None,
) -> Path:
    """Atomically reserve a source filename for a concurrent ingestion writer."""
    stem = slugify(preferred_stem) if preferred_stem else slugify(source.title)
    for index in range(0, 101):
        suffix = "" if index == 0 else f"-{index}"
        candidate = sources_dir / f"{stem}{suffix}.md"
        # Allocation must agree with the source loader's metadata exclusions.
        if (
            candidate.name in _SOURCE_METADATA_FILES
            or _SOURCE_INDEX_SHARD.fullmatch(candidate.name)
        ):
            continue
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        else:
            os.close(fd)
            return candidate
    raise RuntimeError(f"Too many source filename collisions for {source.title!r}")


def _collision_safe_path(sources_dir: Path, source: SourceDoc) -> Path:
    filepath = sources_dir / f"{slugify(source.title)}.md"
    if not filepath.exists():
        return filepath
    stem = filepath.stem
    for index in range(1, 101):
        candidate = sources_dir / f"{stem}-{index}.md"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many source filename collisions for {source.title!r}")


def _ingest_source_paths(sources_dir: Path) -> list[Path]:
    """Use the loader's complete validated inventory, including before backfill."""
    try:
        return _discover_source_paths(sources_dir)
    except FileNotFoundError:
        # A new vault need not have a sources directory yet.
        return []


def _content_hash_matches_existing(
    sources_dir: Path,
    content: str,
) -> Path | None:
    """Return the existing source file whose body SHA-256 matches, if any.

    Compares the stripped body of existing source files (excluding frontmatter
    and the title heading) against the new source's content hash. This catches
    both same-URL re-extraction and different-URL-same-content duplicates.
    """
    target_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
    for existing in _ingest_source_paths(sources_dir):
        raw = existing.read_text(encoding="utf-8")
        _, body = parse_frontmatter(raw)
        # Strip the leading title heading so two sources with different
        # titles but identical body content are recognized as duplicates.
        body_stripped = _dedupe_body(body)
        from obsidian_llm_wiki.core.state import hash_content

        if hash_content(body_stripped) == target_hash:
            return existing
    return None


def _dedupe_body(body: str) -> str:
    """Use the same rendered-body identity for ingest, migration and validation."""
    body = body.strip()
    if body.startswith("# "):
        return body.partition("\n")[2].strip()
    return body


def _indexed_source_path(
    sources_dir: Path, filename: str, *, source_paths: set[Path]
) -> Path | None:
    """Treat an ingest-index filename as untrusted persisted metadata."""
    from obsidian_llm_wiki.core.source_files import source_file_path

    try:
        resolved = source_file_path(sources_dir, filename)
        path = sources_dir / filename
        return path if path in source_paths and resolved.is_file() else None
    except (OSError, ValueError):
        return None


def _backfill_ingest_index(
    registry: VaultRegistry, sources_dir: Path, *, source_paths: list[Path] | None = None
) -> None:
    """Scan legacy source files once, then use indexed duplicate lookups."""
    if source_paths is None:
        source_paths = _ingest_source_paths(sources_dir)
    if registry.ingest_backfill_complete():
        return
    from obsidian_llm_wiki.core.state import hash_content

    for existing in source_paths:
        metadata, body = parse_frontmatter(existing.read_text(encoding="utf-8"))
        body_stripped = _dedupe_body(body)
        requested_url = str(metadata.get("url", ""))
        registry.upsert_ingest_hash(
            requested_url=requested_url or f"legacy:{existing.name}",
            canonical_url=requested_url,
            content_hash=hash_content(body_stripped),
            output_file=existing.name,
        )
    registry.mark_ingest_backfill_complete()


def _archive_clipping(clipping_path: Path, config: Any) -> Path | None:
    """Move a processed clipping to 02-Clippings/processed/, preserving its name.

    Returns the destination path, or None if the move failed.
    """
    return _archive_without_overwrite(clipping_path, config.clippings_dir / "processed")


def _archive_without_overwrite(path: Path, archive_dir: Path) -> Path | None:
    """Atomically claim an archive name; retain the input if archiving fails.

    Both directories normally share a vault filesystem. Hard-link + unlink
    avoids rename's replace-existing semantics. On cross-device paths, retain
    the queued original rather than risking an overwrite or incomplete copy.
    """
    if path.is_symlink() or archive_dir.is_symlink():
        return None
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for index in range(100):
            suffix = f"-{index}" if index else ""
            destination = archive_dir / f"{path.stem}{suffix}{path.suffix}"
            try:
                os.link(path, destination, follow_symlinks=False)
            except FileExistsError:
                continue
            path.unlink()
            return destination
    except OSError as exc:
        logger.warning("Could not archive %s: %s", path, exc)
    return None


def _operation_for_source(
    store: OperationStore,
    *,
    run_id: str,
    source: str,
    source_kind: str,
    resume: bool,
) -> OperationRecord:
    """Create a record, reusing a cancelled/failed record when explicitly resumed."""
    previous = store.latest_for_source(source) if resume else None
    if previous and previous.status in {
        OperationStatus.CANCELLED,
        OperationStatus.FAILED,
        OperationStatus.RUNNING,
        OperationStatus.RETRYING,
    }:
        previous.run_id = run_id
        store.transition(previous, OperationStatus.RETRYING)
        return previous
    record = OperationRecord.create(
        run_id=run_id,
        source=source,
        source_kind=source_kind,
    )
    store.save(record)
    return record


def _serialize_ingest(func: Any) -> Any:
    """Prevent two ingest processes from racing extraction and dedup writes."""

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        arguments = inspect.signature(func).bind_partial(*args, **kwargs).arguments
        if any(arguments.get(option, False) for option in ("plan", "dry_run", "preview")):
            return func(*args, **kwargs)
        raw_vault = arguments.get("vault")
        if raw_vault is None:
            return func(*args, **kwargs)
        vault_path, _config = resolve_vault(str(raw_vault))
        lock_file = vault_path / "04-Wiki" / ".llmwiki" / "ingest.lock"
        if not acquire_lock(lock_file):
            raise typer.BadParameter(
                "another ingest process is already active for this vault"
            )
        try:
            return func(*args, **kwargs)
        finally:
            release_lock(lock_file)

    return wrapped


@app.command()
@_serialize_ingest
def ingest(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    urls: list[str] | None = typer.Option(
        None, "--url", "-u", help="URLs to ingest (can be repeated)"
    ),
    parallel: int = typer.Option(
        3,
        "--parallel",
        "-p",
        help="Bounded concurrent extraction and synthesis requests",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Deprecated alias for --plan"),
    plan: bool = typer.Option(
        False, "--plan", help="List sources without network access or writes"
    ),
    preview: bool = typer.Option(
        False, "--preview", help="Extract sources but do not write or synthesise"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit NDJSON lifecycle and per-source events"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Retry sources from the last cancellation marker"
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-extract known URLs; back up changed source revisions"
    ),
    skip_synthesis: bool = typer.Option(
        False,
        "--skip-synthesis",
        help="Only extract sources; skip LLM synthesis and rendering",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override the default and ingest LLM model for this run"
    ),
) -> None:
    """Ingest URLs and clippings, then synthesise + render the vault.

    ``--plan`` and backwards-compatible ``--dry-run`` are local-only source
    inventories. ``--preview`` performs real extraction while keeping the vault
    unchanged. ``--json`` writes newline-delimited JSON events, one terminal
    source event per source, for tooling and the local Obsidian bridge.

    Clippings from ``02-Clippings/`` are moved to ``02-Clippings/processed/``
    after successful ingestion. Duplicate sources (same content hash) are
    detected and skipped.
    """
    plan_mode = plan or dry_run
    if plan_mode and preview:
        raise typer.BadParameter("--plan cannot be combined with --preview or --dry-run")
    if resume and (plan_mode or preview):
        raise typer.BadParameter("--resume cannot be combined with --plan, --dry-run or --preview")

    preview_mode = preview
    vault_path, config = resolve_vault(vault)
    if parallel:
        config = replace(config, compile_concurrency=parallel)
    if model:
        config = replace(config, llm=replace(config.llm, model=model, ingest_model=model))

    run_id = _run_id()
    mode = "plan" if plan_mode else "preview" if preview_mode else "ingest"
    outcome = _IngestOutcome(run_id, mode, json_output)
    _emit(
        json_output,
        {"type": "run", "event": "started", "run_id": run_id, "mode": mode},
        f"📂 Vault: {vault_path}\n🤖 Model: {config.llm.ingest_model or config.llm.model}",
    )
    try:
        from obsidian_llm_wiki.core.generated_paths import checked_metadata_directory

        # Resume readers precede SQLite; validate before they touch marker/lock
        # files and keep rejection within the terminal NDJSON error boundary.
        checked_metadata_directory(config.llmwiki_dir, config.vault)
        store = OperationStore(config.llmwiki_dir)
        requested_urls = list(dict.fromkeys(urls or []))
        resume_sources = store.read_resume_marker() if resume else []
        interrupted_operations = store.interrupted_ingest_operations() if resume else []
        if interrupted_operations:
            resumed_keys = {(item["source_kind"], item["source"]) for item in resume_sources}
            for operation in interrupted_operations:
                key = (operation.source_kind, operation.source)
                if key not in resumed_keys:
                    resume_sources.append({
                        "source": operation.source,
                        "source_kind": operation.source_kind,
                        "status": operation.status.value,
                    })
                    resumed_keys.add(key)
            with VaultRegistry(config.llmwiki_dir) as registry:
                for previous_run_id in {
                    operation.run_id for operation in interrupted_operations if operation.run_id
                }:
                    registry.release_dirty_worker(previous_run_id)
    except (OSError, ValueError) as exc:
        outcome.inventory_errors.append(f"Metadata inventory failed: {exc}")
        outcome.finish(0)
        return

    # ── Collect clippings ──────────────────────────────────────────────
    from obsidian_llm_wiki.ingest.clippings import collect_clippings

    # ── Scan 00-Inbox for .url files ───────────────────────────────────
    # The 00-Inbox directory is the URL queue: each .url file contains
    # one URL on the first line. These are merged with explicit --url
    # args and deduplicated.
    try:
        # Validate every existing source before backfill, indexed duplicate/
        # refresh reads, or any extraction. Never trust the backfill marker as
        # evidence that today's inventory is still contained.
        source_inventory = _ingest_source_paths(config.sources_dir)
        source_paths = set(source_inventory)
        inbox_urls = _scan_inbox_urls(config)
        inbox_url_paths = _inbox_url_paths(config)
        clipping_sources = collect_clippings(config)
    except (OSError, ValueError) as exc:
        outcome.inventory_errors.append(f"Source inventory failed: {exc}")
        outcome.finish(0)
        return
    if inbox_urls and not json_output:
        typer.echo(f"📥 Found {len(inbox_urls)} URL(s) in 00-Inbox/")
    all_urls = list(dict.fromkeys([*requested_urls, *inbox_urls]))

    clipping_by_path = {str(path): source for path, source in clipping_sources}
    planned: list[_PlannedSource]
    if resume:
        planned = [
            (
                marker["source_kind"],
                marker["source"],
                clipping_by_path.get(marker["source"])
                if marker["source_kind"] == "clipping"
                else None,
                True,
            )
            for marker in resume_sources
        ]
        resumed_keys = {
            (marker["source_kind"], marker["source"]) for marker in resume_sources
        }
        planned.extend(
            ("url", url, None, False)
            for url in all_urls
            if ("url", url) not in resumed_keys
        )
    else:
        planned = [
            ("clipping", str(path), source, False) for path, source in clipping_sources
        ]
        planned.extend(("url", url, None, False) for url in all_urls)

    if plan_mode:
        for source_kind, identifier, source, _resuming in planned:
            details: dict[str, Any] = {"run_id": run_id}
            if source is not None:
                bounded_source, truncated = _bounded_source(
                    source, config.max_source_chars
                )
                details.update(
                    title=bounded_source.title,
                    bytes=len(bounded_source.content.encode("utf-8")),
                    truncated=truncated,
                )
            _emit(
                json_output,
                _source_event(
                    identifier, source_kind, OperationStatus.PLANNED, **details
                ),
                f"   🔍 Would extract: {identifier}",
            )
        _emit(
            json_output,
            {"type": "run", "event": "completed", "run_id": run_id, "mode": mode},
            "\n   🔍 Plan complete — no network access or files written.",
        )
        return

    if not preview_mode:
        try:
            with VaultRegistry(config.llmwiki_dir) as ingest_registry:
                _backfill_ingest_index(
                    ingest_registry, config.sources_dir, source_paths=source_inventory
                )
        except (OSError, ValueError) as exc:
            outcome.inventory_errors.append(f"Source inventory failed during backfill: {exc}")
            outcome.finish(0)
            return

    known_url_duplicates: dict[str, tuple[Path, str, int]] = {}
    refresh_targets: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    if not preview_mode:
        with VaultRegistry(config.llmwiki_dir) as ingest_registry:
            for source_kind, identifier, _source, _resuming in planned:
                if source_kind != "url":
                    continue
                match = ingest_registry.find_ingest_duplicate(
                    requested_url=identifier
                )
                if match is None:
                    continue
                path = _indexed_source_path(
                    config.sources_dir, match.output_file, source_paths=source_paths
                )
                if path is None:
                    continue
                try:
                    raw = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    outcome.inventory_errors.append(f"Source inventory failed for {path}: {exc}")
                    outcome.finish(0)
                    return
                metadata, body = parse_frontmatter(raw)
                if refresh:
                    refresh_targets[identifier] = (path, raw, metadata)
                    continue
                current_hash = hashlib.sha256(_dedupe_body(body).encode("utf-8")).hexdigest()
                if current_hash != match.content_hash:
                    # A human edit or refreshed shared page invalidates the
                    # old URL alias as well as its cached content identity.
                    continue
                known_url_duplicates[identifier] = (
                    path,
                    str(metadata.get("title", path.stem)),
                    len(body.encode("utf-8")),
                )

    new_count = 0
    failed_urls: list[tuple[str, str]] = []
    processed_clipping_paths: list[Path] = []
    cancellation = _CancellationToken()
    planned_urls = [
        identifier
        for source_kind, identifier, _source, _resuming in planned
        if source_kind == "url" and identifier not in known_url_duplicates
    ]
    with (
        extraction_policy(allow_writes=not preview_mode),
        _cooperative_sigint(cancellation),
        _BoundedURLExtractor(planned_urls, config, parallel) as url_extractor,
    ):
        for index, (source_kind, identifier, clipped_source, resuming_source) in enumerate(
            planned
        ):
            if cancellation.cancelled:
                _cancel_remaining(
                    store,
                    run_id,
                    planned[index:],
                    json_output,
                    persist=not preview_mode,
                )
                raise typer.Exit(code=130)

            record: OperationRecord | None = None
            queue_id: int | None = None
            queue_claimed = True
            if not preview_mode:
                record = _operation_for_source(
                    store,
                    run_id=run_id,
                    source=identifier,
                    source_kind=source_kind,
                    resume=resuming_source,
                )
                store.transition(record, OperationStatus.RUNNING)
                with VaultRegistry(config.llmwiki_dir) as ingest_registry:
                    queue_id = ingest_registry.enqueue_dirty(
                        "ingest",
                        identifier,
                        "requested",
                        {"source_kind": source_kind, "run_id": run_id},
                    )
                    queue_claimed = ingest_registry.start_dirty(
                        queue_id, worker_id=run_id
                    )

            try:
                if not queue_claimed:
                    raise RuntimeError("ingest queue item has an active lease")
                known_duplicate = known_url_duplicates.get(identifier)
                if source_kind == "url" and known_duplicate is not None:
                    duplicate_path, duplicate_title, duplicate_size = known_duplicate
                    if queue_id is not None:
                        with VaultRegistry(config.llmwiki_dir) as ingest_registry:
                            ingest_registry.complete_dirty(queue_id, worker_id=run_id)
                    if record is not None:
                        store.transition(
                            record,
                            OperationStatus.SUCCEEDED,
                            title=duplicate_title,
                            bytes_extracted=duplicate_size,
                            output_file=duplicate_path.name,
                        )
                    _archive_inbox_url_files(
                        inbox_url_paths.pop(identifier, []), config
                    )
                    _emit(
                        json_output,
                        _source_event(
                            identifier,
                            source_kind,
                            OperationStatus.SUCCEEDED,
                            run_id=run_id,
                            title=duplicate_title,
                            bytes=duplicate_size,
                            output_file=duplicate_path.name,
                            duplicate_of=duplicate_path.name,
                        ),
                        f"   ⏭ {duplicate_title[:60]} — already indexed",
                    )
                    outcome.succeeded += 1
                    continue
                if source_kind == "clipping":
                    if clipped_source is None:
                        raise OSError(f"Clipping is missing or unreadable: {identifier}")
                    source = clipped_source
                else:
                    source = url_extractor.extract(identifier)
                source, truncated = _bounded_source(source, config.max_source_chars)
                size = len(source.content.encode("utf-8"))
                output_file = ""
                if not preview_mode:
                    from obsidian_llm_wiki.render.obsidian import render_source_page

                    config.sources_dir.mkdir(parents=True, exist_ok=True)

                    # ── Content-hash dedup ──────────────────────────────
                    # Before writing a new source file, check whether an
                    # existing source already has the same content hash.
                    # This catches both same-URL re-extraction and
                    # different-URL-same-content duplicates.
                    from obsidian_llm_wiki.core.state import hash_content

                    _, incoming_body = parse_frontmatter(render_source_page(source))
                    incoming_text = _dedupe_body(incoming_body)
                    content_hash = hash_content(incoming_text)
                    with VaultRegistry(config.llmwiki_dir) as ingest_registry:
                        duplicate = ingest_registry.find_ingest_duplicate(
                            requested_url=identifier if source_kind == "url" else "",
                            content_hash=content_hash,
                        )
                    dup_path = (
                        _indexed_source_path(
                            config.sources_dir, duplicate.output_file, source_paths=source_paths
                        )
                        if duplicate is not None
                        else None
                    )
                    if dup_path is not None and not dup_path.is_file():
                        dup_path = None
                    if dup_path is not None:
                        # The indexed file may have been edited since ingestion.
                        # Verify just this candidate, never discard evidence on
                        # the strength of a stale hash or file existence alone.
                        try:
                            _, existing_body = parse_frontmatter(
                                dup_path.read_text(encoding="utf-8")
                            )
                            existing_text = _dedupe_body(existing_body)
                            if existing_text != incoming_text:
                                dup_path = None
                        except (OSError, UnicodeError):
                            dup_path = None
                    refresh_target = refresh_targets.get(identifier)
                    replacing = False
                    if refresh_target is not None:
                        target, previous_text, previous_meta = refresh_target
                        # A dedup alias does not own another URL's evidence page.
                        from obsidian_llm_wiki.ingest.sources import load_source_file

                        previous_source = load_source_file(target)
                        replacing = previous_meta.get("url") == (source.url or identifier) or (
                            previous_source is not None
                            and previous_source.provenance.requested_url == identifier
                        )
                    # Explicit refresh of an owned source also refreshes metadata:
                    # equal body text must not discard a title/provenance revision.
                    if dup_path is not None and not replacing:
                        with VaultRegistry(config.llmwiki_dir) as ingest_registry:
                            ingest_registry.upsert_ingest_hash(
                                requested_url=identifier,
                                canonical_url=source.url or identifier,
                                content_hash=content_hash,
                                output_file=dup_path.name,
                            )
                            if queue_id is not None:
                                ingest_registry.complete_dirty(
                                    queue_id, worker_id=run_id
                                )
                        _emit(
                            json_output,
                            _source_event(
                                identifier,
                                source_kind,
                                OperationStatus.SUCCEEDED,
                                run_id=run_id,
                                title=source.title,
                                bytes=size,
                                output_file=dup_path.name,
                                truncated=truncated,
                                preview=preview_mode,
                                duplicate_of=dup_path.name,
                            ),
                            f"   ⏭ {source.title[:60]} — duplicate of {dup_path.name}",
                        )
                        if record is not None:
                            store.transition(
                                record,
                                OperationStatus.SUCCEEDED,
                                title=source.title,
                                bytes_extracted=size,
                                output_file=dup_path.name,
                            )
                            if resuming_source:
                                store.remove_resume_source(identifier, source_kind)
                        # Archive the clipping even though it's a duplicate —
                        # it's been "processed" (just found to be a dup).
                        if source_kind == "clipping":
                            clipping_path = Path(identifier)
                            if clipping_path.exists():
                                archive_dest = _archive_clipping(clipping_path, config)
                                if archive_dest is not None:
                                    processed_clipping_paths.append(archive_dest)
                        elif source_kind == "url":
                            _archive_inbox_url_files(inbox_url_paths.pop(identifier, []), config)
                        outcome.succeeded += 1
                        continue

                    if replacing:
                        from obsidian_llm_wiki.core.backups import backup_file
                        from obsidian_llm_wiki.core.review import is_reviewed_page

                        assert refresh_target is not None
                        target, previous_text, _previous_meta = refresh_target
                        if is_reviewed_page(previous_text):
                            raise RuntimeError("Refusing to refresh a reviewed source")
                        if truncated:
                            raise RuntimeError(
                                "Refusing to replace source with truncated extraction"
                            )
                        if target.read_text(encoding="utf-8") != previous_text:
                            raise RuntimeError("Source changed during refresh; retry after editing")
                        backup_file(target, config.llmwiki_dir / "backups")
                        filepath = target
                    else:
                        filepath = _reserve_collision_safe_path(
                            config.sources_dir,
                            source,
                            preferred_stem=(
                                Path(identifier).stem if source_kind == "clipping" else None
                            ),
                        )
                    try:
                        atomic_write(filepath, render_source_page(source))
                    except BaseException:
                        if not replacing:
                            filepath.unlink(missing_ok=True)
                        raise
                    output_file = filepath.name
                    source_paths.add(filepath)
                    with VaultRegistry(config.llmwiki_dir) as ingest_registry:
                        ingest_registry.upsert_ingest_hash(
                            requested_url=identifier,
                            canonical_url=source.url or identifier,
                            content_hash=content_hash,
                            output_file=output_file,
                        )
                        if queue_id is not None:
                            ingest_registry.complete_dirty(queue_id, worker_id=run_id)
                    assert record is not None
                    store.transition(
                        record,
                        OperationStatus.SUCCEEDED,
                        title=source.title,
                        bytes_extracted=size,
                        output_file=output_file,
                    )
                    if resuming_source:
                        store.remove_resume_source(identifier, source_kind)

                    # ── Archive processed clippings ────────────────────
                    # Move successfully ingested clippings to
                    # 02-Clippings/processed/ so they're not re-processed
                    # on the next run.
                    if source_kind == "clipping":
                        clipping_path = Path(identifier)
                        if clipping_path.exists():
                            archive_dest = _archive_clipping(clipping_path, config)
                            if archive_dest is not None:
                                processed_clipping_paths.append(archive_dest)
                    elif source_kind == "url":
                        _archive_inbox_url_files(inbox_url_paths.pop(identifier, []), config)
                new_count += 1
                outcome.succeeded += 1
                _emit(
                    json_output,
                    _source_event(
                        identifier,
                        source_kind,
                        OperationStatus.SUCCEEDED,
                        run_id=run_id,
                        title=source.title,
                        bytes=size,
                        output_file=output_file,
                        truncated=truncated,
                        preview=preview_mode,
                    ),
                    f"   ✅ {source.title[:60]} ({size} bytes)",
                )
            except KeyboardInterrupt:
                if record is not None:
                    store.transition(record, OperationStatus.CANCELLED, error="interrupted")
                if queue_id is not None:
                    with VaultRegistry(config.llmwiki_dir) as ingest_registry:
                        ingest_registry.fail_dirty(
                            queue_id,
                            "interrupted",
                            worker_id=run_id,
                        )
                _cancel_remaining(
                    store,
                    run_id,
                    planned[index:],
                    json_output,
                    already_cancelled=identifier if record is not None else "",
                    persist=not preview_mode,
                )
                raise typer.Exit(code=130) from None
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                outcome.extraction_errors.append(f"{source_kind}:{identifier}: {error}")
                if record is not None:
                    store.transition(record, OperationStatus.FAILED, error=error)
                if queue_id is not None:
                    with VaultRegistry(config.llmwiki_dir) as ingest_registry:
                        ingest_registry.fail_dirty(
                            queue_id,
                            error,
                            worker_id=run_id,
                        )
                if source_kind == "url":
                    failed_urls.append((identifier, error))
                _emit(
                    json_output,
                    _source_event(
                        identifier,
                        source_kind,
                        OperationStatus.FAILED,
                        run_id=run_id,
                        error=error,
                    ),
                    f"   ❌ {identifier}: {error}",
                )

    # A cooperative SIGINT during the final source has no next loop iteration
    # at which to stop. Preserve that completed source, but never report success
    # or start synthesis after cancellation.
    if cancellation.cancelled:
        _cancel_remaining(store, run_id, [], json_output, persist=False)
        raise typer.Exit(code=130)

    # Report archived clippings.
    if processed_clipping_paths and not json_output:
        typer.echo(
            f"\n   📁 Archived {len(processed_clipping_paths)} clipping(s) to "
            f"02-Clippings/processed/"
        )

    if not preview_mode and failed_urls:
        ledger_path = _update_failed_ledger(config.llmwiki_dir, failed_urls)
        _emit(
            json_output,
            {"type": "ledger", "event": "updated", "path": str(ledger_path)},
            f"\n   📋 Updated {ledger_path} ({len(failed_urls)} failed URL(s) this run)",
        )

    if preview_mode:
        outcome.finish(
            new_count,
            "\n   🔍 Preview complete — no files written.",
        )
        return

    # Known URLs still need the pipeline: a prior extraction can have succeeded
    # while synthesis failed, or source metadata/cache compatibility changed.
    if not planned:
        outcome.finish(
            0,
            "\n⚠ No new sources to ingest.\n"
            "   Tip: Use --url to add URLs or add .md files to 02-Clippings/",
        )
        return

    if skip_synthesis:
        outcome.finish(
            new_count,
            "\n   ⏭ Skipping synthesis (--skip-synthesis)",
            synthesis="skipped",
        )
        return

    try:
        with VaultRegistry(config.llmwiki_dir) as source_registry:
            source_load = load_sources_incremental(config.sources_dir, source_registry)
    except (OSError, ValueError) as exc:
        outcome.inventory_errors.append(f"Source inventory failed; no synthesis performed: {exc}")
        outcome.finish(new_count)
        return
    full_corpus = source_load.sources
    if not full_corpus:
        outcome.finish(
            new_count,
            "\n⚠ No source files found in sources/.",
        )
        return

    _emit(
        json_output,
        {
            "type": "pipeline",
            "event": "started",
            "run_id": run_id,
            "sources": len(full_corpus),
        },
        f"\n📦 Total corpus: {len(full_corpus)} source(s)\n\n🤖 Running LLM synthesis pipeline...",
    )
    from obsidian_llm_wiki.core.pipeline import run_pipeline

    try:
        result = asyncio.run(
            run_pipeline(
                vault_path,
                full_corpus,
                config,
                unchanged_filenames=source_load.unchanged_filenames,
                source_file_signatures=source_load.signatures,
            )
        )
    except KeyboardInterrupt:
        _emit(
            json_output,
            {"type": "run", "event": "cancelled", "run_id": run_id, "remaining": 0},
            "\n   ⏹ Cancelled during synthesis; extracted sources remain durable.",
        )
        raise typer.Exit(code=130) from None
    except Exception as exc:
        outcome.synthesis_errors.append(f"Pipeline failed: {str(exc) or type(exc).__name__}")
        outcome.finish(new_count)
        return

    outcome.synthesis_errors.extend(result.errors)
    if not json_output:
        print_result_summary(result, additional_errors=outcome.extraction_errors)
    outcome.finish(new_count, compiled=result.compiled)


def _cancel_remaining(
    store: OperationStore,
    run_id: str,
    remaining: list[_PlannedSource],
    json_output: bool,
    *,
    already_cancelled: str = "",
    persist: bool = True,
) -> None:
    resume_sources = [
        {
            "source": identifier,
            "source_kind": source_kind,
            "status": OperationStatus.CANCELLED.value,
        }
        for source_kind, identifier, _source, _resuming in remaining
    ]
    if persist:
        store.write_resume_marker(run_id, resume_sources)
    for source_kind, identifier, _source, _resuming in remaining:
        if persist and identifier != already_cancelled:
            record = OperationRecord.create(
                run_id=run_id,
                source=identifier,
                source_kind=source_kind,
            )
            store.save(record)
            store.transition(record, OperationStatus.CANCELLED, error="cancelled before extraction")
        _emit(
            json_output,
            _source_event(identifier, source_kind, OperationStatus.CANCELLED, run_id=run_id),
            f"   ⏹ Cancelled: {identifier}",
        )
    _emit(
        json_output,
        {
            "type": "run",
            "event": "cancelled",
            "run_id": run_id,
            "remaining": len(resume_sources),
        },
        "\n   ⏹ Cancelled. Resume with --resume.",
    )


def _update_failed_ledger(sources_dir: Path, new_failures: list[tuple[str, str]]) -> Path:
    """Append failed URLs to the failed_urls.md ledger in .llmwiki/."""
    ledger_path = sources_dir / "failed_urls.md"
    ts = datetime.now(UTC).strftime("%Y-%m-%d")
    existing: dict[str, str] = {}
    if ledger_path.exists():
        text = ledger_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("|") and "---" not in line and "Date" not in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    existing[parts[2].strip()] = parts[3].strip()
    for url, error in new_failures:
        existing[url] = error
    rows = [
        f"| {ts} | {url} | {error[:120].replace(chr(10), ' ')} |"
        for url, error in existing.items()
    ]
    content = LEDGER_TEMPLATE.format(
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows="\n".join(rows),
    )
    atomic_write(ledger_path, content)
    return ledger_path


def _scan_inbox_urls(config: Any) -> list[str]:
    """Scan 00-Inbox/ for .url files and return the URLs they contain.

    Each .url file should contain a single URL on the first non-empty line.
    Files are not deleted — the caller decides whether to archive them.
    """
    inbox_dir = config.vault / "00-Inbox"
    if not inbox_dir.is_dir():
        return []

    urls: list[str] = []
    for f in sorted(inbox_dir.iterdir()):
        if f.suffix != ".url" or not f.is_file():
            continue
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                url = line.strip()
                if url and url.startswith(("http://", "https://")):
                    urls.append(url)
                    break
        except OSError as exc:
            raise OSError(f"Cannot read inbox record {f}: {exc}") from exc
        except UnicodeError as exc:
            logger.warning("Skipping unreadable inbox record %s: %s", f, exc)
            continue

    return urls


def _inbox_url_paths(config: Any) -> dict[str, list[Path]]:
    """Map queued URLs to their source files so only successes are archived."""
    inbox_dir = config.vault / "00-Inbox"
    if not inbox_dir.is_dir():
        return {}
    paths: dict[str, list[Path]] = {}
    for path in sorted(inbox_dir.iterdir()):
        if path.suffix != ".url" or not path.is_file():
            continue
        try:
            url = next(
                (
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip().startswith(("http://", "https://"))
                ),
                "",
            )
        except OSError as exc:
            raise OSError(f"Cannot read inbox record {path}: {exc}") from exc
        except UnicodeError:
            continue
        if url:
            paths.setdefault(url, []).append(path)
    return paths


def _archive_inbox_url_files(paths: list[Path], config: Any) -> list[Path]:
    """Archive successfully ingested 00-Inbox URL records without reprocessing them."""
    if not paths:
        return []
    archive_dir = config.vault / "00-Inbox" / "processed"
    archived: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        destination = _archive_without_overwrite(path, archive_dir)
        if destination is not None:
            archived.append(destination)
    return archived
