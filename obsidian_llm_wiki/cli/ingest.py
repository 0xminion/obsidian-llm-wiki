"""``olw ingest`` — extract URLs + collect clippings → write source files."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import print_result_summary, resolve_vault
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.operations import OperationRecord, OperationStatus, OperationStore
from obsidian_llm_wiki.ingest.sources import load_sources_from_dir
from obsidian_llm_wiki.render.obsidian import atomic_write, slugify

_PlannedSource = tuple[str, str, SourceDoc | None, bool]

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
    """Limit extracted UTF-8 content before any renderer or synthesis can use it."""
    if maximum_bytes <= 0:
        truncated = bool(source.content)
        bounded_content = ""
    else:
        encoded = source.content.encode("utf-8")
        if len(encoded) <= maximum_bytes:
            return source, False
        # Decode only complete UTF-8 characters, enforcing the byte boundary rather
        # than merely counting Python characters (important for non-ASCII sources).
        bounded_content = encoded[:maximum_bytes].decode("utf-8", errors="ignore")
        truncated = True
    if not truncated:
        return source, False
    provenance = replace(
        source.provenance,
        content_sha256=hashlib.sha256(bounded_content.encode("utf-8")).hexdigest(),
        diagnostics=(
            *source.provenance.diagnostics,
            f"content truncated to {len(bounded_content.encode('utf-8'))} UTF-8 bytes "
            f"(limit {maximum_bytes})",
        ),
    )
    return replace(source, content=bounded_content, provenance=provenance), True


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
    if previous and previous.status in {OperationStatus.CANCELLED, OperationStatus.FAILED}:
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


@app.command()
def ingest(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    urls: list[str] | None = typer.Option(
        None, "--url", "-u", help="URLs to ingest (can be repeated)"
    ),
    parallel: int = typer.Option(
        3, "--parallel", "-p", help="Concurrent LLM calls during synthesis"
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
    skip_synthesis: bool = typer.Option(
        False,
        "--skip-synthesis",
        help="Only extract sources; skip LLM synthesis and rendering",
    ),
) -> None:
    """Ingest URLs and clippings, then synthesise + render the vault.

    ``--plan`` and backwards-compatible ``--dry-run`` are local-only source
    inventories. ``--preview`` performs real extraction while keeping the vault
    unchanged. ``--json`` writes newline-delimited JSON events, one terminal
    source event per source, for tooling and the local Obsidian bridge.
    """
    plan_mode = plan or dry_run
    if plan_mode and preview:
        raise typer.BadParameter("--plan cannot be combined with --preview or --dry-run")

    preview_mode = preview
    vault_path, config = resolve_vault(vault)
    if parallel:
        os.environ["COMPILE_CONCURRENCY"] = str(parallel)
        config = _reload_config_with_concurrency(vault_path, parallel)

    store = OperationStore(config.llmwiki_dir)
    requested_urls = list(dict.fromkeys(urls or []))
    resume_sources = store.read_resume_marker() if resume else []
    run_id = _run_id()
    mode = "plan" if plan_mode else "preview" if preview_mode else "ingest"
    _emit(
        json_output,
        {"type": "run", "event": "started", "run_id": run_id, "mode": mode},
        f"📂 Vault: {vault_path}\n🤖 Model: {config.llm.model}",
    )

    # Reading clipping files is local-only and is allowed in plan/preview.
    from obsidian_llm_wiki.ingest.clippings import collect_clippings

    clipping_sources = collect_clippings(config)
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
        resumed_keys = {(marker["source_kind"], marker["source"]) for marker in resume_sources}
        planned.extend(
            ("url", url, None, False)
            for url in requested_urls
            if ("url", url) not in resumed_keys
        )
    else:
        planned = [("clipping", str(path), source, False) for path, source in clipping_sources]
        planned.extend(("url", url, None, False) for url in requested_urls)

    if plan_mode:
        for source_kind, identifier, source, _resuming in planned:
            details: dict[str, Any] = {"run_id": run_id}
            if source is not None:
                bounded_source, truncated = _bounded_source(source, config.max_source_chars)
                details.update(
                    title=bounded_source.title,
                    bytes=len(bounded_source.content.encode("utf-8")),
                    truncated=truncated,
                )
            _emit(
                json_output,
                _source_event(identifier, source_kind, OperationStatus.PLANNED, **details),
                f"   🔍 Would extract: {identifier}",
            )
        _emit(
            json_output,
            {"type": "run", "event": "completed", "run_id": run_id, "mode": mode},
            "\n   🔍 Plan complete — no network access or files written.",
        )
        return

    new_count = 0
    failed_urls: list[tuple[str, str]] = []
    cancellation = _CancellationToken()
    with _cooperative_sigint(cancellation):
        for index, (source_kind, identifier, clipped_source, resuming_source) in enumerate(planned):
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
            if not preview_mode:
                record = _operation_for_source(
                    store,
                    run_id=run_id,
                    source=identifier,
                    source_kind=source_kind,
                    resume=resuming_source,
                )
                store.transition(record, OperationStatus.RUNNING)

            try:
                if source_kind == "clipping":
                    assert clipped_source is not None
                    source = clipped_source
                else:
                    from obsidian_llm_wiki.ingest.extractors import extract

                    source = extract(identifier)
                source, truncated = _bounded_source(source, config.max_source_chars)
                size = len(source.content.encode("utf-8"))
                output_file = ""
                if not preview_mode:
                    from obsidian_llm_wiki.render.obsidian import render_source_page

                    config.sources_dir.mkdir(parents=True, exist_ok=True)
                    filepath = (
                        config.sources_dir / Path(identifier).name
                        if source_kind == "clipping"
                        else _collision_safe_path(config.sources_dir, source)
                    )
                    atomic_write(filepath, render_source_page(source))
                    output_file = filepath.name
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
                new_count += 1
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
                error = str(exc)
                if record is not None:
                    store.transition(record, OperationStatus.FAILED, error=error)
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

    if not preview_mode and failed_urls:
        ledger_path = _update_failed_ledger(config.sources_dir, failed_urls)
        _emit(
            json_output,
            {"type": "ledger", "event": "updated", "path": str(ledger_path)},
            f"\n   📋 Updated {ledger_path} ({len(failed_urls)} failed URL(s) this run)",
        )

    if preview_mode:
        _emit(
            json_output,
            {
                "type": "run",
                "event": "completed",
                "run_id": run_id,
                "mode": mode,
                "sources": new_count,
            },
            "\n   🔍 Preview complete — no files written.",
        )
        return

    if new_count == 0 and not failed_urls:
        _emit(
            json_output,
            {
                "type": "run",
                "event": "completed",
                "run_id": run_id,
                "mode": mode,
                "sources": 0,
            },
            "\n⚠ No new sources to ingest.\n"
            "   Tip: Use --url to add URLs or add .md files to 02-Clippings/",
        )
        return

    if skip_synthesis:
        _emit(
            json_output,
            {
                "type": "run",
                "event": "completed",
                "run_id": run_id,
                "mode": mode,
                "sources": new_count,
                "synthesis": "skipped",
            },
            "\n   ⏭ Skipping synthesis (--skip-synthesis)",
        )
        return

    full_corpus = load_sources_from_dir(config.sources_dir)
    if not full_corpus:
        _emit(
            json_output,
            {
                "type": "run",
                "event": "completed",
                "run_id": run_id,
                "mode": mode,
                "sources": new_count,
            },
            "\n⚠ No source files found in sources/.",
        )
        return

    _emit(
        json_output,
        {"type": "pipeline", "event": "started", "run_id": run_id, "sources": len(full_corpus)},
        f"\n📦 Total corpus: {len(full_corpus)} source(s)\n\n🤖 Running LLM synthesis pipeline...",
    )
    from obsidian_llm_wiki.core.pipeline import run_pipeline

    try:
        result = asyncio.run(run_pipeline(vault_path, full_corpus, config))
    except KeyboardInterrupt:
        _emit(
            json_output,
            {"type": "run", "event": "cancelled", "run_id": run_id, "remaining": 0},
            "\n   ⏹ Cancelled during synthesis; extracted sources remain durable.",
        )
        raise typer.Exit(code=130) from None

    if json_output:
        _emit(
            True,
            {
                "type": "run",
                "event": "completed",
                "run_id": run_id,
                "mode": mode,
                "sources": new_count,
                "compiled": result.compiled,
                "errors": result.errors,
            },
        )
    else:
        print_result_summary(result)


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
    """Append failed URLs to the failed_urls.md ledger."""
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
        f"| {ts} | {url} | {error[:120].replace(chr(10), ' ')} |" for url, error in existing.items()
    ]
    content = LEDGER_TEMPLATE.format(
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows="\n".join(rows),
    )
    atomic_write(ledger_path, content)
    return ledger_path


def _reload_config_with_concurrency(vault_path: Path, parallel: int):
    """Reload config with concurrency override."""
    from obsidian_llm_wiki.config import load_config

    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    return load_config(
        env_file=env_file,
        VAULT_PATH=str(vault_path),
        COMPILE_CONCURRENCY=str(parallel),
    )
