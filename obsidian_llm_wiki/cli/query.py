"""``olw query`` — grounded, session-aware answers against a wiki vault."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.core.query_freshness import (
    eligible_graph,
    page_diagnostics,
    require_current_inputs,
)
from obsidian_llm_wiki.providers.llm import call_llm
from obsidian_llm_wiki.query.context import build_context, extract_cited_paths, validate_citations
from obsidian_llm_wiki.query.graph import build_graph_from_vault, publication_read_token
from obsidian_llm_wiki.query.profiles import MAX_PROFILE_INSTRUCTIONS, QueryProfileStore
from obsidian_llm_wiki.query.retrieval import RetrievalTrace, retrieve
from obsidian_llm_wiki.query.sessions import QuerySessionStore, create_session

_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")


@app.command()
def query(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    ask: str = typer.Option(..., "--ask", "-a", help="Question to ask"),
    max_results: int = typer.Option(
        10, "--max-results", "-n", help="Max pages to retrieve"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable result"),
    session: str = typer.Option("", "--session", help="Continue or name a local query session"),
    instructions: str = typer.Option("", "--instructions", help="Extra query-only instructions"),
    profile: str = typer.Option("default", "--profile", help="Vault-local query profile"),
    save_answer: str | None = typer.Option(
        None,
        "--save-answer",
        help="Save the answer to a Markdown file (defaults to 05-Queries/)",
    ),
    force: bool = typer.Option(False, "--force", help="Deprecated; cannot replace existing files"),
):
    """Answer a question from retrieved wiki context with grounded citations.

    Examples:
        olw query ~/MyVault --ask "What is a transformer model?"
        olw query ~/MyVault -a "Compare RAG vs fine-tuning" -n 5 --profile research
        olw query ~/MyVault -a "What did we decide?" --session planning
    """
    _vault_path, config = resolve_vault(vault)
    session_id = session.strip() or str(uuid.uuid4())
    from obsidian_llm_wiki.core.immutable_publication import current_publication, publication_root
    try:
        committed = current_publication(config.wiki_dir)
        if committed is None:
            publication_read_token(config.wiki_dir)  # retain legacy pending/corrupt diagnostics
        freshness = require_current_inputs(config, committed)
        generation = freshness['generation']
    except ImportError as exc:
        _fail(str(exc), json_output=json_output, session_id=session_id,
              freshness={'status': 'unverifiable', 'changed_paths': [],
                         'reasons': ['Publication input inspector is unavailable']})
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc), json_output=json_output, session_id=session_id,
              freshness=getattr(exc, 'freshness', None))
    instructions = instructions.strip()[:MAX_PROFILE_INSTRUCTIONS]

    destination: Path | None = None
    if save_answer:
        destination = Path(save_answer).expanduser()
        # If the path is relative, resolve it under 05-Queries/.
        if not destination.is_absolute():
            destination = config.queries_dir / destination
        if destination.resolve().is_relative_to(publication_root(config.wiki_dir)):
            _fail('Refusing to write inside the immutable publication namespace',
                  json_output=json_output, session_id=session_id)
        from obsidian_llm_wiki.core.generated_paths import checked_generated_path
        try:
            destination = checked_generated_path(destination, config.queries_dir)
        except ValueError as exc:
            _fail('Refusing answer destination outside 05-Queries: ' + str(exc),
                  json_output=json_output, session_id=session_id)
        if destination.exists():
            _fail(
                (
                    f"Refusing to overwrite existing answer: {destination}. "
                    "Choose a new filename; --force cannot replace human files."
                ),
                json_output=json_output,
                session_id=session_id,
            )

    profile_store = QueryProfileStore(config.llmwiki_dir / "query-profiles.json")
    selected_profile = profile_store.load(profile)
    if selected_profile is None:
        _fail(
            f"Unknown query profile: {profile}",
            json_output=json_output,
            session_id=session_id,
        )

    try:
        inspected = build_graph_from_vault(config.wiki_dir, committed=committed, config=config)
        graph = eligible_graph(inspected)
        retrieval = retrieve(ask, graph, max_results=max_results)
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc), json_output=json_output, session_id=session_id,
              freshness=getattr(exc, 'freshness', {**freshness, 'status': 'unverifiable'}))

    trace = _trace_payload(retrieval.trace)
    trace['freshness'] = freshness
    trace['pages'] = page_diagnostics(inspected)
    sections = build_context(graph, retrieval.candidates, ask, max_chars_per_page=800)
    if not sections:
        rejected = sorted({reason for p in inspected.pages.values()
                           for reason in p.status_reasons})
        _fail('No eligible current evidence found. Historical, orphaned, unchecked and '
              'failed-retained pages are excluded. ' + '; '.join(rejected),
              json_output=json_output, session_id=session_id, retrieval_trace=trace,
              freshness={**freshness, 'status': 'no_eligible_evidence'})

    from obsidian_llm_wiki.core.application import operational_directory
    session_root = operational_directory(config, kind="query-state")
    session_store = QuerySessionStore(session_root / "query-sessions.json")
    previous = session_store.load(session_id) if session.strip() else None
    if previous is not None:
        prior = previous.retrieval_trace.get('freshness', {})
        reusable = (
            prior.get('generation') == generation and prior.get('status') == 'observed_current'
            and bool(previous.retrieved_paths)
            and set(previous.retrieved_paths) <= {section.path for section in sections}
        )
        trace['previous_session'] = 'same_captured_evidence' if reusable else 'excluded'
        if not reusable:
            previous = None
    system = _build_system(selected_profile.instructions, instructions, sections)
    messages = _build_messages(ask, previous)

    if not json_output:
        typer.echo(f'🔍 Searching for: "{ask}"')
        typer.echo(f"🤖 Model: {config.llm.model}")
        typer.echo(f"📄 Retrieved {len(sections)} relevant page(s)")
        typer.echo("\n💭 Thinking...\n")

    try:
        freshness = require_current_inputs(
            config, current_publication(config.wiki_dir), generation=generation,
        )
        raw_answer = call_llm(system, messages, config, max_tokens=2048, task="query")
    except Exception as exc:
        _fail(
            f"Query failed: {exc}",
            json_output=json_output,
            session_id=session_id,
            retrieval_trace=trace,
            freshness=getattr(exc, 'freshness', freshness),
        )

    answer, citations, errors = _ground_answer(raw_answer, [section.path for section in sections])
    if freshness['status'] == 'captured_only':
        answer += (f'\n\n> Captured generation: {generation}. No live source file is available '
                   'for some API-only inputs; this answer describes captured evidence only, '
                   'not verified current upstream information.')
    else:
        answer += (f'\n\n> Captured generation: {generation}. '
                   'Source/policy inputs matched at query observations; '
                   'not an atomic latest-at-commit guarantee.')
    excluded = [p for p in inspected.pages.values()
                if p.orphaned or p.publication_status == 'ineligible']
    if excluded:
        answer += ('\n> Excluded from this answer: orphaned or non-admitted retained content. '
                   'This answer covers only the eligible retrieved evidence, not those pages.')
    trace['freshness'] = freshness
    saved_session = create_session(
        session_id=session_id,
        query=ask,
        retrieved_paths=tuple(section.path for section in sections),
        retrieval_trace=trace,
        profile=selected_profile.name,
        instructions=instructions,
        answer=answer,
        citation_paths=tuple(citations),
    )
    # One acceptance observation after all preparation, before result/session/save.
    # Continuously editable human paths are NOT locked by this observation.
    try:
        freshness = require_current_inputs(
            config, current_publication(config.wiki_dir), generation=generation,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _fail(str(exc), json_output=json_output, session_id=session_id, retrieval_trace=trace,
              freshness=getattr(exc, 'freshness', {**freshness, 'status': 'unverifiable'}))
    if save_answer:
        assert destination is not None
        try:
            _save_answer(destination, answer)
        except OSError as exc:
            _fail(
                f"Could not save answer: {exc}",
                json_output=json_output,
                session_id=session_id,
                retrieval_trace=trace,
                freshness={**freshness, 'status': 'save_failed'},
            )

    operational_directory(config, create=True, kind="query-state")
    session_store.save(saved_session)

    if json_output:
        _emit_json(
            answer=answer,
            citations=citations,
            retrieval_trace=trace,
            session_id=session_id,
            errors=errors,
            freshness=freshness,
        )
        return

    typer.echo(answer)
    typer.echo(
        f"\n---\n*Answer based on {len(sections)} wiki page(s). Model: {config.llm.model}*"
    )
    if save_answer:
        assert destination is not None
        typer.echo(f"Saved answer: {destination}")


def _build_system(profile_instructions: str, instructions: str, sections: tuple) -> str:
    context = "\n\n".join(
        f"--- PAGE: {section.path} ({section.title}) ---\n"
        f"Lifecycle: {section.publication_status}; sources: {dict(section.source_statuses)}\n"
        f"{section.snippet}" for section in sections
    )
    extra_instructions = f"\nAdditional query instructions: {instructions}" if instructions else ""
    return (
        "You answer questions from a personal knowledge wiki. Use only the supplied context. "
        "If it is insufficient, say so. Cite pages only with exact "
        "[[vault-relative/path.md]] paths "
        "that appear in the supplied context.\n\n"
        f"Profile instructions: {profile_instructions}{extra_instructions}\n\n"
        f"--- KNOWLEDGE WIKI CONTEXT ---\n{context}\n--- END CONTEXT ---"
    )


def _build_messages(ask: str, previous: Any) -> list[dict[str, str]]:
    if previous is None or not previous.answer:
        return [{"role": "user", "content": ask}]
    return [
        {"role": "user", "content": previous.query},
        {"role": "assistant", "content": previous.answer},
        {"role": "user", "content": ask},
    ]


def _ground_answer(answer: str, retrieved_paths: list[str]) -> tuple[str, list[str], list[str]]:
    validation = validate_citations(answer, retrieved_paths)
    if validation.valid:
        return answer.strip(), list(validation.cited_paths), []

    uncited_answer = _WIKILINK_RE.sub("", answer).strip()
    references = "\n".join(f"- [[{path}]]" for path in retrieved_paths)
    grounded = f"{uncited_answer}\n\n## References\n{references}".strip()
    return grounded, list(extract_cited_paths(grounded)), [
        "Ungrounded citations were replaced with retrieved references."
    ]


def _save_answer(path: Path, answer: str) -> None:
    content = f"# Answer\n\n{answer.strip()}\n"
    # Exclusive complete-file installation; a concurrent human creation wins.
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".olw-answer-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        Path(temporary).unlink()


def _trace_payload(trace: RetrievalTrace) -> dict[str, Any]:
    return asdict(trace)


def _emit_json(
    *,
    answer: str,
    citations: list[str],
    retrieval_trace: dict[str, Any] | None,
    session_id: str,
    errors: list[str],
    freshness: dict[str, Any] | None = None,
) -> None:
    typer.echo(
        json.dumps(
            {
                "answer": answer,
                "citations": citations,
                "retrieval_trace": retrieval_trace,
                "session": session_id,
                "errors": errors,
                "freshness": freshness or {'status': 'unverifiable'},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _fail(
    message: str,
    *,
    json_output: bool,
    session_id: str,
    retrieval_trace: dict[str, Any] | None = None,
    freshness: dict[str, Any] | None = None,
) -> NoReturn:
    if retrieval_trace is not None:
        retrieval_trace = {**retrieval_trace,
                           'freshness': freshness or {'status': 'unverifiable'}}
    if json_output:
        _emit_json(
            answer="",
            citations=[],
            retrieval_trace=retrieval_trace,
            session_id=session_id,
            errors=[message],
            freshness=freshness,
        )
    else:
        typer.echo(f"❌ {message}", err=True)
    raise typer.Exit(code=1)
