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
from obsidian_llm_wiki.providers.llm import call_llm
from obsidian_llm_wiki.query.context import build_context, extract_cited_paths, validate_citations
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from obsidian_llm_wiki.query.profiles import MAX_PROFILE_INSTRUCTIONS, QueryProfileStore
from obsidian_llm_wiki.query.retrieval import RetrievalTrace, retrieve
from obsidian_llm_wiki.query.sessions import QuerySessionStore, create_session
from obsidian_llm_wiki.render.frontmatter import atomic_write

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
        None, "--save-answer", help="Explicit Markdown destination for the answer"
    ),
    force: bool = typer.Option(False, "--force", help="Allow --save-answer to replace a file"),
):
    """Answer a question from retrieved wiki context with grounded citations.

    Examples:
        olw query ~/MyVault --ask "What is a transformer model?"
        olw query ~/MyVault -a "Compare RAG vs fine-tuning" -n 5 --profile research
        olw query ~/MyVault -a "What did we decide?" --session planning
    """
    _vault_path, config = resolve_vault(vault)
    session_id = session.strip() or str(uuid.uuid4())
    instructions = instructions.strip()[:MAX_PROFILE_INSTRUCTIONS]

    if save_answer:
        destination = Path(save_answer).expanduser()
        if destination.exists() and not force:
            _fail(
                (
                    f"Refusing to overwrite existing answer: {destination}. "
                    "Pass --force to replace it."
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

    graph = build_graph_from_vault(config.wiki_dir)
    try:
        retrieval = retrieve(ask, graph, max_results=max_results)
    except ValueError as exc:
        _fail(str(exc), json_output=json_output, session_id=session_id)

    trace = _trace_payload(retrieval.trace)
    sections = build_context(graph, retrieval.candidates, ask, max_chars_per_page=800)
    if not sections:
        message = (
            "No relevant pages found in the wiki. "
            "Try ingesting some sources first with: olw ingest"
        )
        if json_output:
            _emit_json(
                answer="",
                citations=[],
                retrieval_trace=trace,
                session_id=session_id,
                errors=[message],
            )
        else:
            typer.echo("⚠ No relevant pages found in the wiki.")
            typer.echo("   Try ingesting some sources first with: olw ingest")
        return

    session_store = QuerySessionStore(config.llmwiki_dir / "query-sessions.json")
    previous = session_store.load(session_id) if session.strip() else None
    system = _build_system(selected_profile.instructions, instructions, sections)
    messages = _build_messages(ask, previous)

    if not json_output:
        typer.echo(f'🔍 Searching for: "{ask}"')
        typer.echo(f"🤖 Model: {config.llm.model}")
        typer.echo(f"📄 Retrieved {len(sections)} relevant page(s)")
        typer.echo("\n💭 Thinking...\n")

    try:
        raw_answer = call_llm(system, messages, config, max_tokens=2048, task="query")
    except Exception as exc:
        _fail(
            f"Query failed: {exc}",
            json_output=json_output,
            session_id=session_id,
            retrieval_trace=trace,
        )

    answer, citations, errors = _ground_answer(raw_answer, [section.path for section in sections])
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
    session_store.save(saved_session)

    if save_answer:
        try:
            _save_answer(Path(save_answer).expanduser(), answer)
        except OSError as exc:
            _fail(
                f"Could not save answer: {exc}",
                json_output=json_output,
                session_id=session_id,
                retrieval_trace=trace,
            )

    if json_output:
        _emit_json(
            answer=answer,
            citations=citations,
            retrieval_trace=trace,
            session_id=session_id,
            errors=errors,
        )
        return

    typer.echo(answer)
    typer.echo(
        f"\n---\n*Answer based on {len(sections)} wiki page(s). Model: {config.llm.model}*"
    )
    if save_answer:
        typer.echo(f"Saved answer: {Path(save_answer).expanduser()}")


def _build_system(profile_instructions: str, instructions: str, sections: tuple) -> str:
    context = "\n\n".join(
        f"--- PAGE: {section.path} ({section.title}) ---\n{section.snippet}" for section in sections
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
    atomic_write(path, content)


def _trace_payload(trace: RetrievalTrace) -> dict[str, Any]:
    return asdict(trace)


def _emit_json(
    *,
    answer: str,
    citations: list[str],
    retrieval_trace: dict[str, Any] | None,
    session_id: str,
    errors: list[str],
) -> None:
    typer.echo(
        json.dumps(
            {
                "answer": answer,
                "citations": citations,
                "retrieval_trace": retrieval_trace,
                "session": session_id,
                "errors": errors,
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
) -> NoReturn:
    if json_output:
        _emit_json(
            answer="",
            citations=[],
            retrieval_trace=retrieval_trace,
            session_id=session_id,
            errors=[message],
        )
    else:
        typer.echo(f"❌ {message}", err=True)
    raise typer.Exit(code=1)
