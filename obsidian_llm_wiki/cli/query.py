"""``olw query`` — RAG-style question answering against the vault."""

from __future__ import annotations

import sys

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, safe_read_file


@app.command()
def query(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    ask: str = typer.Option(..., "--ask", "-a", help="Question to ask"),
    max_results: int = typer.Option(
        10, "--max-results", "-n", help="Max pages to retrieve"
    ),
):
    """Query the knowledge wiki using LLM with retrieval-augmented generation.

    Searches concept and entry pages for relevant content, then uses the LLM
    to answer grounded in the retrieved context.

    Examples:
        olw query ~/MyVault --ask "What is a transformer model?"
        olw query ~/MyVault -a "Compare RAG vs fine-tuning" -n 5
    """
    vault_path, config = resolve_vault(vault)

    print(f'🔍 Searching for: "{ask}"')
    print(f"🤖 Model: {config.llm.model}")

    # ── Gather context from wiki pages ─────────────────────────────────
    query_lower = ask.lower()
    query_words = set(query_lower.split())

    scored: list[tuple[int, str, str]] = []

    for dir_path in [config.concepts_dir, config.entries_dir, config.sources_dir]:
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.md"):
            raw = safe_read_file(f)
            if not raw.strip():
                continue
            meta, body = parse_frontmatter(raw)
            title = meta.get("title", f.stem)

            body_lower = body.lower()
            score = sum(body_lower.count(w) for w in query_words if len(w) > 2)
            title_lower = title.lower()
            score += sum(10 for w in query_words if len(w) > 2 and w in title_lower)

            if score > 0:
                snippet = body[:800] if len(body) > 800 else body
                relpath = str(f.relative_to(config.wiki_dir))
                scored.append((score, relpath, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_pages = scored[:max_results]

    if not top_pages:
        print("⚠ No relevant pages found in the wiki.")
        print("   Try ingesting some sources first with: olw ingest")
        return

    context_parts = [
        f"--- PAGE: {relpath} (relevance: {score}) ---\n{snippet}"
        for score, relpath, snippet in top_pages
    ]
    context = "\n\n".join(context_parts)

    print(f"📄 Retrieved {len(top_pages)} relevant page(s)")
    print("\n💭 Thinking...\n")

    from obsidian_llm_wiki.providers.llm import call_llm

    system = (
        "You are a knowledgeable assistant answering questions based on a "
        "personal knowledge wiki. Ground your answer in the provided context. "
        "If the context doesn't contain enough information, say so honestly. "
        f"Cite the specific pages you used.\n\n"
        f"--- KNOWLEDGE WIKI CONTEXT ---\n{context}\n--- END CONTEXT ---"
    )
    messages = [{"role": "user", "content": ask}]

    try:
        answer = call_llm(system, messages, config, max_tokens=2048)
        print(answer)
        print(f"\n---\n*Answer based on {len(top_pages)} wiki page(s). "
              f"Model: {config.llm.model}*")
    except Exception as exc:
        print(f"❌ Query failed: {exc}", file=sys.stderr)
        raise typer.Exit(code=1) from exc
