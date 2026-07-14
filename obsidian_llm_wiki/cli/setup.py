"""``olw setup`` — interactive setup wizard."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.cli import app


@app.command()
def setup():
    """Interactive setup wizard — configure LLM provider, vault path, API keys.

    Writes a .env file to your vault and scaffolds the bundle directory.

    Example:
        olw setup
    """
    print("🔧 obsidian-llm-wiki setup\n")

    # ── Vault path ─────────────────────────────────────────────────────
    vault_path = _ask("Vault path", str(Path.home() / "MyVault"))
    vault = Path(vault_path).expanduser().resolve()
    vault.mkdir(parents=True, exist_ok=True)
    print(f"   → Vault: {vault}")

    # ── LLM provider ───────────────────────────────────────────────────
    provider = _ask_choice("LLM provider", ["ollama", "openai"], "ollama")
    print(f"   → Provider: {provider}")

    if provider == "ollama":
        host = _ask("Ollama host", "http://localhost:11434")
        model = _ask("Model", "gemma3:27b")
        api_key = ""
    else:
        host = _ask("API host", "https://api.openai.com")
        model = _ask("Model", "gpt-4o")
        api_key = _ask("API key (leave empty to skip)", "")

    # ── Write .env ─────────────────────────────────────────────────────
    env_path = vault / ".env"
    lines = [
        "# obsidian-llm-wiki configuration",
        f"LLM_PROVIDER={provider}",
        f"LLM_HOST={host}",
        f"LLM_MODEL={model}",
    ]
    if api_key:
        lines.append(f"LLM_API_KEY={api_key}")
    lines.append(f"VAULT_PATH={vault}")
    lines.append("")
    lines.append("# Content thresholds")
    lines.append("MAX_SOURCE_CHARS=1000000")
    lines.append("MIN_SOURCE_CHARS=50")
    lines.append("")
    lines.append("# Concurrency")
    lines.append("COMPILE_CONCURRENCY=3")
    lines.append("")
    lines.append("# Quality gates")
    lines.append("CONCEPT_MIN_BODY_CHARS=800")
    lines.append("ENTRY_MIN_BODY_CHARS=500")
    lines.append("CLIPPING_MIN_BODY_CHARS=500")

    env_path.write_text("\n".join(lines) + "\n")
    print(f"\n✅ Configuration written: {env_path}")

    # ── Scaffold vault directories ────────────────────────────────────
    # Karpathy-style vault layout: raw sources → LLM-compiled wiki →
    # saved queries.
    vault_dirs = {
        "00-Inbox": "URL queue and incoming files",
        "01-Raw": "Immutable raw source documents",
        "02-Clippings": "Pre-processed markdown from web clippers (Defuddle, Obsidian Web Clipper)",
        "03-Raw-Annotations": "Manual notes, highlights, and commentary on sources",
        "04-Wiki": "LLM-compiled wiki (sources, entries, concepts, mocs)",
        "05-Queries": "Saved query results and analyses filed back into the vault",
    }
    for dirname, _desc in vault_dirs.items():
        (vault / dirname).mkdir(parents=True, exist_ok=True)
    print(f"✅ Vault directories scaffolded: {vault}")

    # ── Scaffold bundle directory ────────────────────────────────────
    wiki_dir = vault / "04-Wiki"
    for subdir in ["sources", "entries", "concepts", "mocs"]:
        (wiki_dir / subdir).mkdir(parents=True, exist_ok=True)
    (wiki_dir / ".llmwiki").mkdir(parents=True, exist_ok=True)
    print(f"✅ Wiki bundle scaffolded: {wiki_dir}")

    # ── Scaffold clippings processed subdirectory ─────────────────────
    (vault / "02-Clippings" / "processed").mkdir(parents=True, exist_ok=True)
    print(f"✅ Clippings archive ready: {vault / '02-Clippings' / 'processed'}")

    # ── Default schema policy ────────────────────────────────────────
    schema_path = wiki_dir / ".llmwiki" / "schema.yaml"
    if not schema_path.exists():
        schema_path.write_text(
            "# Vault-local schema policy for synthesis guidance.\n"
            "# This file is loaded by core/schema.py and injected into\n"
            "# the LLM synthesis prompt as a user-guidance block.\n\n"
            "granularity: detailed\n"
            "concept_name_convention: english-first\n"
            "max_concepts_per_source: 15\n"
            "require_rationale: true\n",
            encoding="utf-8",
        )
        print(f"✅ Default schema policy: {schema_path}")

    print("\nNext steps:")
    print(f"  olw ingest {vault} --url https://example.com/article")


def _ask(prompt: str, default: str) -> str:
    """Ask for a string input with a default."""
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else default


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Ask for a choice from a list."""
    choices_str = "/".join(choices)
    val = input(f"  {prompt} ({choices_str}) [{default}]: ").strip().lower()
    return val if val in choices else default
