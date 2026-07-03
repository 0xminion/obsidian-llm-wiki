"""Interactive setup wizard for llmwiki pipeline configuration.

Guides the user through configuring their LLM provider (ollama / openai /
custom), vault path, OKF bundle path, and API keys, validates connectivity,
writes a .env file, and scaffolds the initial OKF bundle structure.
"""

from __future__ import annotations

import sys
from pathlib import Path


def run_setup() -> None:
    """Interactive setup: ask user for config, validate, write .env.

    Steps:
      1. Ask for LLM provider (ollama / openai / custom)
      2. If openai/custom: ask for API key and host URL
      3. If ollama: ask for Ollama host, LLM model, embedding model
      4. Ask for vault path (default: ~/MyVault)
      5. Ask for OKF bundle path (default: vault/04-Wiki)
      6. Ask for output language (default: auto)
      7. Validate: check vault path, check provider reachable
      8. Write .env file with LLM_PROVIDER, LLM_API_KEY, OKF_VERSION
      9. Create initial bundle structure (sources/, entries/, concepts/,
         mocs/, references/, index.md)
     10. Print success + next steps
    """
    print()
    print("=" * 60)
    print("  🧠 llmwiki Setup Wizard")
    print("  Configure your knowledge compiler pipeline")
    print("=" * 60)
    print()

    # ── Helper: prompt with default ────────────────────────────────────
    def _ask(prompt: str, default: str) -> str:
        """Ask the user for input with a default value."""
        if default:
            result = input(f"{prompt} [{default}]: ").strip()
            return result if result else default
        else:
            return input(f"{prompt}: ").strip()

    def _ask_required(prompt: str) -> str:
        """Ask a required question; repeat until non-empty."""
        while True:
            result = input(f"{prompt}: ").strip()
            if result:
                return result
            print("  ⚠ This field is required. Please enter a value.")

    def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
        """Ask for a choice among a list of options."""
        choices_display = "/".join(choices)
        while True:
            result = input(f"{prompt} [{choices_display}] [{default}]: ").strip().lower()
            if not result:
                return default
            if result in choices:
                return result
            print(f"  ⚠ Please choose one of: {choices_display}")

    # ── Step 1: LLM Provider selection ─────────────────────────────────
    print("📡 LLM Provider Configuration")
    print("-" * 40)
    print("   ollama  — local Ollama instance (default)")
    print("   openai  — OpenAI API or compatible endpoint")
    print("   custom  — custom LLM endpoint with OpenAI-compatible API")
    print()
    llm_provider = _ask_choice(
        "Select LLM provider",
        ["ollama", "openai", "custom"],
        "ollama",
    )
    print(f"   → Provider: {llm_provider}")
    print()

    # Provider-specific configuration
    ollama_host = ""
    ollama_model = ""
    ollama_embed_model = ""
    llm_api_key = ""
    llm_host_url = ""

    if llm_provider == "ollama":
        # ── Ollama host ────────────────────────────────────────────────
        ollama_host = _ask(
            "Ollama host URL",
            "http://localhost:11434",
        )
        print(f"   → Host: {ollama_host}")

        # ── LLM model ──────────────────────────────────────────────────
        ollama_model = _ask(
            "LLM model name",
            "gemma4:31b-cloud",
        )
        print(f"   → Model: {ollama_model}")

        # ── Embedding model ────────────────────────────────────────────
        ollama_embed_model = _ask(
            "Embedding model name",
            "qwen3-embedding:0.6b",
        )
        print(f"   → Embedding model: {ollama_embed_model}")
    else:
        # ── OpenAI / custom: ask for API key and host URL ───────────────
        llm_api_key = _ask_required("API key")
        print(f"   → API key: {'*' * len(llm_api_key)}")

        default_host = "https://api.openai.com/v1" if llm_provider == "openai" else ""
        if default_host:
            llm_host_url = _ask("LLM host URL", default_host)
        else:
            llm_host_url = _ask_required("LLM host URL")
        print(f"   → Host URL: {llm_host_url}")
    print()

    # ── Step 4: Vault path ────────────────────────────────────────────
    print("📂 Vault Configuration")
    print("-" * 40)
    default_vault = str(Path.home() / "MyVault")
    vault_raw = _ask(
        "Vault path (where your Obsidian/wiki files live)",
        default_vault,
    )
    vault_path = Path(vault_raw).expanduser().resolve()
    print(f"   → Vault: {vault_path}")
    print()

    # ── Step 5: OKF bundle path ────────────────────────────────────────
    print("📦 OKF Bundle Configuration")
    print("-" * 40)
    default_bundle = str(vault_path / "04-Wiki")
    bundle_raw = _ask(
        "OKF bundle path (where the compiled knowledge bundle lives)",
        default_bundle,
    )
    bundle_path = Path(bundle_raw).expanduser().resolve()
    print(f"   → Bundle: {bundle_path}")
    print()

    # ── Step 6: Output language ───────────────────────────────────────
    print("🌐 Language Settings")
    print("-" * 40)
    output_language = _ask(
        "Output language (en / zh / auto — leave empty for auto-detect)",
        "auto",
    )
    if output_language.lower() == "auto":
        output_language = ""
    print(f"   → Language: {output_language if output_language else 'auto-detect'}")
    print()

    # ── Step 7: Validation ────────────────────────────────────────────
    print("🔍 Validating configuration...")
    print("-" * 40)

    all_ok = True

    # 7a: Check vault path
    if not vault_path.exists():
        create_choice = _ask(
            f"   Vault path '{vault_path}' does not exist. Create it?",
            "Y",
        )
        if create_choice.lower() in ("y", "yes", ""):
            try:
                vault_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created: {vault_path}")
            except OSError as exc:
                print(f"   ❌ Could not create vault path: {exc}")
                all_ok = False
        else:
            print("   ⚠ Vault path will not be created. You can create it later.")
    else:
        print(f"   ✅ Vault path exists: {vault_path}")

    # Create expected vault directory structure
    for subdir in ["02-Clippings", "04-Wiki"]:
        sub = vault_path / subdir
        if not sub.exists():
            sub.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {sub}")
        else:
            print(f"   ✅ Directory exists: {sub}")

    # 7b: Provider-specific connectivity check
    if llm_provider == "ollama":
        print()
        print("   Checking Ollama connectivity...")
        ollama_reachable = _check_ollama(ollama_host)
        if ollama_reachable:
            print(f"   ✅ Ollama is reachable at {ollama_host}")
        else:
            print(f"   ⚠ Could not reach Ollama at {ollama_host}")
            print("     Make sure Ollama is running. You can continue anyway.")
            # Not blocking — user might start Ollama later

        # 7c: Check model availability
        if ollama_reachable:
            model_available = _check_model(ollama_host, ollama_model)
            if model_available:
                print(f"   ✅ Model '{ollama_model}' is available")
            else:
                print(f"   ⚠ Model '{ollama_model}' not found on Ollama host")
                print(f"     Run: ollama pull {ollama_model}")
                # Not blocking

            embed_available = _check_model(ollama_host, ollama_embed_model)
            if embed_available:
                print(f"   ✅ Embedding model '{ollama_embed_model}' is available")
            else:
                print(f"   ⚠ Embedding model '{ollama_embed_model}' not found")
                print(f"     Run: ollama pull {ollama_embed_model}")
    else:
        # For openai/custom, we could do a lightweight API ping, but we
        # keep it simple: no blocking validation for the API key.
        print(f"   ℹ Provider '{llm_provider}' configured. Connectivity will be")
        print(f"     validated on first run. Host URL: {llm_host_url}")

    print()

    if not all_ok:
        print("❌ Setup could not be completed due to validation failures.")
        print("   Please fix the issues above and run 'llmwiki setup' again.")
        sys.exit(1)

    # ── Step 8: Write .env file ────────────────────────────────────────
    env_path = vault_path / ".env"
    env_content = _build_env_content(
        llm_provider=llm_provider,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        ollama_embed_model=ollama_embed_model,
        llm_api_key=llm_api_key,
        llm_host_url=llm_host_url,
        vault_path=str(vault_path),
        bundle_path=str(bundle_path),
        output_language=output_language,
    )

    if env_path.exists():
        overwrite = _ask(
            f"   .env file already exists at {env_path}. Overwrite?",
            "N",
        )
        if overwrite.lower() not in ("y", "yes"):
            backup_path = env_path.with_suffix(".env.backup")
            env_path.rename(backup_path)
            print(f"   📋 Backed up existing .env to {backup_path}")
            env_path.write_text(env_content)
            print(f"   ✅ Written: {env_path}")
        else:
            env_path.write_text(env_content)
            print(f"   ✅ Overwritten: {env_path}")
    else:
        env_path.write_text(env_content)
        print(f"   ✅ Written: {env_path}")

    # ── Step 9: Create initial OKF bundle structure ────────────────────
    print()
    print("📦 Creating initial OKF bundle structure...")
    print("-" * 40)
    _create_bundle_structure(bundle_path)
    print(f"   ✅ Bundle structure created at: {bundle_path}")

    # ── Step 10: Print success + next steps ────────────────────────────
    print()
    print("=" * 60)
    print("  ✅ Setup complete!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print()
    print("  1. Ingest some sources:")
    print(f"     llmwiki ingest {vault_path} --url https://example.com/article")
    print()
    print("  2. Or use your own clippings:")
    print(f"     Drop .md files into {vault_path / '02-Clippings'}")
    print()
    print("  3. Compile:")
    print(f"     llmwiki compile {vault_path}")
    print()
    print("  4. Query your knowledge:")
    print(f"     llmwiki query {vault_path} --ask \"your question\"")
    print()
    print("  📖 Docs: https://hermes-agent.nousresearch.com/docs")
    print()


# ── Helpers ────────────────────────────────────────────────────────────────

# OKF bundle subdirectories that are always created at setup time.
_BUNDLE_SUBDIRS = ["sources", "entries", "concepts", "mocs", "references"]

# Frontmatter for the bundle-root index.md.
_INDEX_FRONTMATTER = "---\nokf_version: '0.1'\n---\n"


def _create_bundle_structure(bundle_path: Path) -> None:
    """Create the initial OKF bundle directory tree under ``bundle_path``.

    Creates ``sources/``, ``entries/``, ``concepts/``, ``mocs/``, and
    ``references/`` directories, then writes a root ``index.md`` with an
    ``okf_version: '0.1'`` frontmatter block. Existing files/directories
    are left untouched (idempotent).
    """
    bundle_path.mkdir(parents=True, exist_ok=True)

    for subdir in _BUNDLE_SUBDIRS:
        (bundle_path / subdir).mkdir(parents=True, exist_ok=True)

    index_path = bundle_path / "index.md"
    if not index_path.exists():
        index_path.write_text(
            _INDEX_FRONTMATTER + "\n# Knowledge Bundle\n\n"
            "This is the root index of your OKF v0.1 knowledge bundle.\n",
            encoding="utf-8",
        )


def _build_env_content(
    llm_provider: str,
    ollama_host: str,
    ollama_model: str,
    ollama_embed_model: str,
    llm_api_key: str,
    llm_host_url: str,
    vault_path: str,
    bundle_path: str,
    output_language: str,
) -> str:
    """Build the .env file content from configuration values."""
    lines = [
        "# llmwiki configuration",
        "# Generated by llmwiki setup wizard",
        "",
        "# ── LLM Provider ──────────────────────────────",
        f"LLM_PROVIDER={llm_provider}",
        "",
    ]

    # Provider-specific fields
    if llm_provider == "ollama":
        lines.extend([
            "# ── Ollama ──────────────────────────────────────",
            f"OLLAMA_HOST={ollama_host}",
            f"OLLAMA_MODEL={ollama_model}",
            f"OLLAMA_EMBED_MODEL={ollama_embed_model}",
            "",
        ])
    else:
        # openai / custom
        lines.extend([
            "# ── OpenAI / Custom Provider ─────────────────────",
            f"LLM_API_KEY={llm_api_key}",
            f"LLM_HOST={llm_host_url}",
            "",
        ])

    lines.extend([
        "# ── OKF Bundle ──────────────────────────────────",
        f"OKF_BUNDLE_PATH={bundle_path}",
        "OKF_VERSION=0.1",
        "",
        "# ── Vault ───────────────────────────────────────",
        f"VAULT_PATH={vault_path}",
        "",
        "# ── Content thresholds ──────────────────────────",
        "MAX_SOURCE_CHARS=1000000",
        "MIN_SOURCE_CHARS=50",
        "",
        "# ── Concurrency ─────────────────────────────────",
        "COMPILE_CONCURRENCY=3",
        "",
    ])

    if output_language:
        lines.append(f"LLMWIKI_OUTPUT_LANGUAGE={output_language}")
    else:
        lines.append("# LLMWIKI_OUTPUT_LANGUAGE=  (auto-detect)")

    lines.append("")
    return "\n".join(lines) + "\n"


def _check_ollama(host: str) -> bool:
    """Check if Ollama is reachable at the given host."""
    try:
        import urllib.request
        url = f"{host.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        # Try httpx if available
        try:
            import asyncio

            import httpx

            async def _check():
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{host.rstrip('/')}/api/tags")
                    return resp.status_code == 200

            return asyncio.run(_check())
        except Exception:
            return False


def _check_model(host: str, model: str) -> bool:
    """Check if a specific model is available on the Ollama host."""
    try:
        import json
        import urllib.request

        url = f"{host.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check exact match or prefix match (e.g. "gemma4:31b-cloud" vs "gemma4:latest")
            return any(m == model or m.startswith(model.split(":")[0]) for m in models)
    except Exception:
        # Try httpx
        try:
            import asyncio

            import httpx

            async def _check():
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{host.rstrip('/')}/api/tags")
                    data = resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return any(m == model or m.startswith(model.split(":")[0]) for m in models)

            return asyncio.run(_check())
        except Exception:
            return False


# ── CLI entry point (for standalone testing) ──────────────────────────────


if __name__ == "__main__":
    run_setup()
