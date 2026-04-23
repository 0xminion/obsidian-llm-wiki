"""Setup utilities for qmd and git hooks.

Ports setup-qmd.sh and setup-git-hooks.sh to Python.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pipeline._common import append_log_md


def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with sensible defaults."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=kwargs.get("timeout", 300),
        cwd=kwargs.get("cwd"),
    )


def _has_cmd(name: str) -> bool:
    """Check if a command is available on PATH."""
    return subprocess.run(
        ["which", name],
        capture_output=True,
        text=True,
    ).returncode == 0


def setup_qmd(vault_path: Path) -> None:
    """Install and configure qmd for semantic concept search.

    Steps:
    1. Install qmd via npm (if not present)
    2. Configure Qwen3-Embedding-0.6B-Q8 embedding model
    3. Index the concepts collection
    4. Generate embeddings
    5. Print verification

    Args:
        vault_path: Path to the Obsidian vault.
    """
    vault_path = Path(vault_path)

    print("╔══════════════════════════════════════════════╗")
    print("║  QMD Setup — Semantic Concept Search         ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # ── Step 1: Install qmd ──────────────────────────────────────────────
    if _has_cmd("qmd"):
        version_result = _run(["qmd", "--version"])
        version = version_result.stdout.strip() or "unknown"
        print(f"✓ qmd already installed: {version}")
    else:
        print("Installing qmd via npm...")
        result = _run(["npm", "install", "-g", "@tobilu/qmd"])
        if _has_cmd("qmd"):
            version_result = _run(["qmd", "--version"])
            version = version_result.stdout.strip()
            print(f"✓ qmd installed: {version}")
        else:
            print("ERROR: Failed to install qmd. Ensure Node.js >= 22 is installed.")
            print(result.stderr[-300:] if result.stderr else "(no stderr)")
            sys.exit(1)

    # ── Step 2: Configure embedding model ────────────────────────────────
    print()
    print("Configuring Qwen3-Embedding-0.6B-Q8 model...")
    config_dir = Path.home() / ".config" / "qmd"
    config_dir.mkdir(parents=True, exist_ok=True)
    index_yml = config_dir / "index.yml"

    model_config = (
        'models:\n'
        '  embed: "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"\n'
    )

    if index_yml.exists() and "Qwen3-Embedding-0.6B" in index_yml.read_text():
        print("✓ Model already configured")
    else:
        index_yml.write_text(model_config)
        print(f"✓ Model config written to {index_yml}")

    # ── Step 3: Index concepts collection ────────────────────────────────
    concepts_dir = vault_path / "04-Wiki" / "concepts"

    if not concepts_dir.is_dir():
        print(f"ERROR: Concepts directory not found: {concepts_dir}")
        sys.exit(1)

    concept_count = len(list(concepts_dir.glob("**/*.md")))
    print()
    print(f"Indexing {concept_count} concept files from: {concepts_dir}")

    status_result = _run(["qmd", "status"])
    if "concepts" in status_result.stdout:
        print("Collection 'concepts' already exists, updating...")
        update_result = _run(["qmd", "update"])
        for line in update_result.stdout.splitlines():
            if any(line.startswith(p) for p in ("Indexed", "Collection", "✓")):
                print(line)
    else:
        add_result = _run([
            "qmd", "collection", "add", str(concepts_dir),
            "--name", "concepts", "--mask", "**/*.md",
        ])
        for line in add_result.stdout.splitlines():
            if any(line.startswith(p) for p in ("Indexed", "Collection", "Creating", "✓")):
                print(line)

    # ── Step 4: Generate embeddings ──────────────────────────────────────
    print()
    print("Generating embeddings (this may take a few minutes on CPU)...")

    noisy_patterns = (
        "cmake", "CMAKE", "Vulkan", "vulkan", "node-llama-cpp",
        "Cloning", "NOT searching", "C compiler", "Check for working",
        "Detecting", "Found", "Including", "Adding", "Performing", "Configuring",
    )

    embed_result = _run(["qmd", "embed", "-f"])
    for line in embed_result.stdout.splitlines():
        if not any(p in line for p in noisy_patterns):
            print(line)
    # Show last 5 lines of filtered output
    filtered = [
        l for l in embed_result.stdout.splitlines()
        if not any(p in l for p in noisy_patterns)
    ]
    for line in filtered[-5:]:
        print(line)

    # ── Step 5: Verify ──────────────────────────────────────────────────
    print()
    print("━━━ Verification ━━━")
    status_result = _run(["qmd", "status"])
    for line in status_result.stdout.splitlines():
        if any(kw in line for kw in ("Documents", "Vectors", "Collection", "Embedding")):
            print(line)

    print()
    print("Test query (semantic search):")
    query_result = _run([
        "qmd", "query", "prediction markets",
        "--json", "-n", "3", "--min-score", "0.3",
        "-c", "concepts", "--no-rerank",
    ])

    try:
        results = json.loads(query_result.stdout)
        for r in results:
            f = r.get("file", "").split("/")[-1].replace(".md", "")
            s = r.get("score", 0)
            print(f"  {s:.2f}  {f}")
        print(f"\n✓ {len(results)} semantic matches found")
    except (json.JSONDecodeError, TypeError):
        print("  (query test skipped — first run may need model download)")

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  QMD Setup Complete                          ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Model:   Qwen3-Embedding-0.6B-Q8 (639MB)   ║")
    print(f"║  Concepts: {concept_count} files indexed        ║")
    print("║  Search:  qmd query '<text>' -c concepts     ║")
    print("╚══════════════════════════════════════════════╝")

    # Log the operation
    append_log_md(
        vault_path,
        "setup",
        "QMD semantic search setup",
        f"- Model: Qwen3-Embedding-0.6B-Q8\n- Concepts indexed: {concept_count}",
    )


# ── Git hook scripts ──────────────────────────────────────────────────────────

_PRE_COMMIT_HOOK = """\
#!/usr/bin/env bash
# Prevent committing files from 07-WIP/
if git diff --cached --name-only | grep -q "^07-WIP/"; then
  echo "ERROR: Cannot commit files from 07-WIP/ — this is user territory."
  echo "Use 'git reset HEAD 07-WIP/' to unstage."
  exit 1
fi
"""

_COMMIT_MSG_HOOK = """\
#!/usr/bin/env bash
# Allow structured messages: "operation: description (date)"
# Also allow merge commits and reverts
msg=$(cat "$1")
if [[ ! "$msg" =~ ^(ingest|compile|query|lint|review|reindex|setup|Merge|Revert): ]] \\
    && [[ ! "$msg" =~ ^Initial ]]; then
  echo "Warning: commit message should follow 'operation: description (date)' format"
  echo "Got: $msg"
  # Don't block — just warn
fi
"""


def setup_git_hooks(vault_path: Path) -> None:
    """Initialize git repo (if needed) and install pre-commit/commit-msg hooks.

    Hooks installed:
    - pre-commit: blocks commits from 07-WIP/
    - commit-msg: warns on non-structured commit messages

    Args:
        vault_path: Path to the Obsidian vault.
    """
    vault_path = Path(vault_path)

    # ── Init git repo if needed ─────────────────────────────────────────
    git_dir = vault_path / ".git"
    if not git_dir.is_dir():
        print(f"Vault at {vault_path} is not a git repository.")
        print("Initializing...")
        _run(["git", "init"], timeout=30, cwd=vault_path)
        gitignore = vault_path / ".gitignore"
        gitignore.write_text("# Wiki Vault\n.DS_Store\n*.tmp\n")
        _run(["git", "add", "-A"], cwd=vault_path)
        _run(["git", "commit", "-m", "Initial vault commit", "--quiet"], cwd=vault_path)

    # ── Write hooks ─────────────────────────────────────────────────────
    hooks_dir = vault_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    pre_commit_path = hooks_dir / "pre-commit"
    pre_commit_path.write_text(_PRE_COMMIT_HOOK)
    pre_commit_path.chmod(0o755)

    commit_msg_path = hooks_dir / "commit-msg"
    commit_msg_path.write_text(_COMMIT_MSG_HOOK)
    commit_msg_path.chmod(0o755)

    print(f"Git hooks installed in {hooks_dir}")
    print("  pre-commit: blocks 07-WIP/ commits")
    print("  commit-msg: warns on non-structured messages")

    # ── Commit current state if needed ──────────────────────────────────
    diff_result = _run(["git", "diff", "--quiet"], cwd=vault_path)
    cached_diff_result = _run(["git", "diff", "--cached", "--quiet"], cwd=vault_path)
    if diff_result.returncode != 0 or cached_diff_result.returncode != 0:
        _run(["git", "add", "-A"], cwd=vault_path)
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        _run([
            "git", "commit", "-m",
            f"setup: Install git hooks ({date_str})",
            "--quiet",
        ], cwd=vault_path)
        print("Committed current state.")

    print("Done. Vault is now git-tracked with auto-commit support.")

    # Log the operation
    append_log_md(
        vault_path,
        "setup",
        "Git hooks installed",
        "- pre-commit: blocks 07-WIP/ commits\n- commit-msg: warns on non-structured messages",
    )
