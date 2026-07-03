# obsidian-llm-wiki

LLM-powered knowledge compiler for Obsidian vaults — sources in, interlinked wiki out.

`obsidian-llm-wiki` takes raw web sources (URLs, clippings) and compiles them
into an Obsidian vault with LLM-synthesised concepts, tags, summaries, and
cross-links. The core innovation is a **single LLM synthesis call** that
produces a structured JSON bundle (summaries, tags, concepts, relationships,
citations, MOCs) — all markdown rendering is deterministic.

---

## Architecture

```
Sources / URLs / Clippings  →  Ingest  →  Synthesise  →  Merge  →  Render
                                (httpx)    (1 LLM call    (corpus    (deterministic
                                           per source)     dedup)     markdown)
         ↓                       ↓           ↓              ↓           ↓
    SourceDoc               web.py     SynthesisBundle   dedupe    Obsidian vault
                            clippings  (typed JSON)     .py       (wikilinks,
                                                                   frontmatter)
```

### The Synthesis Contract

The LLM produces **one structured JSON object per source** containing
everything the renderers need:

```json
{
  "source_title": "...",
  "source_summary": "2-3 sentence overview",
  "source_tags": ["machine-learning", "optimization"],
  "key_points": ["...", "..."],
  "concepts": [
    {
      "title": "Gradient Descent",
      "slug": "gradient-descent",
      "summary": "Optimization algorithm for minimizing loss",
      "tags": ["optimization", "machine-learning"],
      "sections": [{"heading": "Core", "points": ["..."]}],
      "related": [{"slug": "sgd", "relation": "variant_of"}],
      "claims": [{"text": "Learning rate controls step size"}]
    }
  ],
  "maps": [
    {"title": "Optimization", "slug": "optimization", "concept_slugs": ["gradient-descent"]}
  ]
}
```

All markdown generation is **pure functions** — no LLM calls during render.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/0xminion/obsidian-llm-wiki.git
cd obsidian-llm-wiki
pip install -e .

# Interactive setup
olw setup

# Ingest URLs and build the vault
olw ingest ~/MyVault --url https://example.com/article

# Re-build after manual edits to sources
olw build ~/MyVault

# Query the vault
olw query ~/MyVault --ask "What is a transformer model?"

# Validate the vault
olw validate ~/MyVault
```

---

## Commands

| Command | Description |
|---------|-------------|
| `olw setup` | Interactive setup wizard — configures LLM provider, vault path |
| `olw ingest` | Ingest URLs + clippings → synthesise → render in one pass |
| `olw build` | Re-synthesise changed sources and re-render the vault |
| `olw query` | RAG-style question answering against the vault |
| `olw validate` | Check vault for conformance issues |

---

## Vault Structure

```
{Vault}/
├── 02-Clippings/          ← Pre-extracted markdown (Obsidian Web Clipper)
├── 04-Wiki/               ← Wiki root
│   ├── sources/           ← Original source content
│   ├── entries/           ← LLM-synthesised analysis pages
│   ├── concepts/          ← Evergreen atomic notes (with wikilinks)
│   ├── mocs/              ← Maps of Content (topic groupings)
│   ├── index.md           ← Bundle-root index
│   └── .llmwiki/          ← Internal state (excluded from export)
│       ├── state.json     ← Source hash database for incremental builds
│       └── lock           ← Compile-time PID lock
└── .env                   ← Configuration
```

Every `.md` file carries YAML frontmatter with a `type` field:

```yaml
---
type: Concept
title: Gradient Descent
tags: [machine-learning, optimization]
aliases: [GD, batch gradient descent]
timestamp: 2026-07-03T10:00:00Z
---
```

Cross-links use Obsidian wikilinks: `[[gradient-descent]]` or `[[gradient-descent|GD]]`.

---

## Configuration

All configuration is via a `.env` file in the vault root (created by `olw setup`):

```bash
# LLM Provider
LLM_PROVIDER=ollama          # ollama | openai
LLM_HOST=http://localhost:11434
LLM_MODEL=gemma3:27b
LLM_API_KEY=                  # for openai providers

# Vault
VAULT_PATH=$HOME/MyVault

# Content thresholds
MAX_SOURCE_CHARS=1000000
MIN_SOURCE_CHARS=50

# Concurrency
COMPILE_CONCURRENCY=3

# Language (en, zh, or empty for auto-detect)
OUTPUT_LANGUAGE=

# Quality gates
CONCEPT_MIN_BODY_CHARS=800
ENTRY_MIN_BODY_CHARS=500
CLIPPING_MIN_BODY_CHARS=500
```

### LLM Providers

| Provider | `LLM_PROVIDER` | Auth | Endpoints |
|----------|---------------|------|-----------|
| **Ollama** (local) | `ollama` | None | `/api/chat` |
| **OpenAI-compatible** | `openai` | `Bearer` API key | `/v1/chat/completions` |

For OpenAI, OpenRouter, LM Studio, vLLM, etc., set `LLM_PROVIDER=openai`,
`LLM_API_KEY=<key>`, and `LLM_HOST=<endpoint>`.

---

## Package Structure

```
obsidian_llm_wiki/
  cli/                  # Typer CLI (ingest, build, query, setup, validate)
  core/
    models.py           # SynthesisBundle schema (the synthesis contract)
    pipeline.py         # Single orchestrator (~200 LOC)
    state.py            # Incremental compilation state
    lock.py             # PID-based compile lock
  ingest/
    web.py              # httpx + trafilatura web extraction
    clippings.py        # Clippings quality gate
  synth/
    prompts.py          # Single-call synthesis prompt builder
    parser.py           # JSON → SynthesisBundle validation
    dedupe.py           # Corpus-level concept/tag reconciliation
  render/
    obsidian.py         # Deterministic Obsidian markdown renderer
  providers/
    llm.py              # Ollama + OpenAI-compatible clients
  adapters/             # Optional tools (migrate, visualizer, export, enrich)
  config.py             # Configuration management
pipeline/               # Legacy package (backward compat, 320 tests)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest                     # 374 tests (320 legacy + 54 new)
ruff check .               # lint
```

### Tests

The test suite includes a **golden end-to-end test** (`test_golden_pipeline.py`)
that feeds a fake source through the complete pipeline with a mocked LLM and
asserts the exact vault structure, frontmatter, wikilinks, and state
persistence. This is the test that proves the product works as intended.

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally (default), or any
  **OpenAI-compatible API** endpoint
- Python dependencies: `typer`, `httpx`, `pyyaml`, `python-dotenv`, `trafilatura`