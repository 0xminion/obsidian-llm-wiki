# llmwiki

A knowledge compiler CLI. Raw sources in, interlinked wiki out.

Ported from [llm-wiki-compiler](https://github.com/atomicmemory/llm-wiki-compiler) v0.6.0 — Python-first, agent-native pipeline.

## Architecture

```
Sources / URLs / Clippings  →  Stage 1 (Deterministic)  →  Stage 2 (LLM: one per item)  →  Stage 3 (Resolve + Index)
                                 Extract full content        Entry → Concepts → MoCs          Wikilinks, Index, MOC
                                         ↓                           ↓                              ↓
                                    sources/*.md             entries/*.md + concepts/*.md     wiki/index.md, wiki/MOC.md
```

## Quick Start

```bash
git clone https://github.com/0xminion/obsidian-llm-wiki.git
cd obsidian-llm-wiki
pip install -e .

# Interactive setup
llmwiki setup

# Ingest a URL
llmwiki ingest ~/MyVault --url https://example.com/article

# Ingest multiple URLs + process clippings
llmwiki ingest ~/MyVault -u URL1 -u URL2

# Compile: resolve links, generate index
llmwiki compile ~/MyVault

# Query your knowledge
llmwiki query ~/MyVault --ask "What is gradient descent?"
```

## Commands

| Command | Description |
|---------|-------------|
| `llmwiki setup` | Interactive setup wizard — configures Ollama, vault, keys |
| `llmwiki ingest` | Ingest URLs + clippings → extract → create → compile |
| `llmwiki compile` | Detect changes, regenerate pages, resolve wikilinks |
| `llmwiki lint` | Scan vault for issues (stubs, broken links, malformed citations) |
| `llmwiki query` | RAG-style question answering against the vault |
| `llmwiki candidates` | Review pending candidate pages (approve/reject) |

## Stages

### Stage 1: Extraction (Deterministic, No LLM)
- Web URLs: defuddle → curl → archive.org fallback
- Full content, never truncated
- SHA-256 dedup against existing sources

### Stage 2: Creation (LLM — One Call Per Item)
- **One entry per source** — comprehensive analysis, all insights surfaced
- **One concept per concept** — evergreen atomic notes, 800+ chars minimum, never stubs
- **One MoC per topic** — meaningful cross-references, not shallow

### Stage 3: Compile (Deterministic + LLM)
- Bidirectional wikilink resolver (rule-based)
- Index + MOC generation (deterministic)
- Orphan management for deleted sources
- Incremental: only changed sources re-processed

## Clippings

Drop pre-extracted markdown into `02-Clippings/`. Quality gate:
- Body > 500 chars + has title → **skip extraction**, go straight to Stage 2
- Too short / no title → pass through Stage 1 extraction

## Vault Structure

```
{Vault}/
├── 02-Clippings/      ← Pre-extracted markdown (Obsidian Web Clipper)
├── 04-Wiki/
│   ├── sources/       ← Full original content
│   ├── entries/       ← Summaries + insights
│   ├── concepts/      ← Evergreen atomic notes
│   ├── mocs/          ← Maps of Content
│   ├── index.md       ← Auto-generated wiki index
│   ├── MOC.md         ← Auto-generated topic map
│   └── .llmwiki/      ← Pipeline state (hash db, lock, candidates)
└── .env               ← Configuration
```

## Configuration

All via `.env` in vault root:

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4:31b-cloud
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
VAULT_PATH=$HOME/MyVault
COMPILE_CONCURRENCY=3
CONCEPT_MIN_BODY_CHARS=800
ENTRY_MIN_BODY_CHARS=500
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with `gemma4:31b-cloud` (or any model)
- [defuddle](https://github.com/nousresearch/defuddle) (for web extraction)
