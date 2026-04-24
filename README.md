# Obsidian LLM Wiki

An LLM-powered knowledge management pipeline that transforms raw web content into a structured, interconnected Obsidian wiki. Inspired by Andrej Karpathy's "LLM Knowledge Bases" approach.

**How it works:** Drop URLs into your vault, run one command. The 3-stage pipeline extracts content, plans note structure with semantic concept matching, and creates interlinked Source → Entry → Concept → MoC notes. Supports English and Chinese natively.

```
URLs in 01-Raw/  →  pipeline ingest  →  04-Wiki/{sources, entries, concepts, mocs}
                                              ↓
                                       Auto-updates:
                                       wiki-index.md, tag-registry.md,
                                       edges.tsv, log.md
```

## Quick Start

```bash
git clone https://github.com/0xminion/obsidian-llm-wiki.git
cd obsidian-llm-wiki
pip install -e .
pipeline init ~/MyVault
# Edit API keys:
nano ~/MyVault/Meta/Scripts/.env
```

## Usage

```bash
# Drop a URL (.url files in plain text or Windows InternetShortcut format both work)
echo 'https://example.com/article' > ~/MyVault/01-Raw/my-source.url

# Run the pipeline — that's it (3.8 min for 18 URLs with Ollama local inference)
pipeline ingest ~/MyVault
# Or, without installing the console script:
python3 -m pipeline.cli ingest ~/MyVault
```

### LLM Provider Configuration

Choose your provider via environment variables (in `~/MyVault/Meta/Scripts/.env`):

```bash
# Option 1: Ollama (default) — fast, private, local
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_INSIGHT_MODEL=minimax-m2.7:cloud   # insights + filenames
OLLAMA_FILENAME_MODEL=minimax-m2.7:cloud

# Option 2: OpenRouter — access 200+ models (Claude, GPT, Qwen, etc.)
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-sonnet-4
LLM_API_KEY=sk-openrouter-...

# Option 3: Hermes — full agent with tool access (slower, subprocess)
LLM_PROVIDER=hermes
AGENT_CMD=hermes
```

`pipeline ingest` currently scans `01-Raw/*.url` as its inbox. Other raw assets can live in the vault, but they are not auto-ingested by this command yet.

### Pipeline Commands

```bash
pipeline ingest ~/MyVault           # full pipeline: extract → plan → create
pipeline ingest --parallel 5        # more parallel workers
pipeline ingest --dry-run           # preview without executing
pipeline ingest --review            # stage plans for human review
pipeline ingest --resume            # continue from reviewed plans
pipeline compile ~/MyVault          # semantic cross-linking, MoC rebuild, edges
pipeline lint ~/MyVault             # 15 health checks + synonym detection
pipeline lint --fix                 # auto-fix safe issues
pipeline validate ~/MyVault         # post-write quality gate
pipeline validate --fix             # auto-repair missing sections
pipeline reindex ~/MyVault          # rebuild wiki-index.md
pipeline stats ~/MyVault            # vault dashboard
pipeline tags ~/MyVault             # rebuild tag registry
pipeline query --ask "question"     # Q&A against your vault (Hermes agent)
pipeline query --ask "question" --fast  # fast direct LLM query (sub-5s)
```

## How It Works

### Stage 1: Extract

Pure Python extraction — no LLM involved. Routes URLs to type-specific extractors with hardened URL validation on each network boundary and fallback chains:

| Source | Primary | Fallback |
|--------|---------|----------|
| Web | defuddle | curl extraction |
| X/Twitter | defuddle | curl extraction |
| YouTube | TranscriptAPI | Supadata → faster-whisper |
| Podcasts | AssemblyAI | whisper |

Features: retry with exponential backoff, SSRF-resistant URL validation, content quality validation, SQLite dedup store, dead letter queue for failures, and loud failure when every extraction fails.

### Stage 2: Plan

Single LLM agent batches planning for all extracted sources. Before the agent runs:

1. **Dedup check** — Jaccard similarity against existing vault sources
2. **Semantic concept search** — QMD MCP (HTTP daemon on localhost:8181) finds related concepts via hybrid semantic+keyword search
3. **Tag vocabulary injection** — existing tags passed to agent for reuse

The agent produces creation plans: title, language (EN/ZH), template, tags, concept targets, MoC assignments.

### Stage 3: Create

Default mode writes deterministic templates, then uses the configured LLM only for bounded insights. `--agent` enables the heavier legacy batch-agent creation path.

**Post-creation validation (per-batch):**
- Frontmatter completeness (title, source, date, status, template, tags)
- Required sections present per template type
- Stub/placeholder detection (20+ patterns)
- Minimum body length (200 chars for entries)
- Banned tag detection

**Auto-repair:** Missing sections get content derived from the file's existing body — never boilerplate.

### Compile Pass

Runs after ingest or manually via `pipeline compile`:

**Semantic operations (direct LLM, no subprocess):**
1. **Cross-link analysis** — embedding similarity + LLM validation adds missing `[[wikilinks]]`
2. **Concept merging** — detects near-duplicates, merges with LLM approval, updates all references
3. **MoC rebuild** — resynthesizes topic hubs from related notes via embeddings

**Deterministic operations:**
4. **Wiki index rebuild** — deterministic scan of all entries, concepts, MoCs
5. **Typed edges** — `edges.tsv` from wikilinks, sources, tags (9 relationship types)
6. **Duplicate detection** — title similarity report for human review
7. **Structured metrics report + log entry**

### Lint System

15 health checks; graph-sensitive checks rebuild from disk to avoid stale reports:

| Check | What it catches |
|-------|----------------|
| Orphaned notes | Notes with zero incoming links |
| Unreviewed entries | Entries never human-reviewed |
| Stale reviews | Reviews pending > 14 days |
| Broken wikilinks | Links to non-existent notes |
| Empty notes | Body too short |
| Concept structure | Missing required sections |
| Entry template | Wrong sections for template type |
| Orphaned concepts | Concepts not referenced by entries |
| Wiki index drift | Index counts vs actual files |
| Edges consistency | edges.tsv references to missing notes |
| Stubs | Placeholder content detected |
| Tag quality | Banned tags, too-short tags, synonyms |
| Frontmatter validity | YAML parse errors |
| Required sections | Missing mandatory sections |
| Markdown format | Structural issues |

## Vault Structure

```
01-Raw/              ← drop .url inbox items here
02-Clippings/        ← reserved for manual web clipper imports (not ingested by `pipeline ingest`)
03-Queries/          ← drop .md files with questions for Q&A
04-Wiki/
├── sources/         ← full original content
├── entries/         ← summaries + insights
├── concepts/        ← shared vocabulary (evergreen)
└── mocs/            ← topic hubs
05-Outputs/          ← Q&A responses
06-Config/           ← wiki-index, edges.tsv, tag-registry, log.md
07-WIP/              ← your drafts (untouched by automation)
08-Archive-Raw/      ← processed inbox items
09-Archive-Queries/  ← answered queries
Meta/
├── Scripts/         ← pipeline code, logs, cache
├── prompts/         ← agent prompt templates
└── Templates/       ← note templates
```

## Note Structures

**Entries** (English):
```
Summary → Core insights → Other takeaways → Diagrams → Open questions → Linked concepts
```

**Entries** (Chinese):
```
摘要 → 核心发现 → 其他要点 → 图表 → 开放问题 → 关联概念
```

**Other entry templates:** technical, comparison, procedural.

**Concepts** (evergreen, one idea per note):
```
Core concept → Context (flowing prose) → Links
```

**MoCs** — topic hubs with synthesized summaries.

## Typed Relationships

Relationships in `06-Config/edges.tsv` (4-column TSV):

```
source	target	type	description
```

Types: `extends`, `contradicts`, `supports`, `supersedes`, `tested_by`, `depends_on`, `inspired_by`, `part_of`, `relates_to`

Built automatically during `pipeline compile` from wikilinks, concept sources, shared concept tags, and MoC membership. Shared-tag relationships are emitted as symmetric `relates_to` edges, not directional claims.

## Semantic Search

Uses [QMD MCP](https://github.com/tobi/qmd) running as an HTTP daemon on `localhost:8181` for semantic concept matching. QMD handles embedding generation, indexing, and hybrid keyword+vector search internally. Falls back to local keyword search if QMD is unavailable.

Legacy Ollama embedding (`qwen3-embedding:0.6b`) is still supported for other operations (e.g., compile pass) but concept search now prefers QMD MCP.

`pipeline query` uses the wiki index plus retrieved snippets from relevant entries, sources, concepts, and MoCs before asking the agent.

```bash
# One-time setup
python3 -m pipeline.cli setup-qmd

# Manual queries
qmd query "prediction markets" --json -n 5 -c concepts
```

## Critical Rules

1. Never touch `07-WIP/`
2. Never overwrite existing notes — collision detection appends `-1`, `-2`, etc.
3. No stubs — every section must have real content at creation
4. Tags: topic-specific English only, never platform names (`x.com`, `tweet`, `source`)
5. Chinese body stays Chinese in all 04-Wiki notes
6. YAML wikilinks must be quoted: `source: "[[note]]"`
7. File names match content language (Chinese titles → Chinese filenames)
8. Never use URL slugs as filenames

## Configuration

API keys in `~/MyVault/Meta/Scripts/.env`:

```bash
# YouTube transcripts
TRANSCRIPT_API_KEY=***
SUPADATA_API_KEY=***

# Podcast transcription
ASSEMBLYAI_API_KEY=***

# Defaults
VAULT_PATH=$HOME/MyVault
AGENT_CMD=hermes
PARALLEL=3
```

## Testing

```bash
python3 -m pytest tests/ -v
```

666 tests covering: extraction, planning, creation, validation, lint, compile, vault operations, models, config, security regressions, and integration.

## Recommended Workflow

**Daily:** Drop sources in `01-Raw/`, run `pipeline ingest ~/MyVault`

**Weekly:** `pipeline compile` → review entries → `pipeline lint --fix`

**Monthly:** Check `pipeline stats` for growth trends

## Testing

```bash
python3 -m pytest tests/ -v
```

**666 tests** covering: extraction, planning, creation, validation, lint, compile (semantic + deterministic), LLM client (multi-provider), vault operations, models, config, security regressions, and integration.

## Architecture

Python-first with a unified LLM client supporting multiple providers:

```
pipeline/
├── cli.py              # typer CLI — all commands
├── llm_client.py       # Unified LLM client (Ollama/OpenRouter/Hermes)
├── extract.py          # Stage 1: URL routing, extraction, dedup
├── plan.py             # Stage 2: semantic search, agent planning
├── create/             # Stage 3 creation
│   ├── agent.py         # Agent execution (Hermes, legacy)
│   ├── orchestrator.py  # Batch coordination + post-processing
│   ├── prompts.py       # Prompt construction
│   ├── templates.py     # Template mode (deterministic + fast LLM insights)
│   └── validate.py      # Output validation + auto-repair
├── compile.py          # Compile pass: semantic ops + deterministic ops
├── lint.py             # 15 lint checks with cache support
├── vault.py            # File operations, collision detection
├── store.py            # SQLite content store + vault cache
├── config.py           # Configuration and environment
├── models.py           # Data models (Manifest, Plan, Edge, etc.)
├── qmd.py              # Semantic search orchestrator (QMD MCP → keyword fallback)
├── qmd_mcp.py          # QMD MCP HTTP client (JSON-RPC over HTTP)
├── utils.py            # Shared utilities
└── extractors/         # Type-specific extractors
    ├── web.py
    ├── youtube.py
    ├── podcast.py
    └── _shared.py
```

~2,900 lines of Python, 666 tests, unified LLM client with 3 providers, 0 shell scripts in the critical path.
