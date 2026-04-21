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
./setup.sh ~/MyVault
# Edit API keys:
nano ~/MyVault/Meta/Scripts/.env
```

## Usage

```bash
# Drop a URL
echo 'https://example.com/article' > ~/MyVault/01-Raw/my-source.url

# Run the pipeline — that's it
cd ~/MyVault && ./run.sh
```

### Pipeline Commands

```bash
pipeline ingest ~/MyVault           # full pipeline: extract → plan → create
pipeline ingest --parallel 5        # more parallel agents
pipeline ingest --dry-run           # preview without executing
pipeline ingest --review            # stage plans for human review
pipeline ingest --resume            # continue from reviewed plans
pipeline compile ~/MyVault          # concept convergence, MoC rebuild, edges
pipeline lint ~/MyVault             # 15 health checks + synonym detection
pipeline lint --fix                 # auto-fix safe issues
pipeline validate ~/MyVault         # post-write quality gate
pipeline validate --fix             # auto-repair missing sections
pipeline reindex ~/MyVault          # rebuild wiki-index.md
pipeline stats ~/MyVault            # vault dashboard
pipeline tags ~/MyVault             # rebuild tag registry
pipeline query --ask "question"     # Q&A against your vault
```

## How It Works

### Stage 1: Extract

Pure Python extraction — no LLM involved. Routes URLs to type-specific extractors with fallback chains:

| Source | Primary | Fallback |
|--------|---------|----------|
| Web | defuddle | curl extraction |
| X/Twitter | defuddle | curl extraction |
| YouTube | TranscriptAPI | Supadata → faster-whisper |
| Podcasts | AssemblyAI | whisper |
| PDFs | liteparse | OCR |

Features: retry with exponential backoff, content quality validation, SQLite dedup store, dead letter queue for failures.

### Stage 2: Plan

Single LLM agent batches planning for all extracted sources. Before the agent runs:

1. **Dedup check** — Jaccard similarity against existing vault sources
2. **Semantic concept search** — qmd embeddings (Qwen3-Embedding-0.6B-Q8) find related concepts
3. **Tag vocabulary injection** — existing tags passed to agent for reuse

The agent produces creation plans: title, language (EN/ZH), template, tags, concept targets, MoC assignments.

### Stage 3: Create

N parallel agents write vault files. Each agent receives a batch with extracted content and concept convergence data.

**Post-creation validation (per-batch):**
- Frontmatter completeness (title, source, date, status, template, tags)
- Required sections present per template type
- Stub/placeholder detection (20+ patterns)
- Minimum body length (200 chars for entries)
- Banned tag detection

**Auto-repair:** Missing sections get content derived from the file's existing body — never boilerplate.

### Compile Pass

Runs after ingest or manually via `pipeline compile`:

1. Agent: cross-link analysis + concept merging (semantic judgment)
2. Python: wiki index rebuild (deterministic)
3. Python: typed edges construction from wikilinks and metadata (deterministic)
4. Python: duplicate detection by title similarity (deterministic)
5. Structured metrics report + log entry

### Lint System

15 health checks with SQLite-backed caching for incremental scanning:

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
01-Raw/              ← drop URLs, PDFs, files here
02-Clippings/        ← web clipper saves (already markdown)
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

Built automatically during `pipeline compile` from wikilinks, concept sources, and MoC membership.

## Semantic Search

Uses [qmd](https://github.com/tobi/qmd) with Qwen3-Embedding-0.6B-Q8 for concept matching. Falls back to keyword search if not installed.

```bash
# One-time setup
bash Meta/Scripts/setup-qmd.sh

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

434 tests covering: extraction, planning, creation, validation, lint, compile, vault operations, models, config, and integration.

## Recommended Workflow

**Daily:** Drop sources in `01-Raw/`, run `pipeline ingest ~/MyVault`

**Weekly:** `pipeline compile` → review entries → `pipeline lint --fix`

**Monthly:** Check `pipeline stats` for growth trends

## Architecture

Python-first. The pipeline (`pipeline/cli.py`) is the canonical entry point. Shell scripts in `scripts/` are setup utilities only.

```
pipeline/
├── cli.py              # typer CLI — all commands
├── extract.py          # Stage 1: URL routing, extraction, dedup
├── plan.py             # Stage 2: semantic search, agent planning
├── compile.py          # Compile pass: snapshot/diff, edges, report
├── lint.py             # 15 lint checks with cache support
├── vault.py            # File operations, collision detection
├── store.py            # SQLite content store + vault cache
├── config.py           # Configuration and environment
├── models.py           # Data models (Manifest, Plan, Edge, etc.)
├── qmd.py              # Semantic search integration
├── utils.py            # Shared utilities
├── extractors/         # Type-specific extractors
│   ├── web.py
│   ├── youtube.py
│   ├── podcast.py
│   └── _shared.py
└── create/             # Stage 3 creation
    ├── agent.py         # Agent execution
    ├── orchestrator.py  # Batch coordination + post-processing
    ├── prompts.py       # Prompt construction
    ├── templates.py     # Note templates
    └── validate.py      # Output validation + auto-repair
```

~7,000 lines of Python, 434 tests, 0 shell scripts in the critical path.
