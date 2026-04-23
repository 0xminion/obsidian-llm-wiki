# PRD: Obsidian LLM Wiki v0.2.0

## Executive Summary

Obsidian LLM Wiki is a self-contained pipeline that turns raw web content into a structured, interconnected Obsidian wiki. It uses LLM agents for semantic judgment (planning, cross-linking, concept merging) while keeping deterministic operations in Python code.

**Architecture:** 3-stage Python pipeline (Extract → Plan → Create) with parallel extraction, semantic concept search via qmd (Qwen3-Embedding-0.6B-Q8), parallel write agents, per-batch validation, compile pass with snapshot/diff metrics, incremental lint via SQLite cache, and auto-updating tag registry.

## Problem Statement

1. Extract content from diverse sources (web, X/Twitter, YouTube, podcasts, PDFs, arxiv)
2. Structure it into atomic, evergreen notes with typed relationships
3. Support bilingual content (Chinese sources stay Chinese, English stays English)
4. Maintain vault health through automated linting and indexing
5. Be self-contained — run one command, everything updates

## Architecture

### 3-Stage Pipeline

```
01-Raw/ → pipeline ingest → 04-Wiki/{sources, entries, concepts, mocs}
                                    ↓
                             Post-ingest auto-updates:
                             - tag-registry.md (tag usage counts)
                             - wiki-index.md (note summaries)
                             - edges.tsv (typed relationships)
                             - log.md (operation record)
```

**Stage 1: Extract** (Python, no LLM, ~1-5s per URL)
- Python extractors with type-specific fallback chains
- No LLM — deterministic, fast
- ThreadPoolExecutor parallel extraction
- SQLite content store for URL + content dedup
- Dead letter queue for failed extractions
- Output: `/tmp/obsidian-extracted-{hash}/{hash}.json` per URL

**Stage 2: Plan** (1 LLM agent, ~30-60s)
- Dedup check via Jaccard similarity against existing vault sources
- Semantic concept pre-search via qmd embeddings
- Tag vocabulary injection — existing tags passed to agent for reuse
- Single planning agent produces per-source creation plans
- Deterministic fallback for confident sources (language detection, template selection)
- Output: `/tmp/.../plans.json`

**Stage 3: Create** (N parallel LLM agents, ~60-120s per source)
- Parallel write agents (default 3, configurable `--parallel N`)
- Concept convergence via pre-fetched qmd semantic matches
- Per-batch validation immediately after creation:
  - Frontmatter completeness
  - Required sections per template type
  - Stub/placeholder detection (20+ patterns)
  - Minimum body length enforcement
  - Banned tag detection
- Auto-repair derives real content from existing body — never stubs
- Output: Files written to vault, inbox archived, wiki-index updated

### Compile Pass

Post-ingest or manual (`pipeline compile`). Separates semantic operations from deterministic ones:

| Operation | Type | Description |
|-----------|------|-------------|
| Cross-link analysis | LLM agent | Find notes that should link to each other |
| Concept merging | LLM agent | Merge near-duplicate concepts |
| MoC rebuild | LLM agent | Update Maps of Content |
| Wiki index rebuild | Python | Rebuild from vault files |
| Typed edges | Python | Build from wikilinks, sources, tags |
| Duplicate detection | Python | Title similarity comparison |
| Schema review | LLM agent | Evaluate note templates and lint rules |

Metrics: vault snapshot captured before/after agent run. Diff shows actual changes. Agent output parsed for crosslink/merge counts. Structured CompileResult returned with real metrics.

### Lint System

15 checks with SQLite-backed caching:

- **Incremental:** Wikilink index cached in `cache.db`. On subsequent runs, only re-reads files whose mtime changed.
- **Synonym detection:** Tags compared by normalized form (hyphenation, pluralization) to find duplicates.
- **Auto-fix:** frontmatter normalization, markdown format, banned tag removal.

### Pipeline Flags

| Flag | Purpose |
|------|---------|
| `--parallel N` | Number of parallel write agents (default: 3) |
| `--dry-run` | Preview without executing |
| `--review` | Run Stages 1+2, save plans for manual review |
| `--resume` | Skip Stages 1+2, use reviewed plans |

### Semantic Concept Search

Uses [qmd](https://github.com/tobi/qmd) with Qwen3-Embedding-0.6B-Q8 for semantic similarity.

- Search priority: daemon vector → CLI vector → BM25 (keyword fallback)
- Pipeline auto-detects availability, falls back gracefully if not installed

### Note Structures

**Entry** (standard, English): Summary → Core insights → Other takeaways → Diagrams → Open questions → Linked concepts

**Entry** (Chinese): 摘要 → 核心发现 → 其他要点 → 图表 → 开放问题 → 关联概念

**Entry** (technical): Summary → Key Findings → Data/Evidence → Methodology → Limitations → Linked concepts

**Concept** (evergreen): Core concept → Context (flowing prose) → Links

**MoC**: Topic-specific sections with synthesized summaries.

### Extraction Chain

| Source | Primary | Fallback |
|--------|---------|----------|
| Web | defuddle | curl extraction |
| X/Twitter | defuddle | curl extraction |
| YouTube | TranscriptAPI | Supadata → faster-whisper |
| Podcasts | AssemblyAI | whisper |
| PDFs | liteparse | OCR |

### Python Commands

| Command | Purpose |
|---------|---------|
| `pipeline ingest` | Full 3-stage pipeline (extract → plan → create) |
| `pipeline compile` | Concept convergence, MoC rebuild, edges, duplicate detection |
| `pipeline lint` | 15 health checks + synonym detection |
| `pipeline lint --fix` | Auto-fix safe issues |
| `pipeline validate` | Post-write quality gate |
| `pipeline validate --fix` | Auto-repair missing sections |
| `pipeline reindex` | Rebuild wiki-index.md |
| `pipeline stats` | Dashboard: size, growth, review status, health |
| `pipeline tags` | Rebuild tag-registry.md from actual usage |
| `pipeline query` | Q&A against vault knowledge base |
| `pipeline init` | Initialize or migrate vault structure |

### Data Models

| Model | Stage | Description |
|-------|-------|-------------|
| `ExtractedSource` | 1 | URL, title, content, type, author, hash |
| `Manifest` | 1 | Collection of ExtractedSource |
| `Plan` | 2 | Title, language, template, tags, concept targets, MoC targets |
| `Plans` | 2 | Collection of Plan, with content-size-aware batch splitting |
| `Edge` | compile | source, target, type, description |
| `ConceptMatch` | 2/concept | concept name, similarity score |
| `CompileResult` | compile | Structured metrics from compile pass |

### Content Store (SQLite)

Persistent at `/tmp/.../store.db` during pipeline run:

| Table | Purpose |
|-------|---------|
| `urls` | Extracted URL dedup with canonical URL normalization |
| `content` | Content hash dedup (detects same article from different URLs) |
| `dead_letter_queue` | Failed extractions with reason classification |
| `pending_reviews` | Review/resume workflow state |
| `vault_cache` | File mtime indices, wikilink graph, tag metadata |

### Vault Cache (SQLite)

Persistent at `Meta/Scripts/cache.db` across runs:

- File mtime indices per directory (incremental lint)
- Wikilink graph (fast orphan/link checks)
- Tag registry metadata

## Key Design Decisions

1. **Python-first** — all pipeline logic in Python. Shell scripts are setup utilities only.
2. **LLM for judgment, Python for facts** — agents handle semantic decisions (cross-linking, merging). Deterministic operations (reindex, edges, validation) run in Python.
3. **Per-batch validation** — files validated immediately after agent writes them, not in a separate pass.
4. **Snapshot/diff compile** — vault state captured before/after agent, diff shows actual changes.
5. **Incremental lint** — SQLite-cached wikilink index, only re-reads changed files.
6. **Tag vocabulary injection** — existing tags fed to planning agent for reuse.
7. **No stubs** — validation enforces real content. Auto-repair derives content from existing body.
8. **Self-contained** — no external cron. Run `pipeline ingest`, everything updates.
9. **Chinese stays Chinese** — body text stays in Chinese for Chinese sources.
10. **4-column edges** — `source<tab>target<tab>type<tab>description`.

## Acceptance Criteria

- [x] Pipeline handles URLs, YouTube, podcasts, PDFs
- [x] 3-stage pipeline: Extract → Plan → Create
- [x] Semantic concept search via qmd + Qwen3-Embedding-0.6B-Q8
- [x] `--review` and `--resume` flags for human-in-the-loop
- [x] `--parallel N` for configurable agent concurrency
- [x] Entry templates: standard, chinese, technical, comparison, procedural
- [x] Concept notes use evergreen format (3 sections + frontmatter sources)
- [x] No stub/placeholder content (lint + validation enforced)
- [x] Tags validated against blocklist + synonym detection
- [x] Edges use 4-column TSV format
- [x] Post-ingest auto-updates: tag-registry, wiki-index, edges, log
- [x] Prompts externalized in prompts/*.prompt files
- [x] Collision detection prevents note overwrites
- [x] 15 lint checks with incremental caching
- [x] Output validation with per-batch checking + auto-repair
- [x] Compile pass with snapshot/diff metrics
- [x] SQLite content store for dedup + DLQ
- [x] Vault cache for incremental processing
- [x] Tag vocabulary injected into plan prompt
- [x] Structured CompileResult with real metrics
- [x] 434 unit + integration tests

## Non-Goals

- RAG/vector search (wiki-index.md remains the retrieval layer)
- Multi-user collaboration (single-user)
- Web UI (Obsidian remains the viewer)
- Real-time sync (batch operations only)

## Testing

- **Unit tests**: Data models, config, extraction helpers, prompt building, validation
- **Integration tests**: Full pipeline with mocked agents, lint against real vault fixtures
- **Compile tests**: Snapshot capture, metrics parsing, result serialization
- **434 tests total, all passing**
