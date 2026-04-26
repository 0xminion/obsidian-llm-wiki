---
name: obsidian-ingest
description: |
  Canonical skill for agentic ingestion into obsidian-llm-wiki vaults.
  When user says "obsidian" + URL/PDF/YouTube/others → auto-ingest via pipeline.
  Drop content → run pipeline → structured, interlinked vault entries.
version: "2.0.0"
category: obsidian
trigger: |
  - "obsidian "
  - "vault "
  - "clip "
  - "ingest "
  - "save to obsidian"
  - "wiki "
  - "knowledge base"
  - URL patterns (https://, youtube.com, youtu.be, x.com, twitter.com)
  - "pdf "
  - "article "
  - "read later"
---

# Obsidian LLM Wiki — Agentic Ingestion

**Purpose:** Turn any URL, file, PDF, YouTube video, podcast, or tweet into a structured, interlinked Obsidian vault entry — automatically, via one command.

**Pipeline:** Extract → Plan → Create → Compile

---

## Quick Start (One-Shot Ingestion)

```bash
# 1. Set vault path (one-time)
export VAULT_PATH=~/MyVault

# 2. Drop a URL
python3 -c "print('https://example.com/article')" > "$VAULT_PATH/01-Raw/article.url"

# 3. Run the pipeline
pipeline ingest "$VAULT_PATH"
# Or with parallelism:  pipeline ingest --parallel 3 "$VAULT_PATH"
```

**That's it.** The pipeline handles extraction, planning, creation, cross-linking, and indexing.

---

## Supported Sources

| Source | Primary Extractor | Fallback Chain |
|--------|-----------------|---------------|
| Web Articles | defuddle | curl + liteparse → archive.org → Camoufox |
| YouTube | transcript API | supadata → faster-whisper |
| Podcasts | AssemblyAI | iTunes lookup + RSS fallback |
| X/Twitter | FxTwitter API (nitter) | liteparse + browser |
| PDFs | liteparse (pdftotext/poppler) | direct text extraction |
| Reddit | json endpoint | defuddle fallback |
| Clippings (markdown files) | direct ingest | — (bypasses Stage 1) |

**JS-heavy / Cloudflare sites:** Camoufox browser handles Medium, AKJournals, Substack paywalls.

---

## Vault Structure

```
01-Raw/              ← Drop .url files here (inbox)
02-Clippings/        ← Drop markdown clippings here (bypasses Stage 1)
03-Queries/          ← Drop .md questions for Q&A mode
04-Wiki/
├── sources/         ← Full original content (extracted)
├── entries/         ← Human-readable summaries + insights
├── concepts/        ← Atomic concept notes (evergreen)
└── mocs/            ← Topic hubs (Map of Contents)
05-Outputs/          ← Q&A responses
06-Config/
├── wiki-index.md    ← Auto-generated index
├── edges.tsv        ← Typed relationships between notes
├── tag-registry.md  ← Controlled vocabulary
└── log.md           ← Ingestion log
07-WIP/              ← Your drafts (pipeline ignores)
08-Archive-Raw/      ← Processed inbox items
09-Archive-Queries/  ← Answered queries
10-Archive-Clippings/ ← Processed clippings
Meta/
├── Scripts/         ← Pipeline code, .env, logs
├── Prompts/         ← Runtime prompt templates
└── Templates/       ← Note templates (Jinja2)
```

---

## Pipeline Stages

### Stage 1: Extract
Pure Python — no LLM. Routes URLs to type-specific extractors with validation and retry.

- Exponential backoff (max 3 attempts)
- SSRF-resistant URL validation
- SQLite dedup store (content hash)
- Dead letter queue for failures → `logs/extract_failures.jsonl`

### Stage 2: Plan
Lightweight LLM call — batched planning for all extracted sources.

1. **Dedup check** — Jaccard similarity against existing vault sources
2. **Semantic concept search** — Qdrant embeddings find related concepts
3. **Tag vocabulary injection** — existing tags passed to planner for consistency

Output: `plans.json` with title, language (EN/ZH), template, tags, concept targets, MoC assignments.

### Stage 3: Create
~90% Python / ~10% LLM. Deterministic templates + bounded insights.

- Jinja2 templates for structure (no agent subprocess per note)
- Parallel insight pre-generation via `ThreadPoolExecutor`
- Smart filename generation for long titles (LLM-assisted, semantic)
- Post-creation validation: 15 health checks (frontmatter, stubs, links, tags)

**Fast path:** ~3.8 min for 18 URLs (was ~9.5 min with agent-per-note mode).

### Stage 4: Compile
Cross-linking and index rebuild.

1. **Semantic cross-link analysis** — embedding similarity + LLM validation adds missing `[[wikilinks]]`
2. **Concept merging** — detects near-duplicates, merges with LLM approval, updates all references
3. **MoC rebuild** — resynthesizes topic hubs from related notes
4. **Wiki index rebuild** — deterministic scan of all entries, concepts, MoCs
5. **Typed edges export** — `edges.tsv` from wikilinks, sources, tags (9 relationship types)
6. **Duplicate detection report** — title similarity for human review

---

## CLI Commands

| Command | Purpose |
|---------|---------|
| `pipeline ingest ~/MyVault` | Full pipeline: extract → plan → create |
| `pipeline ingest --parallel 3 ~/MyVault` | Use 3 parallel workers for Stage 3 |
| `pipeline ingest --dry-run ~/MyVault` | Preview plans without writing |
| `pipeline ingest --review ~/MyVault` | Halt after Stage 2 for human review of plans |
| `pipeline ingest --resume ~/MyVault` | Continue from reviewed plans.json |
| `pipeline compile ~/MyVault` | Run compile pass (cross-link, index, edges) |
| `pipeline lint ~/MyVault` | 15 health checks on vault |
| `pipeline lint --fix ~/MyVault` | Auto-fix safe issues |
| `pipeline validate ~/MyVault` | Post-write quality gate |
| `pipeline validate --fix ~/MyVault` | Auto-repair missing sections |
| `pipeline reindex ~/MyVault` | Rebuild wiki-index.md |
| `pipeline stats ~/MyVault` | Show vault growth dashboard |
| `pipeline tags ~/MyVault` | Rebuild tag registry |
| `pipeline query --ask "question" ~/MyVault` | Natural language Q&A against vault |
| `pipeline query --ask "question" --fast ~/MyVault` | Fast direct LLM query (< 5s) |

---

## Configuration

Set in `~/MyVault/Meta/Scripts/.env`:

```bash
# Required
VAULT_PATH=~/MyVault

# LLM Provider (choose one)
# Option 1: Ollama (default) — fast, private, local
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_INSIGHT_MODEL=minimax-m2.7:cloud     # for summaries/insights
OLLAMA_FILENAME_MODEL=minimax-m2.7:cloud    # for smart filenames
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b   # for concept embeddings

# Option 2: OpenRouter — access 200+ models
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-sonnet-4
OPENROUTER_API_KEY=sk-...

# Option 3: Hermes (legacy) — full agent with tool access
LLM_PROVIDER=hermes
AGENT_CMD=hermes                          # slower, use when tool access needed

# Extraction APIs (optional — improves quality)
TRANSCRIPT_API_KEY=***          # YouTube transcripts
SUPADATA_API_KEY=***            # YouTube metadata
ASSEMBLYAI_API_KEY=***          # Podcast transcription

# Parallelism
PARALLEL=3                       # default workers for Stage 3
```

---

## Critical Rules

1. **Never touch `07-WIP/`** — pipeline leaves this directory alone
2. **Never overwrite existing notes** — collision detection appends `-1`, `-2`, etc.
3. **No stubs** — every section must have real content at creation time
4. **Tags: topic-specific English only** — never platform names (`x.com`, `tweet`, `source`)
5. **Chinese body stays Chinese** — YAML frontmatter and tags in English only
6. **YAML wikilinks quoted** — `source: "[[note]]"` not `source: [[note]]`
7. **Filenames match content language** — Chinese titles → Chinese filenames, English → kebab-case
8. **Never use URL slugs as filenames** — derive from semantic title
9. **Collision rename capped at 100** — prevents infinite loops
10. **Stage 3 timeout ≠ failure** — agent subprocess timeout (900s) may leave complete files; always check vault before re-running

---

## Clippings Workflow (02-Clippings)

For content already extracted by other tools (Readwise, Obsidian Web Clipper, etc.):

```bash
# Drop markdown clippings here
cp my-article.md ~/MyVault/02-Clippings/

# Run pipeline — skips Stage 1 extraction, goes straight to Plan → Create
pipeline ingest ~/MyVault
```

**Key differences from URL flow:**
- Content already extracted (no HTTP calls)
- `_strip_quotes()` helper for titles with wrapping quotes
- Source URL regex strips trailing punctuation (prevents `url:` field contamination)
- `\b` word boundaries for type detection (prevents `fakeyoutube.com` matching as YouTube)
- Skips `source.save()` — no orphaned `.json` sidecars
- `collect_clipping_files()` called directly (no `_collect_clipping_files` wrapper)
- `vault.py` imports: `json`, `hashlib`, `logging` — watch for F821 (undefined name) lint

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ollama._types.ResponseError: 429` | Too many concurrent requests. Drop `PARALLEL` to 2 or use batch `/api/embed` endpoint. |
| Camoufox crash after ~100 URLs | Memory leak. Browser recycles every 50 venues. Check `logs/extract_failures.jsonl` for batch status. |
| defuddle returns 403 | Fallback chain: curl+liteparse → archive.org → Camoufox. Check `logs/extract_failures.jsonl`. |
| `plans.json` corruption | Remove `plans.json` and re-run without `--resume`. |
| Empty notes created | Run `pipeline lint --fix`. Likely Stage 3 LLM timeout — increase timeout or switch to template mode. |
| Broken wikilinks | Run `pipeline compile` to regenerate cross-links. |
| Duplicate concepts | Check `06-Config/edges.tsv` for `relates` edges. Merge via `pipeline compile` concept merging. |
| Slow Stage 3 (~9.5 min for 18 URLs) | Switch to template mode (default): ~90% Python, ~10% LLM. Parallel insight pre-generation enabled. |

---

## Architecture Highlight

**Hybrid design: Python does the heavy lifting, LLM for semantic work only.**

| Phase | Python | LLM |
|-------|--------|-----|
| Stage 1 (Extract) | 100% | 0% |
| Stage 2 (Plan) | ~60% | ~40% (embeddings + planning) |
| Stage 3 (Create) | ~90% | ~10% (insights + filenames only) |
| Post-Creation Validation | 100% | 0% |
| Compile Pass | ~70% | ~30% (semantic cross-linking, merging) |
| Query Command | 0% | 100% (agentic research) |

**Performance gains:**
- Stage 3: ~9.5 min → ~3.8 min for 18 URLs (parallel insight pre-gen)
- Embeddings: ~2.3 min → ~0.4 min (batch `/api/embed` endpoint)
- Extraction: ~8 min → ~5 min (Camoufox recycling every 50 URLs)

---

## Related Skills

- `obsidian-llm-wiki-architecture` — Full architecture reference, design rationale, process maps
- `obsidian-llm-wiki-ops` — Operational guide: cron jobs, health monitoring, Qdrant diagnostics, lint system
