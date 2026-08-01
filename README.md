# obsidian-llm-wiki

LLM-powered knowledge compiler for Obsidian vaults — give it any link, get a structured, interlinked wiki with typed relationships, tagged concepts, and deterministic markdown rendering.

`obsidian-llm-wiki` ingests web articles, YouTube videos, PDFs, podcasts, scientific papers, X/Twitter posts, and plain text — synthesises them via LLM into structured concepts with typed cross-references — and renders an Obsidian vault with YAML frontmatter, wikilinks, machine-readable `relations[]`, and MOC groupings. All markdown generation is deterministic; the LLM only produces the intermediate synthesis JSON.

**Version 3.0** — implements the [Five-Year Scalability Roadmap](docs/scalability-roadmap.md) for a target vault of **5,000 sources, 50,000 concepts, and 20,000 MoCs**.

---

## What's New in v3

The v3 rewrite replaced the monolithic JSON/filesystem architecture with a SQLite-backed incremental runtime. Key changes:

| v1/v2 constraint | v3 solution |
|---|---|
| Full concept index in synthesis prompts | Bounded FTS5 + ANN retrieval (50–200 candidates) |
| Pairwise O(N²) semantic dedupe | Multi-table LSH ANN with exact cosine reranking |
| Full-vault render rewrites | Path-incremental rendering with write journals |
| JSON state/cache files | SQLite WAL registry with versioned entities |
| Sequential ingest dedup scans | Indexed URL/content hash lookups |
| Full health scans every build | Incremental round-robin health with explicit full audit |

**947 tests** | **~20,800 lines of Python** | **83 source modules** | MIT licensed

---

## Architecture

```
Sources / URLs / Files  →  Extract  →  Synthesise  →  Merge  →  Render
   (web, YouTube,          (registry)  (LLM calls)   (corpus   (deterministic
    PDF, DOCX, text,                    + cache)      dedup)    markdown)
    podcasts, arXiv,                                  backlinks
    X/Twitter, Nature                                 + graph)
         ↓                     ↓            ↓             ↓          ↓
    SourceDoc            extractors    SynthesisBundle  dedupe   Obsidian vault
                         web.py        (typed JSON)    .py      (wikilinks,
                         youtube.py                             frontmatter,
                         pdf.py                                 relations[])
                         scientific.py
                         podcast.py
```

### Key design decisions

- **Synthesis cache** (`.llmwiki/cache/<filename>.json`) — incremental builds reuse cached syntheses for unchanged sources, so the rendered corpus is always complete — not just the subset that changed in this run.
- **Orphan detection** — when a source is deleted, its exclusively-owned concepts get `orphaned: true` in frontmatter. Shared concepts are preserved.
- **Full-corpus rendering** — `render_vault()` always receives the complete set of sources, not just changed ones. This prevents silent data loss.
- **Typed relationships** — concepts carry `related: [{slug, relation, display}]` with normalised relation types (`variant_of`, `depends_on`, `component_of`, etc.). These render as both human-readable `[[slug|display]] — \`relation\`` in the body and machine-readable `relations[]` in frontmatter.
- **Backlink propagation** — all concept edges are automatically made bidirectional. If A→B exists, B→A is added with the same relation type.
- **Bilingual title normalization** — Chinese-derived titles are deterministically reformatted to English-first bilingual format (`English Title (中文标题)`) at render time, even when the LLM doesn't comply.
- **Gradient confidence scoring** — thin concepts get a continuous confidence score (0.1–1.0) based on body length, replacing the old binary 0.3/1.0 threshold.
- **Content chunking** — sources above `CHUNK_SIZE` (default 30K chars) are split into paragraph-boundary chunks. Each chunk is independently synthesised, and skeletons are merged before Pass 2 expansion.

### The Synthesis Contract

The LLM produces one structured JSON object per source containing everything the renderers need:

```json
{
  "source_title": "Attention Is All You Need",
  "source_summary": "Introduces the Transformer architecture...",
  "source_tags": ["deep-learning", "attention"],
  "key_points": ["Self-attention eliminates recurrence", ...],
  "concepts": [
    {
      "title": "Self-Attention",
      "slug": "self-attention",
      "summary": "Attention mechanism relating positions in a sequence.",
      "tags": ["attention", "neural-network"],
      "sections": [{"heading": "Core", "points": ["..."]}],
      "related": [{"slug": "multi-head-attention", "relation": "component_of"}],
      "claims": [{"text": "Complexity is O(n^2)", "source_ref": "section 3.2"}],
      "confidence": 0.95
    }
  ],
  "maps": [
    {"title": "Attention Mechanisms", "slug": "attention-mechanisms",
     "concept_slugs": ["self-attention", "multi-head-attention"]}
  ]
}
```

### Synthesis modes

| Mode | `SYNTHESIS_MODE` | LLM calls | When to use |
|------|-------------------|-----------|-------------|
| **Single-pass** (default) | `single` | 1 per source | Fast, good for short articles |
| **Two-pass quality** | `two_pass` | 1 + N (per concept) | Deep, evidence-backed sections |

In two-pass mode, Pass 1 extracts a concept skeleton (title, slug, rationale), then Pass 2 expands each concept with a focused prompt producing 300+ word sections. A quality gate assigns gradient confidence scores to thin concepts. Pass 2 uses evidence-selected source windows (not the entire source per concept) under one shared request/rate limiter.

### Incremental Architecture (v2 scalability)

The v3 runtime is built around the SQLite registry (`04-Wiki/.llmwiki/registry.db`, WAL mode, schema v3):

| Component | What it does |
|---|---|
| **Registry** | Source hashes, canonical URLs, artifact versions, concept stable IDs, serialized syntheses, float32 embeddings, dirty/retry queues, render manifests, ingest hashes, maintenance cursor |
| **FTS5 search** | Bounded lexical concept retrieval replaces full-corpus prompt context |
| **ANN index** | Multi-table sparse random-hyperplane LSH with exact cosine reranking — replaces O(N²) pairwise similarity |
| **Version contract** | Every record carries extractor/prompt/schema/model/embedding versions; incompatible versions create explicit dirty queue entries |
| **Write journal** | Lazy preimage journal enables rollback on render failure while preserving concurrent human edits |
| **Sharded artifacts** | Directory indexes, graph JSON, and reports shard above configured thresholds; Mermaid disabled above monolith threshold |
| **Incremental health** | Round-robin page batches by default; `olw health --full` remains the explicit complete audit |
| **Durable ingest** | Indexed URL/content dedupe, bounded extraction workers, failed-URL retry queue with next-attempt time. `olw ingest --resume` reconstructs interrupted operations from SQLite and releases dead-run leases |

### Runtime Budgets

Defaults are deliberately conservative for a small server:

| Budget | Default | Env var |
|---|---|---|
| Synthesis/extraction concurrency | 3 | `COMPILE_CONCURRENCY` |
| Prompt candidates per source | 100 | `CANDIDATE_LIMIT` |
| Prompt candidate characters | 16,000 | `CANDIDATE_CHAR_BUDGET` |
| ANN exact rerank pool | 1,000 | `ANN_CANDIDATE_POOL` |
| Pass 2 evidence characters | 15,000 | `EVIDENCE_WINDOW_CHARS` |
| Index/report shard size | 2,000 | `INDEX_SHARD_SIZE` |
| Graph shard size | 5,000 | `GRAPH_SHARD_SIZE` |
| Graph monolith threshold | 10,000 | `GRAPH_MONOLITH_THRESHOLD` |
| Max source chars | 1,000,000 | `MAX_SOURCE_CHARS` |
| Concept min body | 1,200 chars | `CONCEPT_MIN_BODY_CHARS` |

### Semantic features

- **Cross-lingual embedding links** — when `EMBEDDINGS_ENABLED=true`, concepts across languages are linked via cosine similarity (threshold 0.60) using `embeddinggemma:300m` embeddings.
- **Semantic concept deduplication** — same-language concepts with >0.85 cosine similarity are merged automatically.
- **Embedding-based MoC assignment** — orphan concepts (not in any MoC) are assigned to the most semantically similar MoC.
- **Graph visualization** — `graph.json` (D3.js/Obsidian compatible) and `graph.mmd` (Mermaid) are exported to `.llmwiki/`. Above the monolith threshold, graph JSON is sharded and Mermaid becomes a summary.
- **Vault health report** — `olw health` checks for broken wikilinks, orphan concepts, stub entries, low-confidence concepts, and tag violations.

Embeddings are opt-in and fail closed: if the local embedding endpoint is unavailable or returns an invalid vector, normal synthesis and rendering still complete without semantic deduplication, automatic MoC assignment, or cross-lingual expansion.

Set `EMBEDDING_MODEL` to a model that supports every language in the vault. For English+Chinese vaults, use a multilingual embedding model such as the locally installed `qwen3-embedding:0.6b` rather than assuming a default model is bilingual.

---

## Supported Sources

| Source | Extractor | Optional dep | Install |
|--------|-----------|--------------|---------|
| Web articles / blogs / news | defuddle.md → trafilatura → Wayback | included | — |
| YouTube videos | yt-dlp subtitles → AssemblyAI → oEmbed | `yt-dlp` | transcript + metadata |
| Podcasts (Spotify/Apple/RSS) | cache → RSS transcript → AssemblyAI | API keys optional | publisher or generated transcript |
| PDF files | `pymupdf` (fitz) → LiteParse | `pip install obsidian-llm-wiki[pdf]` | full text with page markers |
| arXiv papers | official accessible HTML → official PDF | `pymupdf` for PDF fallback | structured full text |
| Scientific publishers | citation meta discovery → LiteParse | included | same-site public documents |
| Word `.docx` | `python-docx` | `pip install obsidian-llm-wiki[docx]` | text with heading structure |
| X/Twitter posts & articles | defuddle CLI → defuddle.md → trafilatura | included | full tweet text + metadata |
| JATS/XML (academic) | `defusedxml` | included | structured XML extraction |
| Notion exports | native parser | included | structured content |
| Plain text / markdown | built-in | — | direct file read |

Install all optional extractors: `pip install obsidian-llm-wiki[all]`

The extractor registry auto-detects source type from URL domain or file extension. Unknown URLs fall back to web extraction (defuddle.md → trafilatura → Wayback Machine). The registry uses a fail-closed policy: if a specialized extractor matches a URL but fails, it does NOT silently fall through to web extraction (which would produce garbage from cookie-walled pages).

### YouTube extraction

YouTube transcript extraction uses a multi-layer fallback chain:

1. **yt-dlp subtitle download** (primary) — downloads auto-generated or manual subtitles via `yt-dlp` CLI, parses VTT/SRT into plain text
2. **AssemblyAI remote-URL transcription** (secondary) — gets the direct audio stream URL via yt-dlp, submits to AssemblyAI for ASR transcription
3. **Invidious API** (metadata fallback) — fetches video description
4. **oEmbed API** (last resort) — title + channel only

### Podcast transcript resolution

Podcast ingestion is deliberately cache-first and publisher-first:

```
local transcript cache
  → RSS podcast:transcript artifact (VTT/SRT/HTML/plain text/JSON)
  → publisher page ## Transcript section via defuddle.md
  → AssemblyAI remote URL transcription
  → local faster-whisper (only after media acquisition succeeds)
```

Spotify and Apple episode URLs are resolved through the canonical RSS feed when possible. Podcast Index discovery is optional.

### Extraction fallbacks and access-controlled sources

Set `DEEP_SEARCH_FALLBACK=true` to let a failed or stub web extraction search Semantic Scholar, OpenAlex, arXiv, and Crossref for an accessible equivalent. It is deliberately off by default because it adds external network calls and may recover metadata/abstracts rather than the publisher's full text. For YouTube age/consent walls, set `YOUTUBE_COOKIES_FILE` to a local Netscape cookies file. Proxy routing is opt-in through `RESIDENTIAL_PROXY_URL`; do not set a global `HTTPS_PROXY` unless that broad routing is actually intended.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/0xminion/obsidian-llm-wiki.git
cd obsidian-llm-wiki
pip install -e .

# Optional: install YouTube/PDF/DOCX extractors
pip install -e ".[all]"

# Interactive setup
olw setup

# Ingest any link — YouTube, PDF, web article, etc.
olw ingest ~/MyVault --url https://youtube.com/watch?v=...
olw ingest ~/MyVault --url https://arxiv.org/abs/1706.03762
olw ingest ~/MyVault -u URL1 -u URL2 --parallel 5

# Re-build after manual edits to sources (incremental, uses cache)
olw build ~/MyVault
olw build ~/MyVault --force    # force re-synthesis of all
olw build ~/MyVault --dry-run  # detect changes without writing

# Query the vault (RAG-style)
olw query ~/MyVault --ask "What is a transformer model?"

# Check vault health
olw health ~/MyVault
olw health ~/MyVault --full     # explicit complete audit
olw health ~/MyVault --json     # typed findings as JSON

# Validate the vault
olw validate ~/MyVault
olw validate ~/MyVault --strict   # broken wikilinks = errors

# Maintenance
olw fix ~/MyVault --dry-run     # preview planned repairs
olw fix ~/MyVault --apply       # apply deterministic fixes only

# View pipeline metrics
olw metrics ~/MyVault
```

---

## Commands

| Command | Description |
|---------|-------------|
| `olw setup` | Interactive setup wizard — configures LLM provider, vault path |
| `olw ingest` | Ingest URLs + collect clippings → extract → write source files |
| `olw build` | Re-synthesise changed sources (incremental, cache-backed) and re-render |
| `olw query` | Grounded Q&A against the vault with retrieval + citations |
| `olw validate` | Check vault for conformance (frontmatter, wikilinks, strict mode) |
| `olw health` | Vault health report (incremental scan; `--full` for complete audit) |
| `olw fix` | Preview or explicitly apply conservative backed-up maintenance fixes |
| `olw metrics` | Print pipeline run metrics summary |
| `olw recompile` | Retry one source with bounded truncation recovery |
| `olw providers check` | Inspect endpoint, authentication state, and task-model routing |
| `olw providers models` | List models exposed by the configured LLM provider |

### Ingestion flags

```bash
olw ingest ~/MyVault --url URL        # single URL
olw ingest ~/MyVault -u URL1 -u URL2  # multiple URLs
olw ingest ~/MyVault --parallel 5     # more extraction workers (hard-capped at 8)
olw ingest ~/MyVault --json           # machine-readable structured result
olw ingest ~/MyVault --resume         # resume interrupted ingestion safely
```

### Query flags

```bash
olw query ~/MyVault -a "question" -n 5           # limit retrieval to 5 pages
olw query ~/MyVault -a "question" --json         # machine-readable result
olw query ~/MyVault -a "question" --profile research  # use a query profile
olw query ~/MyVault -a "question" --session planning # continue a session
olw query ~/MyVault -a "question" --save-answer result.md
```

### Fix flags

```bash
olw fix ~/MyVault --dry-run              # preview planned repairs (default)
olw fix ~/MyVault --apply               # apply only non-reviewed deterministic fixes
olw fix ~/MyVault --restore <backup>    # restore a trusted backup
olw fix ~/MyVault --json                 # machine-readable plan or result
```

---

## Vault Structure

```
{Vault}/
├── 02-Clippings/          ← Pre-extracted markdown (Obsidian Web Clipper)
├── 04-Wiki/               ← Wiki root
│   ├── sources/           ← Original source content
│   ├── entries/           ← LLM-synthesised analysis pages
│   ├── concepts/          ← Evergreen atomic notes (with wikilinks + relations)
│   ├── mocs/              ← Maps of Content (topic groupings)
│   ├── views/             ← Generated Dataview/Bases query pages
│   ├── index.md           ← Bundle-root index
│   └── .llmwiki/          ← Internal state
│       ├── registry.db    ← SQLite WAL registry (authoritative index)
│       ├── lock            ← Compile-time PID lock
│       ├── cache/         ← Synthesis cache (one JSON per source)
│       ├── backups/       ← Automated rewrite backups (bounded retention)
│       ├── metrics.json   ← Pipeline run metrics
│       ├── graph.json     ← Knowledge graph (D3.js/Obsidian compatible)
│       ├── graph.mmd      ← Mermaid graph diagram
│       ├── health-report.md
│       ├── query-profiles.json
│       └── transcripts/   ← Cached podcast transcripts
└── .env                   ← Configuration
```

### Concept frontmatter

Every concept page carries YAML frontmatter with typed relations:

```yaml
---
type: Concept
title: Gradient Descent
tags: [machine-learning, optimization]
aliases: [GD, batch gradient descent]
confidence: 0.95
provenance: extracted
timestamp: 2026-07-13T10:00:00Z
relations:
  - sgd|variant_of|SGD
  - learning-rate|depends_on|Learning Rate
---
```

Cross-links use Obsidian wikilinks with typed edge annotations:
`[[sgd|SGD]] — \`variant_of\``

The `relations[]` frontmatter array stores `slug|relation|display` strings. It is machine-readable for Obsidian Dataview queries and graph visualisation plugins without triggering nested-object Properties warnings.

### MOC pages

Maps of Content group related concepts with:
- Bilingual headings when the MoC contains both English and Chinese concepts
- Cross-reference diagrams showing typed relationships between concepts
- Cross-lingual embedding links merged into the Concepts list

A single MoC can deliberately contain English and Chinese concepts from the same umbrella. With embeddings enabled, a high-confidence cross-language semantic sibling of any existing MoC member is added to that MoC during the render pass, so the page, graph export, and health checks agree on membership. Use a multilingual embedding model for this; a monolingual model makes this feature decorative rather than useful.

### Source provenance and Obsidian Properties

Source pages retain requested, resolved, and extracted URLs, extraction stages, content type, retrieval time, hash, and diagnostics in the `provenance` property. It is serialized as a flat list of readable strings rather than a nested YAML object, because Obsidian's Properties pane warns on nested objects. Existing source pages with the legacy nested representation remain readable.

---

## Maintenance System

### Health (`olw health`)

Typed, machine-readable maintenance findings:

| Finding kind | What it detects |
|---|---|
| `broken_relation` | Wikilink/relation target to non-existent note |
| `orphan_concept` | Concept with zero incoming links |
| `moc_assignment_candidate` | Orphaned concept that could join a MoC |
| `tag_normalization` | Tags needing normalization (case, synonyms) |
| `alias_candidate` | Note that could benefit from an alias |
| `empty_generated_stub` | Generated page with no real content |
| `unmanaged_generated_page` | Old generated page absent from current render manifest |
| `broken_body_link` | Body wikilink to missing note |
| `short_concept` | Concept body below minimum |
| `low_confidence` | Concept with confidence < threshold |
| `missing_provenance` | Source page without provenance |
| `undersized_moc` | MoC with too few members |

Default mode scans incrementally (round-robin batches); `--full` performs a complete audit.

### Fix (`olw fix`)

Explicit, backed-up, deterministic repairs — **never** auto-merges pages, assigns MoCs, or modifies reviewed content:

- `--dry-run` (default): print the plan without writing
- `--apply`: execute only deterministic fixes (broken relations, tag normalization, empty stub removal)
- `--restore <path>`: restore a trusted backup
- `--json`: machine-readable plan or result

### Contradictions

Durable records with explicit human-governed lifecycle:

`detected` → `review_ok` / `pending_fix` → `resolved` / `suppressed`

The store records evidence and source revisions but contains no conflict-resolution logic. Terminal decisions are never silently reopened.

### Review protection

Pages marked `reviewed: true` in frontmatter retain their human-authored body during automated rewrites. Frontmatter and provenance may still be updated — only the body is protected.

---

## Query System

`olw query` provides grounded answers with deterministic citations:

1. **Retrieval** — lexical tokenization (CJK n-grams + Latin words) + personalized PageRank over the wiki graph. Graph maturity guardrails prevent PPR from outranking lexical signals on small/sparse graphs.
2. **Context building** — assembles retrieved pages into a grounded prompt with extractable cited paths.
3. **Citation validation** — verifies answer citations resolve to real vault paths.
4. **Sessions** — continue or name query sessions for multi-turn research.
5. **Profiles** — vault-local query profiles with custom instructions.
6. **Save answers** — write results to `03-Queries/` or a custom path.

---

## Configuration

All configuration is via a `.env` file in the vault root (created by `olw setup`):

```bash
# LLM Provider
LLM_PROVIDER=ollama          # ollama | openai
LLM_HOST=http://localhost:11434
LLM_MODEL=gemma3:27b
LLM_API_KEY=                  # for openai providers
LLM_TIMEOUT_MS=1800000        # 30 minutes
LLM_CONTEXT_WINDOW=256000     # 256K tokens (for cloud models)

# Task-specific model overrides (optional — falls back to LLM_MODEL)
LLM_INGEST_MODEL=gemma4:31b-cloud
LLM_MAINTENANCE_MODEL=gemma3:27b
LLM_QUERY_MODEL=gpt-4o
LLM_EXPAND_MODEL=gemma3:27b   # Pass 2 expansion (two-pass mode)

# Vault
VAULT_PATH=$HOME/MyVault

# Content thresholds
MAX_SOURCE_CHARS=1000000
MIN_SOURCE_CHARS=50

# Chunking (two-pass mode)
CHUNK_SIZE=30000

# Concurrency
COMPILE_CONCURRENCY=3
REQUEST_MIN_INTERVAL_MS=0

# Five-year scalability budgets (5k sources / 50k concepts / 20k MoCs)
CANDIDATE_LIMIT=100
CANDIDATE_CHAR_BUDGET=16000
ANN_CANDIDATE_POOL=1000
EVIDENCE_WINDOW_CHARS=15000
INDEX_SHARD_SIZE=2000
REPORT_SHARD_SIZE=2000
GRAPH_SHARD_SIZE=5000
GRAPH_MONOLITH_THRESHOLD=10000

# Language (en, zh, or empty for auto-detect)
OUTPUT_LANGUAGE=

# Synthesis mode
SYNTHESIS_MODE=single        # single | two_pass

# Quality gates
CONCEPT_MIN_BODY_CHARS=1200
ENTRY_MIN_BODY_CHARS=500

# Semantic features
EMBEDDINGS_ENABLED=false
EMBEDDING_MODEL=embeddinggemma:300m
EMBEDDING_HOST=               # falls back to LLM_HOST
SIMILARITY_DEDUP_THRESHOLD=0.85
CROSS_LINGUAL_THRESHOLD=0.82
MOC_ASSIGNMENT_THRESHOLD=0.55

# Retry
RETRY_COUNT=3
RETRY_BASE_MS=1000
RETRY_MULTIPLIER=4

# Extraction APIs (optional)
ASSEMBLYAI_API_KEY=...       # YouTube + podcast transcription
TRANSCRIPT_API_KEY=...       # YouTube Transcript API
SUPADATA_API_KEY=...         # Supadata transcript fallback
PODCAST_INDEX_API_KEY=...    # Podcast feed discovery
PODCAST_INDEX_API_SECRET=... # paired secret
RESIDENTIAL_PROXY_URL=...    # SOCKS5/HTTP proxy for blocked sites
DEEP_SEARCH_FALLBACK=false   # scholarly deep search fallback
```

### LLM Providers

| Provider | `LLM_PROVIDER` | Auth | Endpoints |
|----------|---------------|------|-----------|
| **Ollama** (local) | `ollama` | None | `/api/chat` |
| **OpenAI-compatible** | `openai` | `Bearer` API key | `/v1/chat/completions` |

For OpenAI, OpenRouter, LM Studio, vLLM, etc., set `LLM_PROVIDER=openai`, `LLM_API_KEY=<key>`, and `LLM_HOST=<endpoint>`.

The Ollama client passes `context_window` as `num_ctx` to the Ollama API automatically — required for cloud models like `gemma4:31b-cloud` that support 256K token context.

---

## Package Structure

```
obsidian_llm_wiki/
├── config.py                # Configuration management (.env loading)
├── cli/                     # Typer CLI commands
│   ├── ingest.py            # olw ingest — extract + write source files
│   ├── build.py             # olw build — incremental re-synthesis + render
│   ├── query.py             # olw query — grounded Q&A
│   ├── validate.py          # olw validate — conformance checks
│   ├── health.py            # olw health — vault health + maintenance findings
│   ├── fix.py               # olw fix — deterministic maintenance
│   ├── setup.py             # olw setup — interactive wizard
│   ├── ops.py               # olw metrics, olw recompile
│   └── providers.py         # olw providers check/models
├── core/
│   ├── models.py            # SynthesisBundle schema + serializers + RelationType
│   ├── pipeline.py          # Full-corpus orchestrator with cache + orphan logic
│   ├── registry.py          # SQLite WAL registry — schema v3, ANN, FTS5, queues
│   ├── cache.py             # Synthesis cache + resynthesis overlay
│   ├── state.py             # Incremental compilation state (SHA-256 hashes)
│   ├── lock.py              # PID-based compile lock
│   ├── metrics.py           # Pipeline run metrics
│   ├── orphan.py            # Orphan detection (exclusive vs shared concepts)
│   ├── maintenance.py       # Typed findings + deterministic fix plans
│   ├── backups.py           # Atomic, bounded backups for rewrites
│   ├── contradictions.py    # Durable contradiction records + lifecycle
│   ├── review.py            # Reviewed-page protection
│   ├── request_limiter.py   # Shared async rate limiter
│   ├── schema.py            # Synthesis policy + granularity selection
│   ├── evidence.py          # Synthesis evidence resolution
│   ├── source_content.py    # Bounded source content
│   ├── source_files.py      # Validated source-file paths
│   └── task_models.py       # Per-task model resolution
├── ingest/
│   ├── web.py               # Multi-layer web extraction (defuddle → trafilatura → Wayback)
│   ├── sources.py           # Source loading from sources/ directory
│   ├── clippings.py         # Clippings quality gate
│   ├── connectors.py       # Source connector boundary + dispatcher
│   ├── documents.py         # Safe fetch boundary for local/remote docs
│   ├── provenance.py        # Immutable provenance stamping
│   ├── url_safety.py        # SSRF-resistant URL validation
│   ├── proxy.py             # SOCKS5/HTTP proxy support
│   ├── http_headers.py      # Centralized browser UA + headers
│   ├── liteparse.py         # LiteParse CLI integration
│   ├── supadata_utils.py    # Supadata rate limiting + usage tracking
│   ├── podcast_index.py     # Podcast Index feed discovery
│   ├── transcript_resolver.py # Cache-first podcast transcript acquisition
│   ├── alt_source.py        # Invidious, Semantic Scholar, journal fallbacks
│   └── extractors/
│       ├── __init__.py      # Registry with @register_extractor + fail-closed dispatch
│       ├── youtube.py       # yt-dlp subtitles + AssemblyAI + oEmbed fallback
│       ├── pdf.py           # pymupdf (fitz) + LiteParse fallback
│       ├── docx.py          # python-docx
│       ├── jats.py          # JATS/XML academic articles
│       ├── notion.py        # Notion export parser
│       ├── podcast.py       # Spotify/Apple/RSS podcast extraction
│       ├── scientific.py    # arXiv accessible HTML + citation discovery
│       └── twitter.py       # X/Twitter via defuddle CLI + defuddle.md
├── synth/
│   ├── prompts.py           # Single-pass synthesis prompt + few-shot examples
│   ├── quality.py           # Two-pass synthesis (extract → expand) + chunking
│   ├── parser.py            # JSON → SynthesisBundle validation + extraction
│   ├── dedupe.py            # Concept merge, tag normalization, backlinks, semantic dedup
│   ├── embedding.py         # Cross-lingual embedding links via Ollama
│   ├── retrieval.py         # Bounded FTS5 + ANN concept retrieval
│   ├── evidence_windows.py # Deterministic bounded evidence windows for Pass 2
│   └── language.py          # Language detection + bilingual instructions
├── render/
│   ├── obsidian.py          # Deterministic Obsidian markdown renderer
│   ├── frontmatter.py       # YAML frontmatter, wikilinks, tags, atomic I/O
│   ├── bilingual.py         # English-first bilingual title normalization
│   ├── crossrefs.py         # Cross-reference diagram builder
│   ├── graph_export.py      # Sharded JSON + Mermaid graph visualization export
│   ├── dataview.py          # Dataview/Obsidian Bases view generators
│   ├── source_graph.py      # Source dependency graph
│   └── log.py               # Pipeline run log rendering
├── query/
│   ├── retrieval.py         # Lexical + personalized PageRank retrieval
│   ├── graph.py             # Deterministic wiki graph
│   ├── context.py           # Context building for grounded answers
│   ├── sessions.py          # Query sessions
│   └── profiles.py          # Vault-local query profiles
└── providers/
    └── llm.py               # Ollama + OpenAI-compatible clients with retry
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check obsidian_llm_wiki tests
```

### Test Suite

**947 tests** covering:

- **Golden end-to-end test** (`test_golden_pipeline.py`) — full pipeline with mocked LLM, asserts vault structure, frontmatter, wikilinks, and state
- **Extended golden tests** (`test_golden_extended.py`) — two-pass mode, empty concepts, malformed JSON, multi-source merge, backlink propagation, gradient confidence boundaries, chunk content boundaries, frontmatter robustness, URL classification, rendering golden tests
- **Incremental cache test** (`test_incremental_cache.py`) — multi-source cache reuse + orphan detection on source deletion
- **Two-pass quality tests** (`test_quality.py`) — mocked two-pass flow, quality gate enforcement, pipeline dispatch
- **Extractor tests** (`test_extractors.py`) — registry dispatch, fallback, video ID extraction, URL routing
- **Semantic dedup tests** (`test_semantic_dedupe.py`) — embedding-based concept merging, MoC orphan assignment
- **Scalability tests** (`test_scalability_*.py`) — synthetic 5k/50k/20k fixture, ANN, registry, retrieval, incremental render, synthesis bounds, retry pipeline, ingest concurrency
- **Registry tests** — schema migration, version invalidation, dirty queue, ingest dedupe, concept ownership
- **Health/fix tests** — typed findings, deterministic fix plans, contradiction lifecycle, backups, review protection
- **Query tests** — retrieval, PPR, sessions, profiles, citation validation
- **Model tests** (`test_models.py`, `test_models_phase4.py`) — relation normalization, empty section rejection, RelationType enum
- **Environment isolation** (`conftest.py`) — autouse fixture prevents `load_dotenv` pollution between tests

### Test automation patterns

The test suite follows standard test automation patterns ([Wikipedia: Test automation](https://en.wikipedia.org/wiki/Test_automation)):

| Pattern | Implementation |
|---------|---------------|
| Golden/snapshot tests | `test_golden_pipeline_end_to_end`, `test_two_pass_golden` |
| Boundary value tests | `test_gradient_confidence_boundaries`, `test_chunk_content_boundaries` |
| Error/exception tests | `test_malformed_json_response`, `test_empty_concepts_golden` |
| Equivalence class tests | `test_url_classification` (journal XML, SSRN, YouTube) |
| Regression tests | `test_backlink_propagation_golden`, `test_multi_source_merge_golden` |
| State transition tests | `test_golden_pipeline_incremental_skip` |

### Scalability capacity tests

`tests/new/test_scalability_*.py` exercise the synthetic 5k/50k/20k fixture and verify:
- No O(N²) path runs in incremental mode
- ANN exact-rerank evaluations stay within the configured pool
- Dirty queue claims are bounded and resume after process reopen
- One changed source produces a bounded render plan and bounded shard writes
- Prompt candidate, evidence, graph shard, and request concurrency budgets hold

---

## Critical Rules

1. **Never touch `07-WIP/`** — automation never reads or writes this directory
2. **Never overwrite reviewed pages** — `reviewed: true` preserves human-authored body
3. **No stubs** — every section must have real content at creation
4. **Tags**: topic-specific English only, never platform names (`x.com`, `tweet`, `source`)
5. **Chinese body stays Chinese** in all `04-Wiki` notes
6. **YAML wikilinks must be quoted**: `source: "[[note]]"`
7. **File names match content language** (Chinese titles → Chinese filenames)
8. **Never use URL slugs as filenames** — titles are semantically converted
9. **Backups before rewrites** — every automated rewrite backs up to `.llmwiki/backups/`
10. **Registry is authoritative** — normal builds use indexed rows, not filesystem scans

---

## Obsidian desktop bridge (optional)

`obsidian-plugin/` is a thin desktop-only bridge around the installed `olw` CLI; it does not duplicate compiler logic. Build it with `npm ci && npm run build`, then copy `manifest.json` and `main.js` into your vault's `.obsidian/plugins/obsidian-llm-wiki-bridge/` directory and enable it in Obsidian. It provides commands for URL ingest/preview, query, health checks, maintenance-fix previews, result history, and cancellation. Configure the `olw` executable in the plugin settings if it is not on Obsidian's PATH.

---

## CI

GitHub Actions (`.github/workflows/ci.yml`):
- **Lint + test** — Python 3.11/3.12/3.13 matrix, Ruff + pytest
- **LiteParse contract** — integration test with the `liteparse` extra

---

## Recommended Workflow

**Daily:** Drop sources in `01-Raw/`, run `olw ingest ~/MyVault && olw build ~/MyVault`

**Weekly:** `olw health ~/MyVault` → review findings → `olw fix ~/MyVault --apply` for safe repairs

**Monthly:** `olw health ~/MyVault --full` for complete audit → `olw metrics ~/MyVault` for growth trends

---

## Documentation

- [Five-Year Scalability Roadmap](docs/scalability-roadmap.md) — v2 architecture design and acceptance criteria
- [Implementation Plan](docs/plans/2026-07-31-five-year-scalability-v2.md) — TDD delivery slices and capacity gates
- [Knowledge Engine Hardening Plan](docs/plans/2026-07-13-knowledge-engine-hardening.md) — provenance, safety, maintenance, and query design
- [Evidence Connectors + Render Integrity](docs/plans/2026-07-14-evidence-connectors-render-integrity.md) — Phase 2 hardening

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally (default), or any **OpenAI-compatible API** endpoint
- Python dependencies: `typer`, `httpx[socks]`, `pyyaml`, `python-dotenv`, `trafilatura`, `defusedxml`
- Optional: `yt-dlp` (YouTube), `pymupdf` (PDF), `python-docx` (DOCX), `liteparse` (structured documents)