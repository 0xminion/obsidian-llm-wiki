# obsidian-llm-wiki

LLM-powered knowledge compiler for Obsidian vaults — give it any link, get a
structured, interlinked wiki with typed relationships, tagged concepts, and
deterministic markdown rendering.

`obsidian-llm-wiki` ingests web articles, YouTube videos, PDFs, podcasts,
scientific papers, X/Twitter posts, and plain text — synthesises them via LLM
into structured concepts with typed cross-references — and renders an Obsidian
vault with YAML frontmatter, wikilinks, machine-readable `relations[]`, and MOC
groupings. All markdown generation is deterministic; the LLM only produces
the intermediate synthesis JSON.

---

## Architecture

```
Sources / URLs / Files  →  Extract  →  Synthesise  →  Merge  →  Render
   (web, YouTube,          (registry)  (LLM calls)   (corpus   (deterministic
    PDF, DOCX, text,                    + cache)      dedup)    markdown)
    podcasts, arXiv,                                  backlinks
    X/Twitter, Nature                                 + graph
         ↓                     ↓            ↓             ↓          ↓
    SourceDoc            extractors    SynthesisBundle  dedupe   Obsidian vault
                         web.py        (typed JSON)    .py      (wikilinks,
                         youtube.py                             frontmatter,
                         pdf.py                                 relations[])
                         scientific.py
                         podcast.py
```

### Key design decisions

- **Synthesis cache** (`.llmwiki/cache/<filename>.json`) — incremental builds
  reuse cached syntheses for unchanged sources, so the rendered corpus is
  always complete — not just the subset that changed in this run.
- **Orphan detection** — when a source is deleted, its exclusively-owned
  concepts get `orphaned: true` in frontmatter. Shared concepts are preserved.
- **Full-corpus rendering** — `render_vault()` always receives the complete
  set of sources, not just changed ones. This prevents silent data loss.
- **Typed relationships** — concepts carry `related: [{slug, relation, display}]`
  with normalised relation types (`variant_of`, `depends_on`, `component_of`,
  etc.). These render as both human-readable `[[slug|display]] — \`relation\``
  in the body and machine-readable `relations[]` in frontmatter.
- **Backlink propagation** — all concept edges are automatically made
  bidirectional. If A→B exists, B→A is added with the same relation type.
- **Bilingual title normalization** — Chinese-derived titles are
  deterministically reformatted to English-first bilingual format
  (`English Title (中文标题)`) at render time, even when the LLM doesn't comply.
- **Gradient confidence scoring** — thin concepts get a continuous confidence
  score (0.1–1.0) based on body length, replacing the old binary 0.3/1.0
  threshold.
- **Content chunking** — sources above `CHUNK_SIZE` (default 30K chars) are
  split into paragraph-boundary chunks. Each chunk is independently
  synthesised, and skeletons are merged before Pass 2 expansion.

### The Synthesis Contract

The LLM produces one structured JSON object per source containing everything
the renderers need:

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

In two-pass mode, Pass 1 extracts a concept skeleton (title, slug, rationale),
then Pass 2 expands each concept with a focused prompt producing 300+ word
sections. A quality gate assigns gradient confidence scores to thin concepts.

### Semantic features

- **Cross-lingual embedding links** — when `EMBEDDINGS_ENABLED=true`, concepts
  across languages are linked via cosine similarity (threshold 0.60) using
  `embeddinggemma:300m` embeddings.
- **Semantic concept deduplication** — same-language concepts with >0.85
  cosine similarity are merged automatically.
- **Embedding-based MoC assignment** — orphan concepts (not in any MoC) are
  assigned to the most semantically similar MoC.
- **Graph visualization** — `graph.json` (D3.js/Obsidian compatible) and
  `graph.mmd` (Mermaid) are exported to `.llmwiki/`.
- **Vault health report** — `olw health` checks for broken wikilinks, orphan
  concepts, stub entries, low-confidence concepts, and tag violations.

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
| X/Twitter posts & articles | defuddle.md → VxTwitter API | included | full tweet text + metadata |
| JATS/XML (academic) | `defusedxml` | included | structured XML extraction |
| Plain text / markdown | built-in | — | direct file read |

Install all optional extractors: `pip install obsidian-llm-wiki[all]`

The extractor registry auto-detects source type from URL domain or file
extension. Unknown URLs fall back to web extraction (defuddle.md → trafilatura
→ Wayback Machine). The registry uses a fail-closed policy: if a specialized
extractor matches a URL but fails, it does NOT silently fall through to web
extraction (which would produce garbage from cookie-walled pages).

### YouTube extraction

YouTube transcript extraction uses a multi-layer fallback chain:

1. **yt-dlp subtitle download** (primary) — downloads auto-generated or manual
   subtitles via `yt-dlp` CLI, parses VTT/SRT into plain text
2. **AssemblyAI remote-URL transcription** (secondary) — gets the direct audio
   stream URL via yt-dlp, submits to AssemblyAI for ASR transcription
3. **Invidious API** (metadata fallback) — fetches video description
4. **oEmbed API** (last resort) — title + channel only

This replaces the previous Supadata-only approach, which failed when the API
key was missing or rate limits were hit.

### Podcast transcript resolution

Podcast ingestion is deliberately cache-first and publisher-first:

```
local transcript cache
  → RSS podcast:transcript artifact (VTT/SRT/HTML/plain text/JSON)
  → publisher page ## Transcript section via defuddle.md
  → AssemblyAI remote URL transcription
  → local faster-whisper (only after media acquisition succeeds)
```

Spotify and Apple episode URLs are resolved through the canonical RSS feed
when possible. Podcast Index discovery is optional.

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

# Query the vault (RAG-style)
olw query ~/MyVault --ask "What is a transformer model?"

# Check vault health
olw health ~/MyVault

# Validate the vault
olw validate ~/MyVault
olw validate ~/MyVault --strict   # broken wikilinks = errors

# View pipeline metrics
olw metrics ~/MyVault
```

---

## Commands

| Command | Description |
|---------|-------------|
| `olw setup` | Interactive setup wizard — configures LLM provider, vault path |
| `olw ingest` | Ingest URLs + clippings → extract → synthesise → render |
| `olw build` | Re-synthesise changed sources (incremental, cache-backed) and re-render |
| `olw query` | RAG-style question answering against the vault |
| `olw validate` | Check vault for conformance (frontmatter, wikilinks, strict mode) |
| `olw health` | Vault health report (broken links, orphans, stubs, low confidence) |
| `olw metrics` | Print pipeline run metrics summary |
| `olw recompile` | Retry one source with bounded truncation recovery |
| `olw fix` | Preview or explicitly apply conservative backed-up maintenance fixes |
| `olw providers check` | Inspect endpoint, authentication state, and task-model routing |
| `olw providers models` | List models exposed by the configured LLM provider |

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
│   ├── index.md           ← Bundle-root index
│   └── .llmwiki/          ← Internal state
│       ├── state.json     ← Source hash database for incremental builds
│       ├── lock           ← Compile-time PID lock
│       ├── cache/         ← Synthesis cache (one JSON per source)
│       ├── metrics.json   ← Pipeline run metrics (last 50 runs)
│       ├── graph.json     ← Knowledge graph (D3.js/Obsidian compatible)
│       ├── graph.mmd      ← Mermaid graph diagram
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

The `relations[]` frontmatter array stores `slug|relation|display` strings.
It is machine-readable for Obsidian Dataview queries and graph visualisation
plugins without triggering nested-object Properties warnings.

### MOC pages

Maps of Content group related concepts with:
- Bilingual headings when the MoC contains both English and Chinese concepts
- Cross-reference diagrams showing typed relationships between concepts
- Cross-lingual embedding links merged into the Concepts list

A single MoC can deliberately contain English and Chinese concepts from the
same umbrella. With embeddings enabled, a high-confidence cross-language
semantic sibling of any existing MoC member is added to that MoC during the
render pass, so the page, graph export, and health checks agree on membership.
Use a multilingual embedding model for this; a monolingual model makes this
feature decorative rather than useful.

### Source provenance and Obsidian Properties

Source pages retain requested, resolved, and extracted URLs, extraction stages,
content type, retrieval time, hash, and diagnostics in the `provenance`
property. It is serialized as a flat list of readable strings rather than a
nested YAML object, because Obsidian's Properties pane warns on nested objects.
Existing source pages with the legacy nested representation remain readable.

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

# Vault
VAULT_PATH=$HOME/MyVault

# Content thresholds
MAX_SOURCE_CHARS=1000000
MIN_SOURCE_CHARS=50

# Chunking (two-pass mode)
CHUNK_SIZE=30000

# Concurrency
COMPILE_CONCURRENCY=3

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
SIMILARITY_DEDUP_THRESHOLD=0.85
MOC_ASSIGNMENT_THRESHOLD=0.55

# Retry
RETRY_COUNT=3
RETRY_BASE_MS=1000
RETRY_MULTIPLIER=4

# Extraction APIs (optional)
ASSEMBLYAI_API_KEY=...       # YouTube + podcast transcription
PODCAST_INDEX_API_KEY=...    # Podcast feed discovery
PODCAST_INDEX_API_SECRET=... # paired secret
RESIDENTIAL_PROXY_URL=...    # SOCKS5/HTTP proxy for blocked sites
```

### LLM Providers

| Provider | `LLM_PROVIDER` | Auth | Endpoints |
|----------|---------------|------|-----------|
| **Ollama** (local) | `ollama` | None | `/api/chat` |
| **OpenAI-compatible** | `openai` | `Bearer` API key | `/v1/chat/completions` |

For OpenAI, OpenRouter, LM Studio, vLLM, etc., set `LLM_PROVIDER=openai`,
`LLM_API_KEY=<key>`, and `LLM_HOST=<endpoint>`.

The Ollama client passes `context_window` as `num_ctx` to the Ollama API
automatically — required for cloud models like `gemma4:31b-cloud` that
support 256K token context.

### Embeddings: what they do and what they do not do

Embeddings are opt-in and fail closed: if the local embedding endpoint is
unavailable or returns an invalid vector, normal synthesis and rendering still
complete without semantic deduplication, automatic MoC assignment, or
cross-lingual expansion. When healthy, the pipeline uses embeddings for:

1. same-language near-duplicate concept merging;
2. assigning otherwise orphaned concepts to the most similar MoC; and
3. connecting cross-language semantic siblings under the same MoC.

Set `EMBEDDING_MODEL` to a model that supports every language in the vault.
For English+Chinese vaults, use a multilingual embedding model such as the
locally installed `qwen3-embedding:0.6b` rather than assuming a default model
is bilingual. The model and `LLM_HOST` are resolved at call time after the
vault `.env` is loaded, so per-vault settings actually take effect.

### Extraction fallbacks and access-controlled sources

Set `DEEP_SEARCH_FALLBACK=true` to let a failed or stub web extraction search
Semantic Scholar, OpenAlex, arXiv, and Crossref for an accessible equivalent.
It is deliberately off by default because it adds external network calls and
may recover metadata/abstracts rather than the publisher's full text. For
YouTube age/consent walls, set `YOUTUBE_COOKIES_FILE` to a local Netscape
cookies file. Proxy routing is opt-in through `RESIDENTIAL_PROXY_URL`; do not
set a global `HTTPS_PROXY` unless that broad routing is actually intended.

---

## Package Structure

```
obsidian_llm_wiki/
  cli/                     # Typer CLI commands
    ingest.py              # olw ingest — extract + synthesise + render
    build.py               # olw build — incremental re-synthesis
    query.py               # olw query — RAG-style Q&A
    validate.py            # olw validate — conformance checks
    health.py              # olw health — vault health report
    setup.py               # olw setup — interactive wizard
    ops.py                 # olw metrics, olw recompile
  core/
    models.py              # SynthesisBundle schema + serializers + RelationType
    pipeline.py            # Full-corpus orchestrator with cache + orphan logic
    cache.py               # Synthesis cache + resynthesis overlay
    orphan.py              # Orphan detection (exclusive vs shared concepts)
    state.py               # Incremental compilation state (SHA-256 hashes)
    lock.py                # PID-based compile lock
    metrics.py             # Pipeline run metrics (extractions, syntheses, rendering)
  ingest/
    web.py                 # Multi-layer web extraction (defuddle.md → trafilatura → Wayback)
    sources.py             # Source loading from sources/ directory
    clippings.py           # Clippings quality gate
    proxy.py               # SOCKS5/HTTP proxy support
    http_headers.py        # Centralized browser UA + headers
    liteparse.py           # LiteParse CLI integration for structured documents
    supadata_utils.py      # Supadata rate limiting + usage tracking
    podcast_index.py       # Podcast Index feed discovery
    transcript_resolver.py # Cache-first podcast transcript acquisition
    alt_source.py          # Invidious, Semantic Scholar, journal fallbacks
    extractors/
      __init__.py          # Registry with @register_extractor + fail-closed dispatch
      youtube.py           # yt-dlp subtitles + AssemblyAI + oEmbed fallback
      pdf.py               # pymupdf (fitz) + LiteParse fallback
      docx.py              # python-docx
      jats.py              # JATS/XML academic articles
      podcast.py           # Spotify/Apple/RSS podcast extraction
      scientific.py        # arXiv accessible HTML + citation discovery
      twitter.py           # X/Twitter via defuddle.md + VxTwitter
  synth/
    prompts.py             # Single-pass synthesis prompt + few-shot examples
    quality.py             # Two-pass synthesis (extract → expand) + chunking
    parser.py              # JSON → SynthesisBundle validation + extraction
    dedupe.py              # Concept merge, tag normalization, backlinks, semantic dedup
    embedding.py           # Cross-lingual embedding links via Ollama
    language.py            # Language detection + bilingual instructions
  render/
    obsidian.py            # Deterministic Obsidian markdown renderer
    frontmatter.py         # YAML frontmatter, wikilinks, tags, atomic I/O
    bilingual.py           # English-first bilingual title normalization
    crossrefs.py           # Cross-reference diagram builder
    graph_export.py        # JSON + Mermaid graph visualization export
  providers/
    llm.py                 # Ollama + OpenAI-compatible clients with retry
  config.py                # Configuration management (.env loading)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check obsidian_llm_wiki tests
```

### Test Suite

The test suite covers:

- **Golden end-to-end test** (`test_golden_pipeline.py`) — full pipeline with
  mocked LLM, asserts vault structure, frontmatter, wikilinks, and state
- **Extended golden tests** (`test_golden_extended.py`) — two-pass mode,
  empty concepts, malformed JSON, multi-source merge, backlink propagation,
  gradient confidence boundaries, chunk content boundaries, frontmatter
  robustness, URL classification, rendering golden tests
- **Incremental cache test** (`test_incremental_cache.py`) — multi-source cache
  reuse + orphan detection on source deletion
- **Two-pass quality tests** (`test_quality.py`) — mocked two-pass flow,
  quality gate enforcement, pipeline dispatch
- **Extractor tests** (`test_extractors.py`) — registry dispatch, fallback,
  video ID extraction, URL routing
- **Semantic dedup tests** (`test_semantic_dedupe.py`) — embedding-based
  concept merging, MoC orphan assignment
- **Model tests** (`test_models.py`, `test_models_phase4.py`) — relation
  normalization, empty section rejection, RelationType enum
- **Environment isolation** (`conftest.py`) — autouse fixture prevents
  `load_dotenv` pollution between tests

### Test automation patterns

The test suite follows standard test automation patterns
([Wikipedia: Test automation](https://en.wikipedia.org/wiki/Test_automation)):

| Pattern | Implementation |
|---------|---------------|
| Golden/snapshot tests | `test_golden_pipeline_end_to_end`, `test_two_pass_golden` |
| Boundary value tests | `test_gradient_confidence_boundaries`, `test_chunk_content_boundaries` |
| Error/exception tests | `test_malformed_json_response`, `test_empty_concepts_golden` |
| Equivalence class tests | `test_url_classification` (journal XML, SSRN, YouTube) |
| Regression tests | `test_backlink_propagation_golden`, `test_multi_source_merge_golden` |
| State transition tests | `test_golden_pipeline_incremental_skip` |

---

## Obsidian desktop bridge (optional)

`obsidian-plugin/` is a thin desktop-only bridge around the installed `olw`
CLI; it does not duplicate compiler logic. Build it with `npm ci && npm run
build`, then copy `manifest.json` and `main.js` into your vault's
`.obsidian/plugins/obsidian-llm-wiki-bridge/` directory and enable it in
Obsidian. It provides commands for URL ingest/preview, query, health checks,
maintenance-fix previews, result history, and cancellation. Configure the
`olw` executable in the plugin settings if it is not on Obsidian's PATH.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally (default), or any
  **OpenAI-compatible API** endpoint
- Python dependencies: `typer`, `httpx[socks]`, `pyyaml`, `python-dotenv`,
  `trafilatura`, `defusedxml`
- Optional: `yt-dlp` (YouTube), `pymupdf` (PDF), `python-docx` (DOCX),
  `liteparse` (structured documents)