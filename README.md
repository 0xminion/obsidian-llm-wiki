# obsidian-llm-wiki

LLM-powered knowledge compiler for Obsidian vaults — give it any link, get a
structured, interlinked wiki with typed relationships, tagged concepts, and
deterministic markdown rendering.

`obsidian-llm-wiki` ingests web articles, YouTube videos, PDFs, Word
documents, and plain text — synthesises them via LLM into structured
concepts with typed cross-references — and renders an Obsidian vault with
YAML frontmatter, wikilinks, machine-readable `relations[]`, and MOC
groupings. All markdown generation is deterministic; the LLM only produces
the intermediate synthesis JSON.

---

## Architecture

```
Sources / URLs / Files  →  Extract  →  Synthesise  →  Merge  →  Render
   (web, YouTube,          (registry)  (LLM calls)   (corpus   (deterministic
    PDF, DOCX, text,                    + cache)      dedup)    markdown)
         ↓                     ↓            ↓             ↓          ↓
    SourceDoc            extractors    SynthesisBundle  dedupe   Obsidian vault
                         web.py        (typed JSON)    .py      (wikilinks,
                         youtube.py                             frontmatter,
                         pdf.py                                 relations[])
                         docx.py
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
sections. A quality gate flags thin concepts (`confidence: 0.3`).

---

## Supported Sources

| Source | Extractor | Optional dep | Install |
|--------|-----------|--------------|---------|
| Web articles / blogs / news | `trafilatura` | included | — |
| YouTube videos | Supadata → metadata fallback | API key optional | transcript + metadata |
| Podcasts / RSS | cache → RSS transcript → AssemblyAI → Supadata → Whisper | API keys optional | publisher or generated transcript |
| PDF files | `pymupdf` (fitz) | `pip install obsidian-llm-wiki[pdf]` | full text with page markers |
| Scientific reports (arXiv / publishers) | official accessible HTML → official PDF | `pymupdf` for PDF fallback | structured public full text when available |
| Word `.docx` | `python-docx` | `pip install obsidian-llm-wiki[docx]` | text with heading structure |
| Plain text / markdown | built-in | — | direct file read |

Install all optional extractors: `pip install obsidian-llm-wiki[all]`

The extractor registry auto-detects source type from URL domain or file
extension. Unknown URLs fall back to web extraction (trafilatura).

### Scientific reports and public-access boundaries

For an arXiv URL, ingestion parses the paper identifier and first requests the
official accessible rendition at `https://arxiv.org/html/<paper-id>`. arXiv
HTML availability is partial, so an unavailable, short, or unextractable HTML
conversion falls back to the official `https://arxiv.org/pdf/<paper-id>` through
the existing PDF document extractor.

When an inaccessible publisher landing page advertises an official direct
full-text URL through `citation_fulltext_html_url`, `citation_pdf_url`, or a
same-publisher PDF link, the extractor may follow that public document link.
It does not use cookies, credentials, paywall bypasses, or unlicensed mirrors.
For SSRN, no publicly accessible document means the existing Semantic Scholar
metadata/abstract fallback is used rather than attempting to evade access
controls.

### Podcast transcript resolution

Podcast ingestion is deliberately cache-first and publisher-first:

```text
local transcript cache
  → RSS podcast:transcript artifact (VTT/SRT/HTML/plain text/JSON)
  → publisher page `## Transcript` section via defuddle.md
  → AssemblyAI remote URL transcription
  → Supadata remote media transcription
  → local faster-whisper (only after media acquisition succeeds)
```

Spotify and Apple episode URLs are resolved through the canonical RSS feed
when possible. The RSS enclosure gives remote providers a public media URL;
the pipeline does not scrape first-party player transcript UIs. Publisher
transcripts are preferred over generated ASR output and every acquired
transcript is cached under `.llmwiki/transcripts/` with provenance.

When configured, Podcast Index is the first discovery pass for Spotify, Apple,
and supported generic podcast pages. It searches for publisher RSS feeds, then
the pipeline verifies the requested episode title inside the candidate feed —
it never treats a directory search result as proof of a matching episode. A
miss falls through to the platform-specific iTunes/RSS path.

Configure transcript providers in the vault-root `.env` (never commit this
file):

```bash
ASSEMBLYAI_API_KEY=...    # first remote provider for public RSS enclosures
SUPADATA_API_KEY=...      # platform-specialized remote fallback
PODCAST_INDEX_API_KEY=...     # optional free canonical-feed discovery
PODCAST_INDEX_API_SECRET=...  # paired Podcast Index developer secret
```

AssemblyAI fetches public media from its own infrastructure, avoiding a local
audio download. It cannot bypass authentication, DRM, paywalls, or an origin
that blocks AssemblyAI itself.

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

# Validate the vault
olw validate ~/MyVault
olw validate ~/MyVault --strict   # broken wikilinks = errors
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
│       └── cache/         ← Synthesis cache (one JSON per source)
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
timestamp: 2026-07-09T10:00:00Z
relations:
  - target: sgd
    type: variant_of
    display: SGD
  - target: learning-rate
    type: depends_on
    display: Learning Rate
---
```

Cross-links use Obsidian wikilinks with typed edge annotations:
`[[sgd|SGD]] — \`variant_of\``

The `relations[]` frontmatter array is machine-readable for Obsidian Dataview
queries and graph visualisation plugins.

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

# Vault
VAULT_PATH=$HOME/MyVault

# Content thresholds
MAX_SOURCE_CHARS=1000000
MIN_SOURCE_CHARS=50

# Concurrency
COMPILE_CONCURRENCY=3

# Language (en, zh, or empty for auto-detect)
OUTPUT_LANGUAGE=

# Synthesis mode
SYNTHESIS_MODE=single        # single | two_pass

# Retry
RETRY_COUNT=3
RETRY_BASE_MS=1000
RETRY_MULTIPLIER=4

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
  cli/                   # Typer CLI (ingest, build, query, setup, validate)
  core/
    models.py            # SynthesisBundle schema + serializers + RelationType
    pipeline.py          # Full-corpus orchestrator with cache + orphan logic
    cache.py             # Synthesis cache (per-source JSON persistence)
    orphan.py            # Orphan detection (exclusive vs shared concepts)
    state.py             # Incremental compilation state
    lock.py              # PID-based compile lock
  ingest/
    web.py               # httpx + trafilatura web extraction
    clippings.py         # Clippings quality gate
    sources.py           # Source loading from sources/ directory
    extractors/
      __init__.py        # Registry with @register_extractor pattern
      youtube.py         # yt-dlp + youtube-transcript-api
      pdf.py             # pymupdf (fitz)
      docx.py            # python-docx
  synth/
    prompts.py           # Single-pass synthesis prompt builder
    quality.py           # Two-pass quality synthesis (extract → expand)
    parser.py            # JSON → SynthesisBundle validation + empty section filter
    dedupe.py            # Corpus-level concept/tag reconciliation
  render/
    obsidian.py          # Deterministic Obsidian markdown renderer
  providers/
    llm.py               # Ollama + OpenAI-compatible clients
  config.py              # Configuration management
```

---

## Development

```bash
pip install -e ".[dev]"
pytest                     # 434 tests (320 legacy + 114 new)
ruff check obsidian_llm_wiki pipeline tests tests/new
```

### Tests

The test suite includes:

- **Golden end-to-end test** (`test_golden_pipeline.py`) — full pipeline with
  mocked LLM, asserts vault structure, frontmatter, wikilinks, and state.
- **Incremental cache test** (`test_incremental_cache.py`) — multi-source cache
  reuse + orphan detection on source deletion.
- **Extractor tests** (`test_extractors.py`) — registry dispatch, fallback,
  video ID extraction.
- **Two-pass quality tests** (`test_quality.py`) — mocked two-pass flow,
  quality gate enforcement, pipeline dispatch.
- **Model tests** (`test_models_phase4.py`) — relation normalization, empty
  section rejection, RelationType enum.

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally (default), or any
  **OpenAI-compatible API** endpoint
- Python dependencies: `typer`, `httpx[socks]`, `pyyaml`, `python-dotenv`,
  `trafilatura`
- Optional: `yt-dlp`, `youtube-transcript-api` (YouTube),
  `pymupdf` (PDF), `python-docx` (DOCX)