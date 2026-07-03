# okf-pipeline

OKF v0.1-compliant knowledge compiler — sources in, interlinked OKF bundle out.

`okf-pipeline` is a Python-first, agent-native CLI that takes raw web sources
(URLs, clippings) and compiles them into a portable, interlinked knowledge
bundle conforming to the [OKF v0.1](#what-is-okf) specification. It supports
Ollama and any OpenAI-compatible LLM provider, generates an interactive
Cytoscape.js graph visualization, crawls the web for enrichment, and exports
self-contained tarballs for distribution.

---

## What is OKF?

**OKF** (Open Knowledge Format) v0.1 is a vendor-neutral specification for
portable knowledge bundles. The core design principles are:

- **Markdown + YAML** — every page is a plain markdown file with optional YAML
  frontmatter. No proprietary formats, no databases, no lock-in.
- **Vendor-neutral** — the bundle format is independent of the tool that
  produced it. Any OKF-compliant reader can open any OKF bundle.
- **Portable** — a bundle is just a directory tree (or tarball). Copy it, zip
  it, host it on a static site, or open it locally in any text editor.
- **No SDK required** — consuming an OKF bundle requires nothing more than a
  markdown reader. The format is self-describing via frontmatter `type` fields
  and standard markdown cross-links.

Each OKF page carries YAML frontmatter with a required `type` field
(`Source`, `Entry`, `Concept`, `Map of Content`, `Reference`) and optional
metadata (`title`, `description`, `tags`, `timestamp`, `resource`). Cross-links
use standard markdown link syntax (`[label](/concepts/foo.md)`), not
proprietary wikilink syntax.

---

## Architecture

The pipeline is organised into three stages:

```
Sources / URLs / Clippings  →  Stage 1 (Deterministic)  →  Stage 2 (LLM)  →  Stage 3 (Compile)
                                Extract full content      Entry → Concepts   Resolve links + Index
                                         ↓                       ↓                ↓
                                   sources/*.md          entries/*.md        index.md, log.md
                                                         concepts/*.md        viz.html (optional)
                                                         mocs/*.md
```

### Stage 1: Extraction (Deterministic, No LLM)

- Web URLs are fetched via [defuddle](https://github.com/nousresearch/defuddle)
  with an archive.org fallback.
- Full content is extracted — never truncated.
- SHA-256 dedup against existing sources prevents reprocessing identical
  content.
- Pre-extracted clippings in `02-Clippings/` that pass the quality gate
  (body > 500 chars + has title) skip extraction entirely and go straight to
  Stage 2.

### Stage 2: Creation (LLM — One Call Per Item)

- **One entry per source** — comprehensive analysis with all insights surfaced.
- **One concept per concept** — evergreen atomic notes (800+ chars minimum,
  never stubs).
- **One MoC per topic** — meaningful cross-references, not shallow link dumps.
- LLM calls are batched with configurable concurrency.

### Stage 3: Compile (Deterministic + LLM)

- Standard markdown cross-link resolver (rule-based, bidirectional).
- Per-directory and bundle-root `index.md` generation (deterministic).
- `log.md` changelog with ISO 8601 date headings.
- Orphan management for deleted sources.
- Incremental: only changed sources are re-processed.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/0xminion/obsidian-llm-wiki.git
cd obsidian-llm-wiki
pip install -e .

# Interactive setup (configures LLM provider, vault path, API keys)
okf setup

# Ingest URLs and compile the bundle
okf ingest ~/MyVault --url https://example.com/article

# Or ingest multiple URLs with parallel LLM calls
okf ingest ~/MyVault -u URL1 -u URL2 --parallel 5

# Recompile after manual edits to sources
okf compile ~/MyVault

# Lint the bundle for OKF conformance
okf lint ~/MyVault

# Generate an interactive HTML graph visualization
okf visualize ~/MyVault

# Crawl web seeds and mint reference pages
okf enrich ~/MyVault --web-seed https://example.com/article

# Export the bundle to a portable tarball
okf export ~/MyVault
```

---

## Commands

| Command | Description |
|---------|-------------|
| `okf setup` | Interactive setup wizard — configures LLM provider, vault path, API keys, scaffolds bundle |
| `okf ingest` | Ingest URLs + clippings → extract → create → compile in one pass |
| `okf compile` | Detect changes in sources, regenerate pages, resolve links, rebuild index |
| `okf lint` | Run OKF v0.1 conformance checks (OKF-001 through OKF-007) |
| `okf query` | RAG-style question answering against the knowledge bundle |
| `okf candidates` | Review pending candidate pages (list / show / approve / reject) |
| `okf visualize` | Generate a self-contained Cytoscape.js HTML graph (`viz.html`) |
| `okf enrich` | Crawl web seeds and mint/enrich reference pages via LLM agent |
| `okf export` | Pack the OKF bundle into a portable gzipped tarball |
| `okf import` | Extract an OKF tarball into a target directory with lint verification |
| `okf migrate` | Migrate a legacy Obsidian vault to OKF v0.1 format in place |

---

## OKF Bundle Structure

```
{Vault}/
├── 02-Clippings/          ← Pre-extracted markdown (Obsidian Web Clipper)
├── 04-Wiki/               ← OKF bundle root
│   ├── sources/           ← Full original content (Stage 1 output)
│   ├── entries/           ← LLM-generated analysis pages
│   ├── concepts/          ← Evergreen atomic notes
│   ├── mocs/              ← Maps of Content (topic overviews)
│   ├── references/        ← Enrichment-minted reference pages
│   ├── index.md           ← Bundle-root index (frontmatter: okf_version)
│   ├── log.md             ← Compilation changelog (ISO 8601 date headings)
│   ├── viz.html           ← Interactive Cytoscape.js graph (optional)
│   └── .llmwiki/          ← Internal pipeline state (excluded from export)
│       ├── state.json     ← Source hash database for incremental compile
│       ├── lock           ← Compile-time PID lock
│       └── candidates/     ← Draft pages awaiting human review
└── .env                   ← Configuration
```

Every `.md` file (except `index.md` and `log.md`) carries YAML frontmatter
with a required `type` field:

```yaml
---
type: Concept
title: Gradient Descent
description: Optimization algorithm for minimizing loss functions
tags: [machine-learning, optimization]
timestamp: 2025-06-17
resource: https://example.com/source-article
---
```

---

## Configuration

All configuration is via a `.env` file in the vault root (created by
`okf setup`):

```bash
# ── LLM Provider ──────────────────────────────
LLM_PROVIDER=ollama          # ollama | openai | custom
LLM_API_KEY=                  # required for openai/custom providers

# ── Ollama ────────────────────────────────────
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4:31b-cloud
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
OLLAMA_TIMEOUT_MS=1800000    # 30 minutes

# ── OKF Bundle ────────────────────────────────
OKF_VERSION=0.1
OKF_BUNDLE_PATH=$HOME/MyVault/04-Wiki

# ── Vault ──────────────────────────────────────
VAULT_PATH=$HOME/MyVault

# ── Content thresholds ────────────────────────
MAX_SOURCE_CHARS=1000000
MIN_SOURCE_CHARS=50
PROMPT_BUDGET_CHARS=200000
CONCEPT_MIN_BODY_CHARS=800
ENTRY_MIN_BODY_CHARS=500
CLIPPING_MIN_BODY_CHARS=500

# ── Concurrency ───────────────────────────────
COMPILE_CONCURRENCY=3

# ── Language (en, zh, or empty for auto-detect) ─
LLMWIKI_OUTPUT_LANGUAGE=

# ── Retry ─────────────────────────────────────
RETRY_COUNT=3
RETRY_BASE_MS=1000
RETRY_MULTIPLIER=4
```

### LLM Providers

`okf-pipeline` supports two provider types via `LLM_PROVIDER`:

| Provider | `LLM_PROVIDER` | Auth | API Endpoints |
|----------|---------------|------|---------------|
| **Ollama** (local) | `ollama` | None | `/api/chat`, `/api/embed` |
| **OpenAI-compatible** | `openai` | `Bearer` API key | `/v1/chat/completions`, `/v1/embeddings` |

For OpenAI-compatible providers (OpenAI, LM Studio, vLLM, OpenRouter, etc.),
set `LLM_PROVIDER=openai`, `LLM_API_KEY=<your-key>`, and point
`OLLAMA_HOST` / `LLM_HOST_URL` to your endpoint.

---

## Visualization

```bash
okf visualize ~/MyVault
okf visualize ~/MyVault --output ~/graph.html --name "My Knowledge Base"
```

The `visualize` command scans all concept `.md` files in the bundle, builds a
node/edge graph from their cross-links, and writes a self-contained
`viz.html` file. The HTML uses [Cytoscape.js](https://js.cytoscape.org/) v3.30.4
(loaded from CDN) to render an interactive force-directed graph with:

- **Color-coded node types** — Concept (teal), Entry (pink), Source (amber),
  Reference (purple), Map of Content (green).
- **Search & filter** — filter nodes by text or by type.
- **Layout switcher** — CoSE (force-directed), concentric, breadth-first,
  circle, grid.
- **Detail panel** — click any node to see its title, type, tags, description,
  full body, and backlinks ("cited by").
- **Backlink computation** — the graph computes reverse edges so the detail
  panel shows which concepts reference the selected node.

The output file is fully self-contained (except for the CDN script tag) and
can be opened directly in any modern browser via `file://`.

---

## Enrichment

```bash
okf enrich ~/MyVault --web-seed https://example.com/article
okf enrich ~/MyVault --web-seed-file seeds.txt --web-allowed-host example.com
okf enrich ~/MyVault --no-web    # dry-run, no network calls
```

The enrichment agent performs a bounded web crawl starting from seed URLs.
For each fetched page, it asks the LLM to decide one of three actions:

- **Enrich** — append a citation to an existing concept's `## Citations`
  section.
- **Mint** — create a new `Reference` page in `references/` with frontmatter
  (`type: Reference`, `resource: <source-url>`).
- **Skip** — no action needed for this page.

The crawler follows outbound links within the allowed host (when specified)
up to the `--web-max-pages` cap (default 20). An enrichment summary is
appended to `log.md` after each run.

---

## Export / Import

### Export

```bash
okf export ~/MyVault
okf export ~/MyVault --output ~/backup.tar.gz
okf export ~/MyVault --no-compress    # write .tar instead of .tar.gz
```

Packs the entire `04-Wiki/` bundle directory into a gzipped tarball. Internal
pipeline artifacts (`.llmwiki/`, `.git/`, `__pycache__/`, lock files) are
excluded so the tarball contains only portable OKF content.

### Import

```bash
okf import ~/backup.tar.gz ~/restored-vault
okf import ~/bundle.tar ~/target --no-verify    # skip lint check
```

Extracts the tarball into the target directory, initialises the `.llmwiki/`
state directory so the bundle is immediately usable, then runs the OKF linter
to verify conformance. Lint errors are reported but do not block import; use
`--no-verify` to skip the lint check entirely.

---

## Migration from obsidian-llm-wiki

```bash
okf migrate ~/MyVault
okf migrate ~/MyVault --dry-run    # preview changes without writing
```

The `migrate` command rewrites a legacy Obsidian-style vault in place to
comply with OKF v0.1. It performs the following transformations:

1. **Wikilinks → markdown links** — `[[slug]]` and `[[slug|alias]]` are
   rewritten to standard `[alias](/concepts/slug.md)` links.
2. **Inline citations → Citations section** — `^[citation text]` footnote
   syntax is extracted to a `# Citations` section at the bottom of the page
   with numbered `[^N]` references.
3. **Add `type` field** — frontmatter gains a `type` field inferred from the
   containing directory (`concepts/` → `Concept`, `entries/` → `Entry`, etc.).
4. **Remove Obsidian-specific keys** — `aliases`, `orphaned`,
   `orphaned_reason` are stripped from frontmatter.
5. **Replace legacy index/MOC** — old root `index.md` and `MOC.md` are
   deleted and replaced by per-directory `index.md` files and a bundle-root
   `index.md` generated by the OKF index generator.
6. **Generate `log.md`** — a change log with a migration entry is created.

The command returns a summary dict with counts of files migrated, wikilinks
converted, files deleted, and indexes generated.

---

## OKF Conformance

```bash
okf lint ~/MyVault
okf lint ~/MyVault --strict    # treat warnings as errors
okf lint ~/MyVault --json      # machine-readable JSON output
```

The `lint` command scans every `.md` file in the bundle and reports issues
tagged with stable rule IDs. The linter is read-only — it never modifies
files on disk.

### Lint Rules

| Rule | Severity | Description |
|------|----------|-------------|
| **OKF-001** | error | Missing YAML frontmatter block (no leading `---` fence) |
| **OKF-002** | error | Frontmatter has no `type` field, or it is empty |
| **OKF-003** | warning | `timestamp` present but not a valid ISO 8601 value |
| **OKF-004** | warning | `tags` present but not a YAML list |
| **OKF-005** | info | A markdown cross-link target does not resolve to an existing file |
| **OKF-006** | warning | `index.md` carries unexpected frontmatter (bundle-root `index.md` may carry frontmatter only when it contains an `okf_version` key) |
| **OKF-007** | warning | `log.md` date headings (`## YYYY-MM-DD`) that are not valid ISO 8601 dates |

Errors cause a non-zero exit code. Warnings are reported but do not fail
unless `--strict` is used. Use `--json` for CI/CD integration.

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally (default), or any
  **OpenAI-compatible API** endpoint (OpenAI, LM Studio, vLLM, OpenRouter,
  etc.)
- **[defuddle](https://github.com/nousresearch/defuddle)** for web content
  extraction (Stage 1)
- Python dependencies (installed automatically via `pip install -e .`):
  - `typer` — CLI framework
  - `httpx` — HTTP client for LLM API calls
  - `pyyaml` — YAML frontmatter parsing
  - `python-dotenv` — `.env` configuration loading

### Development

```bash
pip install -e ".[dev]"    # installs pytest, pytest-cov, pytest-asyncio, ruff
pytest                     # run the test suite (230+ tests)
ruff check .               # lint
```