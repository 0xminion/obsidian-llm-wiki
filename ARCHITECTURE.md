# System Architecture Document — obsidian-llm-wiki v0.1.0

## Table of Contents
1. [System Overview](#1-system-overview)
2. [User Journey](#2-user-journey)
3. [Architecture](#3-architecture)
4. [Data Flow Diagram](#4-data-flow-diagram)
5. [Stage 1: Extraction](#5-stage-1-extraction)
6. [Stage 2: Planning](#6-stage-2-planning)
7. [Stage 3: Creation](#7-stage-3-creation)
8. [Content Store (SQLite)](#8-content-store-sqlite)
9. [Dead Letter Queue](#9-dead-letter-queue)
10. [Review/Approval Workflow](#10-reviewapproval-workflow)
11. [CLI Commands](#11-cli-commands)
12. [Data Models](#12-data-models)
13. [Error Handling & Recovery](#13-error-handling--recovery)
14. [Tools & Dependencies](#14-tools--dependencies)
15. [Configuration](#15-configuration)
16. [Shell Scripts](#16-shell-scripts)

---

## 1. System Overview

obsidian-llm-wiki is a Python pipeline that transforms raw web URLs into a structured, interconnected Obsidian wiki. It implements Andrej Karpathy's LLM-Wiki pattern:

```
URL → Source → Entry → Concept → MoC (Map of Content)
```

The system ingests content from YouTube, podcasts, tweets, web articles, and arxiv papers. It extracts transcripts/text, plans wiki entries, creates structured markdown files, and maintains cross-references between them.

**Design principles:**
- No external cron jobs — automation baked into the pipeline
- Deterministic first, LLM fallback — heuristics handle 80% of planning
- Template first, agent fallback — structure is template-generated, only insights need intelligence
- Dead letter queue — failures are tracked, not forgotten
- Review before write — optional approval workflow before vault writes

---

## 2. User Journey

### Simple case: ingest 3 URLs

```
User: drops 3 .url files into ~/MyVault/01-Raw/
       └── article.url       (URL=https://example.com/post)
       └── video.url          (URL=https://youtube.com/watch?v=abc123)
       └── podcast.url        (URL=https://podcasts.apple.com/.../id123)

User: runs `pipeline ingest ~/MyVault`

Pipeline:
  Stage 1 (10s):  Extracts content from all 3 URLs in parallel
  Stage 2 (5s):   Plans entry structure deterministically (no LLM)
  Stage 3 (30s):  Creates vault files (template mode) or calls agent

Result:
  ~/MyVault/04-Wiki/sources/example-post.md       ← Source note
  ~/MyVault/04-Wiki/entries/example-post.md        ← Entry note
  ~/MyVault/04-Wiki/concepts/Example Topic.md      ← Concept (if new)
  ~/MyVault/04-Wiki/mocs/Technology.md              ← MoC updated
  ~/MyVault/01-Raw/                                 ← .url files archived
```

### Review case: approve before writing

```
User: `pipeline ingest ~/MyVault --review`
Pipeline: extracts, plans, stages files in pending_reviews table

User: `pipeline approve --dry-run`
Pipeline: shows what would be written

User: `pipeline approve`
Pipeline: writes files, reindexes, archives inbox
```

### Failure case: Cloudflare-blocked URL

```
User: drops blocked-site.url (Cloudflare challenge page)

Pipeline Stage 1:
  Attempt 1: defuddle → gets Cloudflare page → rejected
  Attempt 2: curl+liteparse → same → rejected
  Attempt 3: archive.org fallback → succeeds or fails

If all fail:
  → Recorded to DLQ with reason="cloudflare"
  → User runs `pipeline dlq` to see failed URLs
  → User runs `pipeline dlq --clear --reason=cloudflare` to reset
```

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (cli.py)                         │
│  ingest | approve | reject | dlq | lint | reindex | stats  │
└──────────┬──────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐     ┌──────────┐     ┌──────────────┐
    │  Stage 1    │     │ Stage 2  │     │   Stage 3    │
    │  Extract    │────▶│  Plan    │────▶│   Create     │
    │ (extract.py)│     │(plan.py) │     │ (create.py)  │
    └──────┬──────┘     └────┬─────┘     └──────┬───────┘
           │                 │                   │
    ┌──────▼──────┐     ┌────▼─────┐     ┌──────▼───────┐
    │  Extractors │     │   QMD    │     │   Review     │
    │  youtube    │     │ semantic │     │  (review.py) │
    │  podcast    │     │  search  │     │  approve ◄── │
    │  web        │     └──────────┘     │  reject  ◄── │
    └──────┬──────┘                      └──────┬───────┘
           │                                    │
    ┌──────▼────────────────────────────────────▼───────┐
    │              Content Store (store.db)              │
    │  urls | content | dead_letter_queue | reviews      │
    └───────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐
    │    Vault    │
    │  04-Wiki/   │
    │  sources/   │
    │  entries/   │
    │  concepts/  │
    │  mocs/      │
    └─────────────┘
```

### Module map

```
pipeline/
├── cli.py          # CLI entry point, command routing
├── config.py       # Config dataclass, .env loading, path resolution
├── models.py       # Data models: ExtractedSource, Plan, Manifest, Edge
├── store.py        # SQLite content store (dedup, DLQ, reviews)
├── extract.py      # Stage 1 router: detect type → delegate to extractor
├── extractors/
│   ├── _shared.py  # Shared: curl, run, title extraction, validation
│   ├── youtube.py  # YouTube: TranscriptAPI → Supadata → whisper
│   ├── podcast.py  # Podcast: iTunes API → RSS → AssemblyAI/whisper
│   └── web.py      # Web: defuddle → curl → liteparse → archive.org
├── plan.py         # Stage 2: deterministic planning + agent fallback
├── create/
│   ├── __init__.py # Re-exports: create_all, create_file_templates
│   ├── agent.py    # Agent orchestration, batch creation
│   ├── orchestrator.py # Stage 3 entry point, batch splitting
│   ├── prompts.py  # Prompt loading, batch prompt construction
│   ├── templates.py # Template-based file creation
│   └── validate.py # Output validation and auto-repair
├── qmd.py          # Shared qmd semantic search (parallel queries)
├── review.py       # Review workflow: stage → approve → write
├── vault.py        # Vault operations: write files, edges, reindex, archive
├── compile.py      # Compile pass: concept merge, MoC rebuild
├── lint.py         # Vault health checks (12+ checks)
├── stats.py        # Dashboard generation
└── utils.py        # Shared utilities: escape_yaml, extract_body, strip_qmd_noise
```

---

## 4. Data Flow Diagram

```
                    ┌─────────────┐
                    │ 01-Raw/*.url│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  CLI ingest │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │     Stage 1: Extract    │
              │                         │
              │  for each URL:          │
              │    detect_source_type() │
              │    ┌────────────────┐   │
              │    │ URL dedup?     │───┼──▶ ContentStore.is_url_extracted()
              │    └───────┬────────┘   │
              │            │ not dup    │
              │    ┌───────▼────────┐   │
              │    │ extract_url()  │   │
              │    │  retry loop    │   │
              │    │  quality check │   │
              │    └───────┬────────┘   │
              │            │            │
              │    ┌───────▼────────┐   │
              │    │ Content dedup? │───┼──▶ ContentStore.get_content_duplicate()
              │    └───────┬────────┘   │
              │            │ not dup    │
              │    ┌───────▼────────┐   │
              │    │ Save .json     │   │
              │    │ Register store │   │
              │    └───────┬────────┘   │
              │            │            │
              │    On failure:          │
              │    ┌───────▼────────┐   │
              │    │ DLQ record     │───┼──▶ ContentStore.dlq_add()
              │    └────────────────┘   │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │ manifest.json│
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │     Stage 2: Plan       │
              │                         │
              │  Step 0: dedup_check()  │──▶ Jaccard similarity vs vault
              │  Step 1: qmd search     │──▶ semantic concept matching
              │  Step 2: deterministic  │──▶ heuristics (80% of sources)
              │  Step 3: agent fallback │──▶ hermes (uncertain 20%)
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  plans.json │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐     ┌──────▼──────┐   ┌─────▼─────┐
    │ --review│     │  --template  │   │ (default) │
    │ stage   │     │  template    │   │  agent    │
    └────┬────┘     └──────┬──────┘   └─────┬─────┘
         │                 │                 │
    ┌────▼────┐     ┌──────▼──────┐   ┌─────▼─────┐
    │pending_ │     │ Source.md   │   │ Agent     │
    │reviews  │     │ Entry.md    │   │ creates   │
    │table    │     │ Concept.md  │   │ all files │
    └────┬────┘     │ MoC update  │   └─────┬─────┘
         │          └──────┬──────┘         │
    ┌────▼────┐            │                │
    │pipeline │     ┌──────▼──────┐         │
    │approve  │     │  Vault      │◄────────┘
    └────┬────┘     │  04-Wiki/   │
         │          └─────────────┘
    ┌────▼────┐
    │ Write   │
    │ files   │
    │ reindex │
    │ archive │
    └─────────┘
```

---

## 5. Stage 1: Extraction

### Entry point: `extract_all(urls, cfg, parallel)`

```
extract_all()
  ├── ContentStore.open()          # SQLite connection
  ├── ThreadPoolExecutor(parallel) # Parallel extraction
  │   └── for each url:
  │       └── extract_url(url, cfg, store)
  │           ├── store.is_url_extracted(url)  → skip if dup
  │           ├── detect_source_type(url)      → YOUTUBE|PODCAST|TWITTER|WEB
  │           ├── retry loop (max_retries):
  │           │   ├── call extractor
  │           │   ├── validate_extraction()    → check quality
  │           │   └── if bad: sleep(2^attempt), retry
  │           ├── store.get_content_duplicate() → skip if content dup
  │           ├── store.register_url() + register_content()
  │           ├── source.save()                → write {hash}.json
  │           └── on failure:
  │               ├── store.dlq_add()          → record to DLQ
  │               └── store.register_url(status="failed")
  └── manifest.save()                          → write manifest.json
```

### Extractor chain per source type

**YouTube** (`youtube.py`):
```
extract_youtube(url, cfg)
  ├── oEmbed metadata (title, author)
  ├── _try_youtube_transcript():
  │   ├── 1. TranscriptAPI (full URL, Bearer auth)
  │   ├── 2. Supadata (POST JSON, x-api-key)
  │   └── 3. yt-dlp + faster-whisper (last resort)
  └── return ExtractedSource
```

**Podcast** (`podcast.py`):
```
extract_podcast(url, cfg)
  ├── Extract IDs from Apple Podcasts URL (id + ?i=)
  ├── _find_feed_url():
  │   ├── 1. iTunes lookup (entity=podcast)
  │   ├── 2. iTunes lookup (entity=podcastEpisode)
  │   └── 3. iTunes search by name
  ├── _parse_rss_episode():
  │   ├── Match by episode ID in GUID/link
  │   ├── Match by title slug (60% keyword overlap)
  │   └── Fallback to latest episode
  ├── _transcribe_podcast_audio():
  │   ├── yt-dlp download
  │   ├── AssemblyAI upload + poll
  │   └── Fallback: faster-whisper
  └── return ExtractedSource
```

**Web** (`web.py`):
```
extract_web(url, cfg, source_type)
  ├── retry loop (max_retries):
  │   └── _extract_web_content():
  │       ├── arxiv? → arxiv HTML → alphaxiv.org
  │       ├── 1. defuddle parse --markdown
  │       ├── 2. curl + liteparse (rotate UA on retry)
  │       └── 3. defuddle parse --json
  ├── fallback: archive.org Wayback Machine
  └── return ExtractedSource
```

### Quality validation (`validate_extraction`)

```
rejects:
  - Empty content
  - Length < 5 chars
  - Cloudflare challenge page (detected by patterns)
  - Error indicators ("This site can't be reached", etc.)
```

---

## 6. Stage 2: Planning

### Entry point: `plan_sources(manifest, cfg)`

```
plan_sources()
  ├── Step 0: dedup_check()
  │   └── Jaccard 3-gram similarity vs existing vault sources
  │       threshold: 0.85 → duplicate
  ├── Step 1: concept_search()
  │   └── qmd semantic search (Qwen3-Embedding-0.6B-Q8)
  │       per source: title + content[:500] → top 5 concept matches
  ├── Step 2: generate_plans_deterministic()
  │   └── for each source:
  │       ├── detect_language(content)     → EN|ZH
  │       ├── select_template(type, content) → standard|technical|chinese
  │       ├── extract_tags(content)        → topic keywords
  │       ├── generate_plan_heuristic()    → title, lang, template, tags, concepts
  │       └── confidence check → uncertain? → agent queue
  ├── Step 3: agent fallback (only for uncertain)
  │   └── generate_plans() via hermes chat
  └── Plans.save() → plans.json
```

### Deterministic planning logic

```
generate_plan_heuristic(entry, concept_matches):
  title:     extract_title(content) or entry.title or entry.url
  language:  CJK chars / total chars > 0.2 → ZH, else EN
  template:  PODCAST/YOUTUBE → standard
             technical markers → technical
             else → standard
  tags:      topic keyword matching (crypto, economics, AI, etc.)
  concepts:  qmd matches > 0.5 → update existing
             qmd matches < 0.3 or none → suggest new from title
```

### Confidence check (deterministic vs agent)

```
needs_agent = True if:
  - No title found (title == URL)
  - Content < 50 chars
  - No concept matches AND no concept_new suggestion
```

---

## 7. Stage 3: Creation

### Mode 1: Agent-based (`create_all`) — default

```
create_all(plans, cfg, parallel)
  ├── concept_convergence() → qmd semantic search per plan
  ├── split_batches(plans, parallel)
  ├── for each batch:
  │   └── create_batch():
  │       ├── build_batch_prompt() → agent prompt with all data
  │       ├── _run_agent() → hermes chat -q "prompt" -Q
  │       ├── agent writes Source, Entry, Concept, MoC files
  │       ├── validate file existence (title_to_filename)
  │       └── retry on failure (max_retries)
  ├── validate_output() → check frontmatter, sections, stubs
  ├── _repair_violations() → auto-fix missing sections
  ├── reindex → wiki-index.md
  ├── log → log.md
  └── archive_inbox → move .url to archive
```

### Mode 2: Template-based (`create_file_templates`) — `--template`

```
create_file_templates(plans, cfg, use_agent_insights)
  └── for each plan:
      ├── generate_source_content() → deterministic YAML + content
      ├── write source directly to sources_dir
      ├── generate_entry_insights() → agent for Summary + Core insights only
      │   └── minimal prompt: "Write EXACTLY these two sections"
      ├── generate_entry_content() → template sections + agent insights
      ├── write_entry(cfg, plan, content)
      ├── for each new concept:
      │   └── write_concept(cfg, name, content, sources)
      └── for each MoC target:
          └── update_moc(cfg, moc_name, entry_name, description)
```

### Mode 3: Review (`--review`)

```
stage_for_review(plans, cfg)
  ├── generate_source_content() + store.review_add()
  ├── generate_entry_insights() + generate_entry_content() + store.review_add()
  └── for each concept: generate + store.review_add()

pipeline approve:
  ├── for each pending review:
  │   └── write file_path with file_content
  ├── reindex
  └── archive_inbox

pipeline reject:
  └── clear pending_reviews table
```

---

## 8. Content Store (SQLite)

### Database: `.pipeline/store.db`

```
TABLE urls:
  url_hash      TEXT PK     -- MD5(normalized_url)[:12]
  url           TEXT        -- original URL
  canonical_url TEXT        -- normalized (lowercase, no tracking params)
  source_type   TEXT        -- youtube|podcast|twitter|web
  extracted_at  REAL        -- unix timestamp
  status        TEXT        -- ok|failed
  content_hash  TEXT FK     -- → content.content_hash

TABLE content:
  content_hash  TEXT PK     -- MD5(normalized_content)[:16]
  title         TEXT
  source_type   TEXT
  word_count    INTEGER
  created_at    REAL
  vault_filename TEXT       -- filename in vault (for dedup lookup)

TABLE dead_letter_queue:
  id             INTEGER PK
  url            TEXT
  reason         TEXT        -- cloudflare|paywall|timeout|empty_content|network|unknown
  attempts       INTEGER     -- incremented on retry
  last_error     TEXT
  first_failed_at REAL
  last_failed_at  REAL
  metadata       TEXT JSON   -- {source_type, attempts}
  status         TEXT        -- pending|resolved

TABLE pending_reviews:
  id             INTEGER PK
  plan_hash      TEXT
  plan_data      TEXT JSON   -- full Plan dict
  file_type      TEXT        -- source|entry|concept
  file_path      TEXT        -- target path in vault
  file_content   TEXT        -- generated markdown
  created_at     REAL
  status         TEXT        -- pending|approved|rejected
```

### URL normalization

```
https://Example.COM/page?utm_source=twitter&id=1#section
→ https://example.com/page?id=1
(lowercase, strip tracking params, strip fragment, strip trailing /)
```

### Content hashing

```
"  This IS   Test Content  "
→ normalize: collapse whitespace, lowercase, first 2000 chars
→ MD5[:16]
```

---

## 9. Dead Letter Queue

### When records are created

```
extract_url() exhausts all retries (max_retries attempts):
  → store.dlq_add(url, reason, error, metadata)
  → reason auto-classified from error message:
      "cloudflare" in error → cloudflare
      "paywall" in error    → paywall
      "timeout" in error    → timeout
      "empty" in error      → empty_content
      "connection" in error → network
      else                  → unknown
```

### DLQ operations

```
dlq_add(url, reason, error):
  if URL already in DLQ (pending):
    → increment attempts, update last_error
  else:
    → INSERT new record

dlq_get_pending(limit) → list of failed items
dlq_resolve(id) → mark as resolved
dlq_clear(reason=None) → delete pending items (optionally filtered)
```

### CLI usage

```
pipeline dlq                    → show pending failures
pipeline dlq --clear            → clear all pending
pipeline dlq --reason=cloudflare → clear only cloudflare failures
```

---

## 10. Review/Approval Workflow

```
                    ┌──────────────────────┐
                    │ pipeline ingest      │
                    │ --review             │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Extract + Plan       │
                    │ (Stages 1+2)         │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ stage_for_review()   │
                    │                      │
                    │ For each plan:       │
                    │  generate content    │
                    │  store.review_add()  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐  ┌─────▼──────┐  ┌──────▼───────┐
     │pipeline       │  │pipeline    │  │pipeline      │
     │approve        │  │approve     │  │reject        │
     │               │  │--dry-run   │  │              │
     └────────┬──────┘  └─────┬──────┘  └──────┬───────┘
              │               │                │
     ┌────────▼──────┐  ┌─────▼──────┐  ┌──────▼───────┐
     │Write files    │  │Show what   │  │Clear pending │
     │Mark approved  │  │would be    │  │reviews       │
     │reindex        │  │written     │  │              │
     │archive inbox  │  │            │  │              │
     └───────────────┘  └────────────┘  └──────────────┘
```

---

## 11. CLI Commands

| Command | Description | Key flags |
|---------|-------------|-----------|
| `ingest` | Full pipeline: extract → plan → create | `--parallel`, `--dry-run`, `--review`, `--resume`, `--template`, `--verbose` |
| `approve` | Write pending review files to vault | `--dry-run` |
| `reject` | Discard all pending reviews | |
| `dlq` | Show/manage dead letter queue | `--clear`, `--reason` |
| `store-stats` | Show content store statistics | |
| `lint` | Vault health checks (12 checks) | `--fix` |
| `validate` | Validate pipeline output | `--fix` |
| `reindex` | Rebuild wiki-index.md | |
| `stats` | Generate vault dashboard | |
| `compile` | Concept convergence + MoC rebuild | |

### Ingest flags

```
--parallel N     Parallel workers per stage (default: 3)
--dry-run        Preview without writing
--review         Stage for approval (skip Stage 3)
--resume         Continue from saved plans (skip Stages 1+2)
--template       Use deterministic template creation + insight agent
--verbose        Debug logging
```

---

## 12. Data Models

### ExtractedSource (Stage 1 output)

```python
@dataclass
class ExtractedSource:
    url: str              # Original URL
    title: str            # Extracted or derived title
    content: str          # Transcript, text, or description
    type: SourceType      # youtube|podcast|twitter|web|pdf|unknown
    author: str = ""      # Channel name, author, podcast name
    source_file: str = "" # Source .url filename

    @property
    def hash -> str       # MD5(url)[:12]
    @property
    def content_hash -> str # MD5(normalized_content)[:16]
```

### Plan (Stage 2 output)

```python
@dataclass
class Plan:
    hash: str              # → ExtractedSource.hash
    title: str             # Content title for filename
    language: Language      # en|zh
    template: Template      # standard|technical|chinese|comparison|procedural
    tags: list[str]         # Topic-specific English tags
    concept_updates: list[str]  # Existing concepts to link
    concept_new: list[str]      # New concepts to create
    moc_targets: list[str]      # MoCs to update
```

### SourceType enum

```python
class SourceType(str, Enum):
    WEB = "web"
    YOUTUBE = "youtube"
    PODCAST = "podcast"
    PDF = "pdf"
    TWITTER = "twitter"
    UNKNOWN = "unknown"
```

### File structure

```
04-Wiki/
├── sources/           # Raw extracted content
│   └── {title}.md
├── entries/           # Processed wiki entries
│   └── {title}.md
├── concepts/          # Evergreen concept notes
│   └── {concept}.md
└── mocs/              # Maps of Content (topic indexes)
    └── {topic}.md
```

---

## 13. Error Handling & Recovery

### Extraction failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Cloudflare challenge | `_is_challenge_page()` patterns | Retry with different UA → archive.org fallback → DLQ |
| Paywall | "Subscribers Only" in content | Metadata-only source → DLQ |
| Network timeout | `subprocess.TimeoutExpired` | Exponential backoff retry → DLQ |
| Empty content | `validate_extraction()` | Retry → DLQ |
| Podcast ID mismatch | iTunes lookup returns 0 | Search by name → RSS fallback |
| YouTube 403 | TranscriptAPI/Supadata fail | yt-dlp + whisper fallback |

### Planning failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Agent timeout | `subprocess.TimeoutExpired` | Retry with backoff |
| Agent returns invalid JSON | `_parse_agent_output()` | Object-by-object parsing (partial recovery) |
| No plans generated | `len(plans) == 0` | Log error, return empty |
| qmd timeout | `subprocess.TimeoutExpired` | Return empty matches (non-blocking) |

### Creation failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Agent empty output | `not output` | Retry (max_retries) |
| Files not created | `file.exists()` check | Retry with backoff |
| Validation violations | `validate_output()` | `_repair_violations()` auto-fix |
| YAML frontmatter errors | Regex parsing | Auto-repair (add missing fields) |

### Pipeline-level recovery

```
--resume flag:
  Loads saved manifest.json + plans.json
  Skips Stages 1+2, runs only Stage 3
  Use case: review plans, then create

Pipeline lock:
  Directory-based lock at 06-Config/.pipeline.lock
  PID check for stale locks
  Time-based stale detection (30 min)

Content store:
  SQLite WAL mode for crash recovery
  Atomic commits per operation
```

---

## 14. Tools & Dependencies

### Python pipeline

| Tool | Purpose | Source |
|------|---------|--------|
| `hermes` | LLM agent for planning + creation | External CLI |
| `qmd` | Semantic search (Qwen3-Embedding) | External CLI |
| `defuddle` | Web content extraction | External CLI |
| `liteparse` | HTML to text conversion | External CLI |
| `yt-dlp` | YouTube/podcast audio download | External CLI |
| `faster-whisper` | Local speech recognition | Python module |
| `typer` | CLI framework | pip dependency |
| `sqlite3` | Content store | stdlib |

### APIs

| API | Purpose | Auth |
|-----|---------|------|
| YouTube oEmbed | Video metadata | None |
| TranscriptAPI | YouTube transcripts | Bearer token |
| Supadata | YouTube transcripts (fallback) | x-api-key |
| iTunes Lookup | Podcast metadata + RSS | None |
| iTunes Search | Podcast search by name | None |
| AssemblyAI | Audio transcription | Bearer token |
| archive.org | Web page archival | None |

### Shell scripts (supplementary)

All lint, compile, validate, stats, and review functionality has been migrated to Python (see CLI Commands above). Remaining shell scripts:

| Script | Purpose |
|--------|---------|
| `query-vault.sh` | Q&A with vault via qmd semantic search |
| `update-tag-registry.sh` | Tag registry rebuild |

---

## 15. Configuration

### Config class (`config.py`)

```python
@dataclass
class Config:
    vault_path: Path          # ~/MyVault
    extract_dir: Path | None  # auto: /tmp/obsidian-extracted-{hash}
    extract_timeout: int = 45 # seconds per extraction
    plan_timeout: int = 300   # seconds for planning agent
    create_timeout: int = 900 # seconds for creation agent
    max_retries: int = 3      # retry attempts
    parallel: int = 3         # parallel workers
    agent_cmd: str = "hermes" # agent CLI command
    qmd_cmd: str = "qmd"      # semantic search CLI
    qmd_collection: str = "..." # qmd collection path
    transcript_api_key: str   # from .env
    supadata_api_key: str     # from .env
    assemblyai_api_key: str   # from .env
```

### Environment (.env)

```
TRANSCRIPT_API_KEY=sk_...
SUPADATA_API_KEY=sd_...
ASSEMBLYAI_API_KEY=...
```

### Vault structure (expected)

```
~/MyVault/
├── 01-Raw/              # Inbox: .url files
├── 04-Wiki/
│   ├── sources/         # Source notes
│   ├── entries/         # Entry notes
│   ├── concepts/        # Concept notes
│   └── mocs/            # Maps of Content
├── 06-Config/           # Pipeline config + lock
├── 07-WIP/              # Working files, logs
└── Meta/
    ├── prompts/         # Agent prompt templates
    ├── Templates/       # Note templates
    └── Scripts/         # Shell scripts
```

---

## 16. Shell Scripts

All core pipeline functionality (compile, lint, validate, stats, review) has been fully migrated to Python. See the module map and CLI Commands section above.

### Remaining scripts

**`query-vault.sh`** — Q&A interface using qmd semantic search. Queries the vault's vector index for interactive exploration.

**`update-tag-registry.sh`** — Rebuilds `tag-registry.md` from all vault files by extracting tags from YAML frontmatter.

---

*Document generated from codebase at commit `8199d4b`.*
