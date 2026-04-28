# Obsidian LLM Wiki

An LLM-assisted knowledge pipeline that turns raw URLs and markdown clippings into a structured Obsidian wiki: Source notes, Entry notes, evergreen Concepts, Maps of Content, typed graph edges, indexes, and reviewable provenance.

The design rule is simple: **Python owns facts and files; LLMs only provide bounded semantic judgment.** That keeps the system inspectable, testable, and safe enough to run as an unattended vault maintenance tool.

```text
01-Raw/*.url or 02-Clippings/*.md
        ↓
pipeline ingest
        ↓
04-Wiki/{sources, entries, concepts, mocs}
        ↓
compile / lint / graph-doctor / migrate / query
        ↓
06-Config/{wiki-index.md, tag-registry.md, edges.tsv, schema-version.json}
```

## Quick start

```bash
git clone https://github.com/0xminion/obsidian-llm-wiki.git
cd obsidian-llm-wiki
python3 -m pip install -e .

pipeline init ~/MyVault
nano ~/MyVault/Meta/Scripts/.env
```

Add a URL and run the pipeline:

```bash
echo 'https://example.com/article' > ~/MyVault/01-Raw/my-source.url
pipeline ingest ~/MyVault
```

Without installing the console script:

```bash
python3 -m pipeline.cli ingest ~/MyVault
```

## Configuration

Runtime config lives in `~/MyVault/Meta/Scripts/.env`.

```bash
# Default local/direct LLM path
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_INSIGHT_MODEL=minimax-m2.7:cloud
OLLAMA_FILENAME_MODEL=minimax-m2.7:cloud

# Cloud LLM fallback
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-sonnet-4
LLM_API_KEY=***

# Agent fallback for complex query mode
LLM_PROVIDER=hermes
AGENT_CMD=hermes

# Optional extraction APIs
TRANSCRIPT_API_KEY=***
SUPADATA_API_KEY=***
ASSEMBLYAI_API_KEY=***

PARALLEL=3
USE_QMD_MCP=true
QMD_MCP_URL=http://localhost:8181
```

Set `USE_QMD_MCP=false` to force local keyword fallback and avoid constructing a QMD client.

## Command surface

```bash
pipeline ingest ~/MyVault                  # extract → plan → create
pipeline ingest ~/MyVault --parallel 5     # tune Stage 3 workers
pipeline ingest ~/MyVault --dry-run        # preview without writes
pipeline ingest ~/MyVault --review         # stage plans for human approval
pipeline ingest ~/MyVault --resume         # continue from reviewed plans

pipeline approve ~/MyVault --json          # approve staged review queue atomically
pipeline reject ~/MyVault --json           # reject staged review queue
pipeline review-status ~/MyVault --json    # inspect pending review rows

pipeline compile ~/MyVault                 # semantic cross-links, merges, MoCs, edges, index
pipeline lint ~/MyVault                    # vault health checks
pipeline lint ~/MyVault --fix              # safe auto-fixes
pipeline validate ~/MyVault                # post-write quality gate
pipeline validate ~/MyVault --fix          # repair missing safe sections

pipeline doctor ~/MyVault --json           # first-run + config diagnostics
pipeline config-doctor ~/MyVault --json    # redacted config diagnostics alias
pipeline graph-doctor ~/MyVault --json     # unresolved links, stale edges, duplicate stems
pipeline migrate ~/MyVault --yes --json    # idempotent schema/assets migrations
pipeline release-check --json              # package/release metadata hygiene

pipeline fixture ~/MyVault --overwrite     # deterministic demo corpus
pipeline fixture ~/MyVault --adversarial --overwrite --json
pipeline telemetry ~/MyVault --json        # recent redacted pipeline events
pipeline stats ~/MyVault --json            # vault dashboard
pipeline reindex ~/MyVault                 # rebuild wiki-index.md
pipeline tags ~/MyVault                    # rebuild tag-registry.md
pipeline query ~/MyVault --ask "question" # Hermes-agent Q&A
pipeline query ~/MyVault --ask "question" --fast  # direct LLM Q&A
pipeline setup-qmd ~/MyVault               # install/configure QMD helper
pipeline setup-hooks ~/MyVault             # install vault git hooks
```

## How it works

### Stage 1: Extract

Pure Python routing and extraction. No LLM is used here.

| Source | Primary path | Fallback path |
|---|---|---|
| Web / X / PDFs | defuddle / liteparse | curl extraction → archive.org → camoufox where available |
| YouTube | transcript API | Supadata → audio download → faster-whisper |
| Podcasts | AssemblyAI | local whisper |
| Markdown clippings | direct read from `02-Clippings/` | none; Stage 1 is bypassed |

Network boundaries are hardened:

- URL parsing rejects unsupported schemes, credentials, localhost, private/reserved/link-local/multicast/unspecified targets, and weird IPv4 encodings.
- curl requests pin DNS via `--resolve` and **fail closed** if a safe public pin cannot be established.
- Secret-bearing headers are sent through curl config on stdin, not argv.
- YouTube fetches are canonicalized after hostname allowlist validation; raw user URLs are not passed through to `yt-dlp`.

### Stage 2: Plan

Planning decides what files should exist and how they should connect.

1. Content and URL deduplication via SQLite-backed store.
2. Concept matching through QMD MCP when available; keyword fallback otherwise.
3. Deterministic heuristics for obvious plans.
4. Direct LLM call only for uncertain cases.
5. Optional human review queue with `--review` / `approve` / `reject`.

### Stage 3: Create

Creation is template-first. Python builds paths, frontmatter, wikilinks, concept stubs, MoC membership, and collision-safe writes. The LLM is only asked for bounded summaries/insights.

Safety properties:

- All note stems go through `safe_note_stem()`.
- Every constructed path is containment-checked before write.
- Source and entry notes use distinct stems (`foo` and `foo-source`) so Obsidian stem resolution stays unambiguous.
- YAML is emitted with safe builders; wikilinks in YAML are quoted.
- Batch validation runs before writes are considered successful.

### Compile pass

`pipeline compile` combines semantic work with deterministic rebuilds.

Semantic operations:

- missing wikilink suggestions from embeddings + LLM validation;
- near-duplicate concept merge proposals;
- MoC synthesis from related notes;
- `semantic_status` and `semantic_degraded_reason` in `CompileResult` so empty/failed LLM responses are not falsely reported as success.

Deterministic operations:

- rebuild `wiki-index.md`;
- rebuild `edges.tsv` from entries, sources, concepts, and MoCs;
- rewrite duplicate report, including clean “0 duplicates” state;
- clear edge cache after direct edge rewrites.

Semantic pair generation is blocked by tags, title tokens, and bounded fallback windows before expensive similarity/LLM work. The common path avoids full O(N²) scans.

## Vault structure

```text
01-Raw/                 .url inbox
02-Clippings/           pre-extracted markdown inbox
03-Queries/             query files for Q&A
04-Wiki/
├── sources/            original extracted content
├── entries/            summaries and insights
├── concepts/           evergreen atomic notes
└── mocs/               maps of content
05-Outputs/             Q&A answers
06-Config/
├── wiki-index.md
├── tag-registry.md
├── edges.tsv
├── log.md
└── schema-version.json
07-WIP/                 user drafts; pipeline must not touch
08-Archive-Raw/         processed URL inbox
09-Archive-Queries/     processed questions
10-Archive-Clippings/   processed markdown clippings
Meta/
├── Scripts/            .env, logs, cache.db, telemetry
├── prompts/            runtime prompt overrides seeded from package assets
└── Templates/          runtime note templates seeded from package assets
```

## Graph and migrations

`pipeline graph-doctor` checks graph integrity without mutating vault state:

- unresolved wikilinks;
- stale `edges.tsv` rows;
- malformed edge rows;
- duplicate Obsidian stems across note collections.

`pipeline migrate` is the versioned vault migration entry point. Current schema version is `1`; it backfills vault structure/assets and writes `06-Config/schema-version.json`.

## Semantic search and QMD

QMD MCP is used for semantic concept retrieval and compile embeddings when available. Query modes are explicit so behavior is honest:

- `auto` — vector semantic search, then lexical fallback;
- `vec` — vector only;
- `lex` — BM25/keyword only.

```bash
pipeline setup-qmd ~/MyVault
qmd query "prediction markets" --json -n 5 -c concepts
```

If QMD is unavailable or disabled, the pipeline falls back to local keyword matching. If QMD returns embeddings during compile, those vectors are consumed directly instead of discarding semantic signal.

## Agentic skill installation

This repository includes a Hermes-compatible ingestion skill at:

```text
skills/obsidian-ingest.md
```

Install it into a Hermes profile by copying it as `SKILL.md` inside that profile’s skills directory:

```bash
# Replace coder with your active Hermes profile if different.
mkdir -p ~/.hermes/profiles/coder/skills/obsidian/obsidian-ingest
cp skills/obsidian-ingest.md \
  ~/.hermes/profiles/coder/skills/obsidian/obsidian-ingest/SKILL.md
```

Then restart/reload the agent session and ask it to list skills. Trigger phrases include: `obsidian`, `vault`, `clip`, `ingest`, `save to obsidian`, `wiki`, `knowledge base`, URLs, PDFs, YouTube links, and “read later”.

The skill’s job is operational, not magical: write inbox files, run `pipeline ingest`, run health gates, and report exact vault paths.

## Critical rules

1. Never touch `07-WIP/`.
2. Never trust titles, LLM filename suggestions, review DB paths, or note aliases as filesystem paths.
3. Every note write must be under the expected vault collection directory after path resolution.
4. Never overwrite existing notes without collision handling or an explicit migration path.
5. Source and entry stems must be distinct to avoid ambiguous Obsidian links.
6. Chinese content stays Chinese in generated body sections.
7. Tags are topic-specific English; avoid platform/source labels.
8. Empty LLM output during semantic work is degraded/failure, not success.
9. Secret values must not appear in argv, telemetry, docs, logs, or summaries.

## Testing and release gates

Current verified baseline: **852 passing tests**.

```bash
ruff check .
pyflakes pipeline tests
pytest -q
```

Installed-wheel smoke test:

```bash
rm -rf /tmp/obsidian-llm-wiki-wheel /tmp/obsidian-llm-wiki-venv /tmp/obsidian-llm-wiki-vault
python3 -m pip wheel . -w /tmp/obsidian-llm-wiki-wheel --no-deps
python3 -m venv /tmp/obsidian-llm-wiki-venv
/tmp/obsidian-llm-wiki-venv/bin/pip install /tmp/obsidian-llm-wiki-wheel/*.whl
/tmp/obsidian-llm-wiki-venv/bin/pipeline init /tmp/obsidian-llm-wiki-vault
/tmp/obsidian-llm-wiki-venv/bin/pipeline fixture /tmp/obsidian-llm-wiki-vault --adversarial --overwrite --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline graph-doctor /tmp/obsidian-llm-wiki-vault --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline migrate /tmp/obsidian-llm-wiki-vault --yes --json
```

Expected seeded asset counts after `pipeline init`:

- prompts: `8`
- templates: `9`

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture/ARCHITECTURE.md)
- [Product requirements](docs/product/PRD.md)
- [Changelog](docs/release/CHANGELOG.md)
- [Release process](docs/release/RELEASE.md)
- [Patch notes](docs/release/PATCH_NOTES.md)
- [Audits](docs/audits/)
- [Code reviews](docs/reviews/)
