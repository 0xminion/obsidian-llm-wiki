# obsidian-llm-wiki

A knowledge compiler for Obsidian: extract accessible sources, ask an LLM for structured synthesis, then render Markdown entries, concepts, Maps of Content (MoCs), and typed links. Rendering is deterministic; extraction and model output still need review.

**Python 3.11+ · Ollama or an OpenAI-compatible API · MIT**

## Quick start

```bash
git clone https://github.com/0xminion/obsidian-llm-wiki.git
cd obsidian-llm-wiki
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
olw setup

# Inspect first: --plan is local-only; --preview makes extraction/network calls.
olw ingest "$HOME/MyVault" --url https://arxiv.org/abs/1706.03762 --plan
olw ingest "$HOME/MyVault" --url https://arxiv.org/abs/1706.03762 --preview
olw ingest "$HOME/MyVault" --url https://arxiv.org/abs/1706.03762

# Sources already in 04-Wiki/sources/:
olw build "$HOME/MyVault"
olw health "$HOME/MyVault" --full
olw validate "$HOME/MyVault" --strict
olw query "$HOME/MyVault" --ask "What do these sources say about attention?"
```

`ingest` normally extracts **and builds**. Use `--skip-synthesis` to extract only. It also collects Markdown clippings from `02-Clippings/` and archives successfully handled clippings under `02-Clippings/processed/`. Local text/Markdown and supported document paths can be supplied through `--url` as well. Optional extractors:

```bash
python -m pip install -e '.[pdf,docx,youtube,liteparse]'
# .[podcast] adds local faster-whisper; .[all] installs all optional extras.
```

Remote transcript services need separate credentials. Installing extras does not guarantee access to a site, a transcript, OCR quality, or complete full text. See [extraction boundaries](docs/implementation.md#extraction-and-admission).

The [50-URL live-source audit](docs/live-source-audit.md) records real article, X, PDF/document, YouTube and podcast failures and fixes. Podcast ingestion requires an actual transcript, not a show description. Public Drive file viewers are supported only when they advertise a matching anonymous download; private files and confirmation pages remain failures. Strict provider schemas are separate from semantic/evidence acceptance.

## What is incremental—and what is not

- **Source synthesis is cache-aware.** Unchanged successful syntheses are reused. Failed replacements retain last-good knowledge rather than being treated as source deletion.
- **Unchanged builds have a checkpoint fast path.** Source discovery still enumerates/stats the source directory. A matching input/configuration checkpoint skips corpus synthesis loading, merge, and render; it does not inspect every generated note for manual damage.
- **Changed builds still do O(corpus) work.** SQLite indexes, bounded prompt/ANN candidates, and selective file writes do **not** make the complete build dependency-local. The pipeline loads and merges the live synthesis corpus and constructs render manifests.
- **No million-note certification.** Synthetic registry, graph, or pipeline benchmarks measure different paths. The [scaling guide](docs/scaling.md) separates current behavior, benchmark scope, and a proposed million-note architecture.

## Common commands

| Command | Effect |
| --- | --- |
| `olw build VAULT` | Synthesize changed/incompatible sources and reconcile generated output; eligible unchanged builds return early. |
| `olw build VAULT --refresh-render` | Bypass the no-op checkpoint and reconcile pages. Reuses compatible syntheses; **does not force** LLM calls. Dirty/incompatible sources may still need synthesis. |
| `olw build VAULT --force` | Read and re-synthesize all sources, then render. Potentially expensive; not a way to bypass user-edit protection. |
| `olw build VAULT --dry-run` | Read-only content plan from `state.json`: new/changed/unchanged/deleted files. No SQLite open, migration, provider calls, or writes; not a compatibility/render prediction. |
| `olw ingest VAULT --url URL --refresh` | Re-extract a known URL, backing up changed body or title/provenance metadata at the same source filename. |
| `olw ingest VAULT --resume` | Resume interrupted ingestion/retry work; inspect per-source results. |
| `olw ingest VAULT --json` | Emit NDJSON lifecycle/per-source events, not one JSON document. |
| `olw recompile VAULT article.md` | Retry one complete source and update cached synthesis; run `build` afterward to render. |
| `olw health VAULT --full` | Full health scan; omit `--full` for a rotating bounded batch. |
| `olw fix VAULT --dry-run` | Preview maintenance repairs; `--apply` explicitly applies deterministic non-reviewed fixes. |
| `olw metrics VAULT` | Inspect pipeline metrics. |
| `olw providers check` / `olw providers models` | Inspect provider configuration / available models. |

`build`, `recompile`, and `ingest` return **1** on reported failures and **130** on cancellation; successful/no-op work returns **0**. Ingest NDJSON ends with `failed` or `partial_failure` when work failed, not `completed`. Successful work may remain durable even when other sources failed.

Run `olw COMMAND --help` for options. There is **no `olw migrate` command**. Before an upgrade, follow [backup, migration rehearsal, and recovery](docs/migration.md).

## Knowledge, language, and human ownership

- Concepts have registry `canonical_id` values; slugs remain human-readable filenames and wikilink targets. Matching, collision handling, and merging have distinct rules; a title or cosine score alone is not proof of identity. See [data model](docs/data-model.md).
- Embeddings are optional (`EMBEDDINGS_ENABLED=false` by default). Similarity retrieves candidates and can suggest topic membership; it is **not** semantic equivalence or permission to destructively merge notes. Explicit typed links and human-confirmed canonical-ID pairs in [`.llmwiki/equivalences.json`](docs/data-model.md#human-confirmed-equivalences) are separate from embedding scores and work with embeddings disabled.
- With `OUTPUT_LANGUAGE` unset, synthesis follows heuristically detected source language. Set it explicitly to request another output language; evidence quotes remain verbatim. Rendering preserves supplied titles, aliases, and prose; it does not translate or reorder them English-first. English/Chinese navigation headings follow note language. Mixed-language MoCs link counterparts without copying their foreign-language prose.
- Short supported summaries and zero concepts are valid. There is no mandatory concept count or word quota. Structural/placeholder/evidence checks reject known bad output; length-based confidence/health diagnostics are not factual verification.
- The renderer preserves reviewed pages, previously generated pages whose recorded content changed, and unknown pre-existing notes. This protects both body and frontmatter, but may leave stale generated information requiring review. Existing source Markdown is input, not disposable output.
- Rewrites of recognized unchanged generated notes are backed up. Backups and render rollback are not a substitute for a full vault snapshot, nor a cross-filesystem/SQLite atomic transaction. Maintenance `fix --apply` has its own narrower rules; preview it separately.

## Vault layout

```text
MyVault/
├── .env
├── schema.yaml                 # optional synthesis policy
├── 02-Clippings/               # clipping inbox and processed/ archive
├── 03-Raw-Annotations/         # manual notes
├── 04-Wiki/
│   ├── sources/                # source Markdown; original URLs and provenance
│   ├── entries/                # per-source analysis
│   ├── concepts/               # concept notes and typed relations
│   ├── mocs/                   # Maps of Content
│   ├── views/                  # generated views
│   ├── index.md
│   └── .llmwiki/
│       ├── registry.db         # SQLite WAL runtime registry (schema 6)
│       ├── equivalences.json   # optional human-confirmed canonical-ID pairs
│       ├── state.json          # compatibility/source compilation state
│       ├── cache/              # portable synthesis JSON and overlays
│       ├── render-page-hashes.json
│       ├── backups/
│       └── …                   # locks, metrics, graphs, transcripts, reports
└── 05-Queries/                 # saved query answers
```

Current source loading is flat (`sources/*.md`), not recursive stable-ID file partitioning. Do not relocate files into proposed shard directories without implementing a migration.

## Configuration

Start with [`.env.example`](.env.example). CLI commands resolve the selected vault's `.env`; explicit process environment normally overrides file values. An explicitly supplied `load_config(env_file=...)` has different file-wins precedence. Never commit API keys.

```dotenv
LLM_PROVIDER=ollama
LLM_HOST=http://localhost:11434
LLM_MODEL=gemma3:27b
SYNTHESIS_MODE=single
OUTPUT_LANGUAGE=
COMPILE_CONCURRENCY=3
EMBEDDINGS_ENABLED=false
```

For OpenAI-compatible endpoints use `LLM_PROVIDER=openai`, the appropriate `LLM_HOST`, `LLM_MODEL`, and `LLM_API_KEY`. Task overrides are `LLM_INGEST_MODEL`, `LLM_EXPAND_MODEL`, `LLM_MAINTENANCE_MODEL`, and `LLM_QUERY_MODEL`. Set `EMBEDDING_MODEL`/`EMBEDDING_HOST` separately if enabling embeddings; choose a model tested for your languages. [Implementation and resource settings](docs/implementation.md) explain limits and fallback behavior.

## Development

```bash
# Reproducible development environment from the checked-in uv.lock:
uv sync --locked --extra dev --extra pdf --extra docx --extra liteparse --extra youtube
source .venv/bin/activate
python -m pytest
python -m ruff check obsidian_llm_wiki tests
# Optional external parser contract; requires .[liteparse]:
python -m pytest tests/integration
```

Default pytest discovery covers `tests/new`; integration tests are separate. `uv.lock` pins the resolved dependency set; an ordinary `pip install -e` resolves version ranges instead. The audit did not change CI to locked installs. See the [audit issue ledger](docs/audit-issues.md) for reproduction test names and remaining risks. A listed regression is not a claim that a particular checkout passed it.

### Optional Obsidian desktop bridge

`obsidian-plugin/` invokes the installed CLI; it does not duplicate compiler logic. In that directory run `npm ci && npm run build`, then copy `manifest.json` and `main.js` to your vault's `.obsidian/plugins/obsidian-llm-wiki-bridge/`. Enable the plugin and configure the `olw` executable if it is not on Obsidian's PATH. The bridge is desktop-only.

## Documentation

- [Documentation index and status](docs/README.md)
- [Current implementation](docs/implementation.md)
- [Identities, relationships, provenance, and persistent data](docs/data-model.md)
- [Safe migration and recovery](docs/migration.md)
- [Scaling limits and proposed architecture](docs/scaling.md)
- [Audit issue ledger](docs/audit-issues.md)
- [Verification receipts and pending final acceptance](docs/verification.md)

Historical [roadmap](docs/scalability-roadmap.md) and [plans](docs/plans/2026-07-31-five-year-scalability-v2.md) are retained as design history, **not active operating instructions or capacity guarantees**.

## License

[MIT](LICENSE).
