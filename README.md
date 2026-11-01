# obsidian-llm-wiki

A knowledge compiler for Obsidian: extract accessible sources, ask an LLM for structured synthesis, then render Markdown entries, concepts, Maps of Content (MoCs), and typed links. Rendering is deterministic; extraction and model output still need review.

**Python 3.11+ · Ollama or an OpenAI-compatible API · MIT**

**Validation status:** the implementation has extensive local regression and packaging coverage, but complete live semantic/coverage acceptance remains blocked. The final bounded comparison passed both preflights and then stopped with a local-adapter HTTP 502 on its first semantic request. See [current delivery status and limits](docs/delivery-status.md); repository integration is not semantic certification.

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
olw current "$HOME/MyVault" --json  # validated generation entrypoint and Obsidian URI
olw health "$HOME/MyVault" --full
olw validate "$HOME/MyVault" --strict
olw query "$HOME/MyVault" --ask "What do these sources say about attention?"
```

`ingest` normally extracts **and builds**. Use `--skip-synthesis` to extract only. It also collects Markdown clippings from `02-Clippings/`, retaining the original clippings and inbox files in place; operational state tracks duplicates. New source files use non-overwriting installation. Local text/Markdown and supported document paths can be supplied through `--url` as well. Optional extractors:

```bash
python -m pip install -e '.[pdf,docx,youtube,liteparse]'
# .[podcast] adds local faster-whisper; .[all] installs all optional extras.
```

Remote transcript services need separate credentials. Installing extras does not guarantee access to a site, a transcript, OCR quality, or complete full text. See [extraction boundaries](docs/implementation.md#extraction-and-admission).

Legacy Word 97–2003 `.doc` extraction prefers **LibreOffice + python-docx**. LibreOffice is a separate application and must be available on the CLI's `PATH`; Python extras alone do not install it. The native route preserves text without a PDF intermediate and reports omitted images/embedded content. See [legacy Word setup, limits and verification](docs/legacy-word-extraction.md).

The [50-URL live-source audit](docs/live-source-audit.md) records real article, X, PDF/document, YouTube and podcast failures and fixes. Podcast ingestion requires an actual transcript, not a show description. Public Drive file viewers are supported only when they advertise a matching anonymous download; private files and confirmation pages remain failures. Strict provider schemas are separate from semantic/evidence acceptance.

## What is incremental—and what is not

- **Source synthesis is cache-aware.** Unchanged successful syntheses are reused. Failed replacements retain last-good knowledge rather than being treated as source deletion.
- **Unchanged builds have a validated no-op path.** Source bodies and generation files are hashed, and registry/cache/page coherence is checked. A compatible successful generation skips synthesis and publication; this is zero model calls, not zero I/O. Corrupt published files are refused rather than silently repaired.
- **Changed builds still do O(corpus) work.** A new private workspace copies prior generated state, loads and merges the live synthesis corpus, and publishes a complete immutable generation. SQLite indexes and bounded prompt/ANN candidates do **not** make the complete build dependency-local. Old generations are retained; there is no automatic pruning.
- **No million-note certification.** Synthetic registry, graph, or pipeline benchmarks measure different paths. The [scaling guide](docs/scaling.md) separates current behavior, benchmark scope, and a proposed million-note architecture.

## Common commands

| Command | Effect |
| --- | --- |
| `olw build VAULT` | Synthesize changed/incompatible sources and publish a complete immutable generation; eligible unchanged builds return early. |
| `olw current VAULT --json` | Validate and resolve the current generation's `entrypoint` and `obsidian_uri`; no homepage rewrite. Refuses absent/corrupt generations. |
| `olw build VAULT --refresh-render` | Bypass the no-op path and publish a new projection through normal review. Reuses compatible syntheses, but changed/incompatible inputs may call models and incur cost. Does not repair a corrupt committed generation in place. |
| `olw build VAULT --force` | Re-synthesize all sources, then publish a new generation. Potentially expensive; existing human files and generations remain unchanged. |
| `olw build VAULT --dry-run` | Read-only content plan from current (or legacy fallback) `state.json`: new/changed/unchanged/deleted files. No writable SQLite open, migration, provider calls, or writes; not a synthesis-compatibility/render prediction. |
| `olw ingest VAULT --url URL --refresh --preview` | Observe a fresh extraction without replacing source files; may contact extraction/transcription services. Review and save accepted text under a new input filename. Mutating `--refresh` is refused. |
| `olw ingest VAULT --resume` | Resume interrupted ingestion/retry work; inspect per-source results. |
| `olw ingest VAULT --json` | Emit NDJSON lifecycle/per-source events, not one JSON document. |
| `olw recompile VAULT article.md` | Force a complete-input retry for the named source and publish a complete new immutable generation, reusing other compatible syntheses. No follow-up `build` is required to render it. |
| `olw health VAULT --full` | Full read-only scan of the current projection (legacy fallback before publication). Compatibility batch options do not enable rotating scans; `--write` is refused. |
| `olw fix VAULT --dry-run` | Preview maintenance findings without writes or model calls. This is also the default mode. |
| `olw fix VAULT --apply` | Republish a complete projection through normal synthesis/review gates; changed/incompatible inputs may call models and incur cost. Not an in-place patcher; human-review findings are not automatically resolved. |
| `olw metrics VAULT` | Inspect pipeline metrics. |
| `olw providers check` / `olw providers models` | Inspect provider configuration / available models. |

`build`, `recompile`, and `ingest` return **1** on reported failures and **130** on cancellation; successful/no-op work returns **0**. Ingest NDJSON ends with `failed` or `partial_failure` when work failed, not `completed`. Successful work may remain durable even when other sources failed.

Run `olw COMMAND --help` for options. There is **no `olw migrate` command**. Before an upgrade, follow [backup, migration rehearsal, and recovery](docs/migration.md).

`fix --restore` is a retained compatibility option that is always refused, even for a trusted backup. Inspect/copy backup content to a separate human annotation, not over an existing human file or immutable generation. See [immutable publication](docs/immutable-publication.md) and the superseding [application-caller contract](docs/immutable-application-callers.md).

## Knowledge, language, and human ownership

- Concepts have registry `canonical_id` values; slugs remain human-readable filenames and wikilink targets. Matching, collision handling, and merging have distinct rules; a title or cosine score alone is not proof of identity. See [data model](docs/data-model.md).
- Embeddings are optional (`EMBEDDINGS_ENABLED=false` by default). Similarity retrieves candidates and can suggest topic membership; it is **not** semantic equivalence or permission to destructively merge notes. Explicit typed links and human-confirmed canonical-ID pairs in [`.llmwiki/equivalences.json`](docs/data-model.md#human-confirmed-equivalences) are separate from embedding scores and work with embeddings disabled.
- With `OUTPUT_LANGUAGE` unset, synthesis follows heuristically detected source language. Set it explicitly to request another output language; evidence quotes remain verbatim. Rendering preserves supplied titles, aliases, and prose; it does not translate or reorder them English-first. English/Chinese navigation headings follow note language. Mixed-language MoCs link counterparts without copying their foreign-language prose.
- Short supported summaries and zero concepts are valid. There is no mandatory concept count or word quota. Structural/placeholder/evidence checks reject known bad output; length-based confidence/health diagnostics are not factual verification.
- Required grounding review uses [bounded literal context v4](docs/fragment-context-v4.md) for attribution and referent resolution, never as uncited factual evidence. Policy upgrades can invalidate required-review caches and trigger new model work; historical v3 receipts and immutable generations are retained. `GROUNDING_REVIEW=off` remains unchecked, not semantic certification.
- Public builds and maintenance publish new immutable generations, retaining existing source Markdown, human policy, legacy notes, annotations, and old generations at their original paths. Edit sources, retained legacy notes, or `03-Raw-Annotations/entities/<canonical-id>.md`, not installed generation files. Manual changes to a generation are corruption and are refused, not overwritten.
- Legacy renderer backup/edit-protection machinery runs inside private workspaces; it is not permission to patch installed pages. `fix --apply` follows the same publication/review path and may call models. Full-vault snapshots remain necessary; publication is not an atomic transaction across extraction, external APIs, and all vault state.

## Vault layout

```text
MyVault/
├── .env
├── 02-Clippings/               # clipping inbox; originals retained
├── 03-Raw-Annotations/entities/<canonical-id>.md  # human notes
├── 04-Wiki/
│   ├── sources/                # editable inputs; URLs and provenance
│   ├── entries/, concepts/, mocs/, index.md  # legacy notes retained, if present
│   ├── .llmwiki/
│   │   ├── schema.yaml         # optional human synthesis policy
│   │   ├── equivalences.json   # optional human-confirmed canonical-ID pairs
│   │   └── …                   # legacy metadata retained, not current output
│   └── OLW-Publications/       # application-owned namespace
│       ├── owner.json, writer.lock
│       ├── current.json        # authoritative committed-generation pointer
│       ├── generations/<generation>/
│       │   ├── index.md        # resolve with olw current VAULT --json
│       │   ├── entries/, concepts/, mocs/, sources/, views/
│       │   └── .llmwiki/
│       │       ├── registry.db # closed SQLite snapshot (schema 7)
│       │       ├── state.json, cache/, render-page-hashes.json
│       │       └── source-inputs/, publications/, …
│       ├── .staging/           # private workspaces; not current/retrievable
│       ├── operations/         # mutable ingestion/resume state
│       └── query-state/        # mutable query sessions
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

Default pytest discovery covers `tests/new`; integration tests are separate. Use `python -m pytest tests/` for both. Before collection, the test configuration redirects HOME/XDG/temp paths into a disposable test area and removes inherited service credentials and `VAULT_PATH`; individual tests must explicitly supply their own configuration. This protects home-cache fallbacks even when a test calls `monkeypatch.undo()`. Keep optional LibreOffice and LiteParse executables on PATH when running their integration checks; inspect skips rather than treating missing executable discovery as successful coverage.

`uv.lock` pins the resolved dependency set; an ordinary `pip install -e` resolves version ranges instead. The audit did not change CI to locked installs. See the [audit issue ledger](docs/audit-issues.md) for reproduction test names and remaining risks. A listed regression is not a claim that a particular checkout passed it.

### Optional Obsidian desktop bridge

`obsidian-plugin/` invokes the installed CLI; it does not duplicate compiler logic. In that directory run `npm ci && npm run build`, then copy `manifest.json` and `main.js` to your vault's `.obsidian/plugins/obsidian-llm-wiki-bridge/`. Enable the plugin and configure the `olw` executable if it is not on Obsidian's PATH. The bridge is desktop-only.

## Documentation

- [Documentation index and status](docs/README.md)
- [Current implementation](docs/implementation.md)
- [Immutable publication and ownership](docs/immutable-publication.md)
- [Current application callers and refused compatibility options](docs/immutable-application-callers.md)
- [Identities, relationships, provenance, and persistent data](docs/data-model.md)
- [Safe migration and recovery](docs/migration.md)
- [Scaling limits and proposed architecture](docs/scaling.md)
- [Audit issue ledger](docs/audit-issues.md)
- [Verification receipts and pending final acceptance](docs/verification.md)

Historical [roadmap](docs/scalability-roadmap.md) and [plans](docs/plans/2026-07-31-five-year-scalability-v2.md) are retained as design history, **not active operating instructions or capacity guarantees**.

## License

[MIT](LICENSE).
