# Current implementation

Active reference for the audited working tree. The implementation—not older plans or function docstrings—defines behavior. Main paths: [`core/pipeline.py`](../obsidian_llm_wiki/core/pipeline.py), [`cli/ingest.py`](../obsidian_llm_wiki/cli/ingest.py), [`core/registry.py`](../obsidian_llm_wiki/core/registry.py), and [`render/obsidian.py`](../obsidian_llm_wiki/render/obsidian.py).

## Pipeline and commit boundaries

```text
URL / local document / clipping
  → extraction + admission + provenance
  → sources/*.md
  → source inventory and file-signature comparison
  → compatible cached synthesis OR bounded candidate retrieval + LLM
  → structural/content/evidence acceptance
  → full live-corpus merge + optional semantic processing
  → deterministic page/graph generation and changed-path writes
  → registry synthesis/ownership + state + source signatures
  → successful pipeline checkpoint
```

1. `olw ingest` persists admitted sources, indexes URL/content identity, and normally invokes compilation. `--plan` (also legacy `--dry-run`) is local-only; `--preview` permits real extraction without source writes or synthesis. `--skip-synthesis` separates acquisition from compilation. A refreshed URL can replace its body or title/provenance revision in place with a backup. An already-known URL still reaches compilation unless `--skip-synthesis` is set, allowing due failed synthesis to recover without a new extraction. Source names reserve `index.md` and navigation-shard patterns so an article titled “Index” is not silently excluded.
2. `olw build` calls `load_sources_incremental`. It **enumerates and stats every source**, compares `(mtime_ns, size_bytes)` against committed signatures, and reads changed bodies. A before/after stat guards ordinary concurrent changes. Unchanged files become metadata stubs. Discovery validates resolved containment before reading: links escaping the source directory are rejected, while contained links keep their logical filename. Missing previously populated directories, unreadable files, and incomplete inventories must not be treated as authoritative deletion lists.
3. With identical source sets, all signatures trusted, matching artifact versions, and a successful configuration checkpoint, `run_pipeline` returns before loading synthesis/state or rendering. This is not an audit of generated files. A same-size source edit with an intentionally preserved mtime can evade signature detection; `--force` reads all sources at the cost of re-synthesis.
4. Otherwise the checkpoint is cleared, live syntheses are loaded, and changed/incompatible sources are compiled. Failed replacements retain last-good syntheses. A failed first-time source has no last-good output to retain. Invalid inputs do not advance successful state. A frontmatter-only source title change updates synthesis metadata/cache and entry/graph labels without a new LLM request; it is not discarded merely because the body hash matches.
5. The live corpus is merged in memory. Related concepts may receive additional coherent re-synthesis in two-pass mode; the overlay cache prevents that text reverting immediately to mechanical concatenation. Optional embedding failures degrade semantic enhancements, not base source synthesis.
6. Rendering computes fingerprints/manifests over the full bundle, then writes only changed eligible paths. Per-path journaling and backups support rollback of caught render failures, without overwriting detected concurrent human edits. Deletions of registry/cache source records occur after rendering. Concepts whose last source owners were deleted are retained as orphaned pages; remaining owners preserve shared concepts.
7. Successful syntheses, ownership, `state.json`, and observed source signatures are persisted; an error-free run publishes its checkpoint. Cleanup releases unfinished leases and the compile lock even if metrics persistence fails.

Source dependency JSON/Mermaid and graph shards use the shared generated-artifact containment/write/delete boundary. A source-graph export exception now propagates through `render_vault` rather than being a warning-only success; the enclosing render journal handles caught failure rollback. Calling an exporter directly does not create that enclosing journal. Sharding still builds the full graph in memory, and safety checks do not eliminate all filesystem race windows.

There is **no single transaction across SQLite, portable JSON, all Markdown, and external APIs**. A hard kill between stages can leave partially published artifacts. Last-good state, invalidated checkpoints, backups, and repeatable reconciliation reduce damage; they do not make the process exactly-once. Use the [recovery procedure](migration.md).

## Synthesis and acceptance

- `SYNTHESIS_MODE=single`: one structured request for an ordinary source, but chunking and retries can add requests.
- `SYNTHESIS_MODE=two_pass`: skeleton extraction followed by focused concept expansion. Large inputs are chunked for extraction; Pass 2 uses selected evidence windows, not necessarily the entire source.
- A shared request limiter bounds concurrent synthesis/chunk/expansion/repair work. Bounded source workers and bounded per-source chunk workers avoid allocating one live coroutine per source/chunk; the chunk text/result lists still reside in memory. Request concurrency is shared across workers, not multiplied by source × chunk counts.
- Parse/content retry uses the complete source rather than silently falling back to a prefix. Every single-pass chunk is parsed and `validate_synthesis`-checked against that chunk **before** merging; one exhausted invalid chunk rejects the entire source. Two-pass skeleton acceptance likewise checks each chunk while allowing bodyless skeletons for later expansion. Final merged synthesis/evidence is checked against the complete source. Cancellation propagates rather than becoming a successful partial result.
- [`synth/validation.py`](../obsidian_llm_wiki/synth/validation.py) rejects blank/generic placeholders, explicit provider-inability nonanswers, incomplete concept sections, and supplied quotes that cannot be found in the source. Quoted/source-appropriate discussions of access restrictions are not automatically provider disclaimers. It allows a concise substantive summary and an empty concept list. It imposes no fixed word count or minimum concept count.
- Exact quote occurrence establishes location, **not entailment**. Repeated quotes are ambiguous; legacy `source_ref` claims can lack verifiable spans. A polished, structurally valid hallucination can still pass.
- `CONCEPT_MIN_BODY_CHARS` and `ENTRY_MIN_BODY_CHARS` remain diagnostic/legacy quality settings, not an instruction to pad content. Length-derived confidence is not calibrated factual confidence.

### Language and bilingual notes

`OUTPUT_LANGUAGE` overrides detected source language. Otherwise detection is heuristic, with English as a fallback—not a mandate to translate everything into English. Generated explanations follow the selected language; quoted evidence should retain its source language.

The active `render/bilingual.py` normalizer only makes paths safe and aligns source-local references with canonical slugs. It preserves supplied titles/aliases, including Chinese-only and explicitly bilingual names, without translation or English-first reordering. The historically named `ensure_english_first_bilingual` and related presentation utilities remain importable for explicit legacy callers; the production normalizer does not use them. Chinese titles can generate Unicode-safe slugs rather than an empty ASCII fallback. Parser conversion applies the same old→normalized slug map to related edges, claims, and MoC membership.

Deterministic entry/concept/MoC headings are localized for English/Chinese note language. A MoC can link both languages but does not embed another-language concept summary/preview as its own prose; this is link navigation, not automatic translation. Typed edges produce clickable crosslinks outside diagrams. [Human assertions in `equivalences.json`](data-model.md#human-confirmed-equivalences) create symmetric `equivalent_to` links and authorize direct counterpart membership even with embeddings off, without merging notes. Cosine scores alone do neither. Optional embedding-based orphan-to-MoC topic assignment remains a heuristic and needs review.

## CLI outcome contract

`build`, `recompile`, and `ingest` report failure with exit 1, cancellation with 130, and success/no-op with 0. Ingest NDJSON emits terminal `failed` or `partial_failure`, explicit errors, and extraction/synthesis/inventory failure counts; consumers must inspect the terminal event, not just whether a source was acquired. Successful work is not erased because another item failed.

`build --dry-run` returns before SQLite opens: it checks all source bodies against `state.json` and prints new/changed/unchanged/deleted filenames, force behavior, and content-only synthesis candidates. It makes no provider calls or vault writes, including on an absent vault. It cannot predict schema/cache compatibility, queue eligibility, or render changes; JSON state may lag SQLite. `ingest --plan` is local-only; `--preview` deliberately performs real extraction but does not persist it.

## Extraction and admission

Supported routing is a capability, not a promise that every URL yields complete evidence.

| Input | Current routing and limits |
| --- | --- |
| Generic web | Scientific preflight for recognized landing pages, then defuddle.md, optional discovered-document/LiteParse route, trafilatura, local defuddle CLI, applicable specialist/archive fallbacks. Ordering and early success matter: not every fallback runs on every request. |
| PDF | Bounded document download with MIME/signature checks; PyMuPDF first, optional LiteParse fallback. An image-only or protected PDF may fail or lack usable text. |
| DOCX / other supported binary documents | DOCX has a `python-docx` route; other document formats rely on the optional parser's actual capabilities. An extension match alone is not successful extraction. |
| Scientific papers / JATS | Official accessible HTML/XML or discovered public document URLs. Discovery is candidate-bounded, and selected document redirects must satisfy the applicable host restriction. XML parsing uses defusedxml. |
| YouTube | Transcript services/subtitles and configured ASR fallbacks depend on availability and credentials. Title/channel metadata is not a transcript; explicit unavailable-transcript placeholders are rejected at the public admission boundary. |
| Podcasts | Cache/publisher transcript first; feed discovery, remote ASR, or local transcription when configured and media is obtainable. RSS feeds, transcript artifacts, and media are distinct steps. Missing transcript content must not become a successful transcript source. |
| X/Twitter / Notion / local text | Specialized handling; short genuine content may be legitimate. Remote restricted shells still fail admission. Local text is not authenticated third-party evidence. |

Specialists that match and fail do not silently fall through to generic HTML as if it were their content. A non-applicable specialist may allow other routing. Generic web/deep-search quality checks reject known stubs, metadata-only output, and boilerplate; short specialists may retain an extraction-quality diagnostic instead of being rejected. Exact denied/login titles and the application's own missing-transcript shell are rejected even for specialists. These are heuristic checks, not exhaustive paywall or truncation detection.

`DEEP_SEARCH_FALLBACK=true` is an opt-in **process environment** switch read directly by the extractor, not a `Config` field loaded from vault `.env`. For example:

```bash
DEEP_SEARCH_FALLBACK=true olw ingest "$HOME/MyVault" --url https://example.org/paper --preview
```

It can query Semantic Scholar, OpenAlex, arXiv, and Crossref for an accessible alternative. A result may be an abstract or a different revision, not the requested full text. Inspect requested/resolved/extracted URLs and diagnostics before accepting it. It does not grant access to paywalled material or guarantee identity.

URL safety rejects direct local/private targets and validates redirects on protected fetch paths. Document byte caps, HTML caps, candidate caps, and parser time/output caps are distinct. They do **not** establish a sandbox for every optional subprocess, outsourced fetcher, redirect/DNS race, media download, or decompression path. `RESIDENTIAL_PROXY_URL` and `YOUTUBE_COOKIES_FILE` are explicit settings; use only authorized access, and account for third-party disclosure when enabling remote extraction/transcription.

Known truncated input or input exceeding the synthesis byte cap is rejected rather than published as complete. Raising the limit is not recovery of already lost text: re-extract the source. Undetected upstream truncation remains possible.

## Resource settings

Names/defaults below come from [`config.py`](../obsidian_llm_wiki/config.py). They constrain particular operations, not total vault capacity.

| Setting | Default | Scope |
| --- | ---: | --- |
| `COMPILE_CONCURRENCY` | 3 | Synthesis workers/request limiter; ingest also has `--parallel`. |
| `REQUEST_MIN_INTERVAL_MS` | 0 | Minimum interval between requests under the limiter. |
| `CHUNK_SIZE` | 30000 | Text chunk target in characters. |
| `MIN_SOURCE_CHARS` / `MAX_SOURCE_CHARS` | 50 / 1000000 | Despite the names, pipeline admission compares **UTF-8 bytes**. |
| `CANDIDATE_LIMIT` / `CANDIDATE_CHAR_BUDGET` | 100 / 16000 | Synthesis candidate count / serialized character budget. |
| `ANN_CANDIDATE_POOL` | 1000 | Exact cosine rerank pool, not guaranteed ANN recall. |
| `EVIDENCE_WINDOW_CHARS` | 15000 | Pass 2 selected evidence text. |
| `EVIDENCE_CONTEXT_PARAGRAPHS` | 1 | Neighbor context in evidence selection. |
| `INDEX_SHARD_SIZE` / `REPORT_SHARD_SIZE` | 2000 / 2000 | Derived output shard limits. |
| `GRAPH_SHARD_SIZE` / `GRAPH_MONOLITH_THRESHOLD` | 5000 / 10000 | Graph artifacts; Mermaid also has its own safety limits. |
| `MAX_DOCUMENT_BYTES` / `MAX_DOCUMENT_CANDIDATES` | 50000000 / 10 | Binary download cap / discovered candidates. |
| `MAX_HTML_BYTES` | 10000000 | Bounded HTML retrieval paths. |
| `PARSER_TIMEOUT_SECONDS` | 120 | Optional document subprocess. |
| `MAX_PARSER_STDOUT_BYTES` / `MAX_PARSER_STDERR_BYTES` | 1000000 / 16384 | Parser output limits; overflow is failure, not valid prefix text. |
| `CONCEPT_MIN_BODY_CHARS` / `ENTRY_MIN_BODY_CHARS` | 1200 / 500 | Quality/health diagnostics, not a generation quota. |
| `CLIPPING_MIN_BODY_CHARS` | 500 | Clipping admission heuristic, distinct from synthesis acceptance. |
| `RETRY_COUNT` / `RETRY_BASE_MS` / `RETRY_MULTIPLIER` | 3 / 1000 / 4 | Provider retries, separate from content retries and durable queue scheduling. |
| `SIMILARITY_DEDUP_THRESHOLD` | 0.85 | Candidate similarity threshold; identity permission is additionally required. |
| `CROSS_LINGUAL_THRESHOLD` | 0.82 | Cross-language candidate threshold. |
| `MOC_ASSIGNMENT_THRESHOLD` | 0.55 | Optional topic-membership heuristic. |

`LLM_TIMEOUT_MS` defaults to 1800000 and `LLM_CONTEXT_WINDOW` to 256000; confirm the chosen provider/model can support them. Configuration validates numeric ranges and finite similarity thresholds. Malformed numeric strings can fall back to defaults, so do not assume every typo is an error.

## Queue, retry, and locking

SQLite `dirty_queue` records entity/reason, payload, attempts, next-attempt time, state, worker, error, lease expiry/token, and a successor payload/due-time slot. Due-item claims are bounded; the current lease is 600 seconds. Pipeline sources are claimed when a bounded worker starts, not all at enqueue time. A 30-second asynchronous heartbeat renews still-live claims for that unique pipeline worker lifecycle. Ordinary re-enqueue preserves retry scheduling. Re-enqueue during a live claim leaves its payload immutable and coalesces the newest invalidation into a durable successor. Completing the old work promotes that successor rather than acknowledging it as finished. Failure returns work to pending with exponential backoff (capped at a day), or promotes waiting successor work. Completion/failure requires the unexpired owner/token. Pipeline cleanup releases unfinished claims; after acquiring the exclusive compile lock, startup releases only proven-abandoned `pipeline:` claims (not `ingest:` or similarly prefixed owners). Finalization return values are checked: lease loss becomes an error rather than a successful acknowledgment/checkpoint.

`build --force` requests `reset_retry=True` for pending synthesis work; ordinary scheduling does not reset backoff. A reset cannot steal or mutate a live claim. Callers must use a unique worker identity per claim lifecycle, or a fresh `lease_token` echoed to completion/failure when reusing worker IDs. These mechanisms do not constitute a distributed exactly-once scheduler or atomic file publication. Do not use concurrent ingest/build writers as a supported scaling strategy. The heartbeat runs while asynchronous inference yields; synchronous event-loop stalls/process suspension can still outlast a lease. Renewal does not resurrect expired claims and is not a distributed lease guarantee. The compile lock uses an OS guard plus PID metadata; do not manually remove an apparently stale lock while a process may still own it.

## Human-edit protection

For generated Markdown the renderer compares current content with `.llmwiki/render-page-hashes.json`. A mismatch, missing trusted hash, or `reviewed: true` prevents replacement. The protected file's entire contents are preserved, not just its body. Sources already on disk are preserved as inputs. Unchanged recognized generated pages can be replaced with a backup when the model fingerprint changes.

Missing hashes after upgrade intentionally protect legacy notes; neither `--force` nor `--refresh-render` means “overwrite my edits.” Move a reviewed legacy page aside only after backing up and deciding how to merge it with newly generated output. Maintenance `olw fix --apply` is a separate deterministic repair path: preview its plan, and mark pages `reviewed: true` when they must not be maintained automatically.

## Query is a separate path

Synthesis retrieves candidates from SQLite FTS5/optional ANN. `olw query` builds its own wiki page/graph context and ranks lexical matches plus personalized PageRank. A result limit does not imply bounded corpus reads. Citation validation checks path resolution, not whether the cited text entails the answer. Saved answers default to `05-Queries/`.
