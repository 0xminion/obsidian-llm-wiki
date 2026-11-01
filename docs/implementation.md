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
  → deterministic page/graph generation in a private workspace
  → private registry synthesis/ownership + state + source signatures
  → complete immutable generation + atomic current.json pointer
```

1. `olw ingest` persists admitted sources, indexes URL/content identity, and normally invokes compilation. `--plan` (also legacy `--dry-run`) is local-only; `--preview` permits real extraction without source writes or synthesis. `--skip-synthesis` separates acquisition from compilation. Mutating `--refresh` is refused; `--refresh --preview` can inspect new extraction, which an operator can save under a new input filename. Existing sources, clippings, and inbox files stay in place. An already-known URL still reaches compilation unless `--skip-synthesis` is set, allowing due failed synthesis to recover without a new extraction. Source names reserve `index.md` and navigation-shard patterns so an article titled “Index” is not silently excluded.
2. `olw build` uses `core.application.source_inventory` to read the full human source corpus without a writable registry or flat-state dependency. The loader validates the inventory and containment; incomplete inventories must not be interpreted as source deletion.
3. `run_pipeline` delegates to `publish_pipeline`. It hashes observed source/policy inputs and validates the current generation's file inventory and registry/cache/page coherence. A matching compatible, error-free generation returns a zero-model-call no-op unless force/recompile/refresh was requested. This validates output; it is not merely an mtime/signature checkpoint.
4. A changed publication creates a new private staging vault, seeding a validated previous generation or closed legacy metadata. Human output pages are never copied into writable output locations. Sources are captured separately; synthesis/review/identity and any schema/cache migration run privately. Failed replacements retain last-good knowledge with stale status rather than certifying old content as current.
5. The live corpus is merged in memory. Related concepts may receive additional coherent re-synthesis in two-pass mode only with `GROUNDING_REVIEW=off` (never-reviewed cross-source rewrites are disabled under required review); the overlay cache prevents that text reverting immediately to mechanical concatenation. Optional embedding failures degrade semantic enhancements, not base source synthesis.
6. Rendering, registry/cache/state updates, source views, and qualified generation links are produced inside that workspace. This is a complete projection, not a small in-place update. The implementation rechecks input/policy bytes and validates semantic-plan/cache/SQL/page coherence before publication.
7. A complete generation is installed without replacement and the application-owned current pointer is advanced. Existing human files and previous generations remain unchanged. `olw current VAULT --json` returns the validated entrypoint; callers must not treat a flat legacy homepage or an uncommitted workspace as current.

Source dependency JSON/Mermaid and graph shards use the shared generated-artifact containment/write/delete boundary. A source-graph export exception now propagates through `render_vault` rather than being a warning-only success; the enclosing render journal handles caught failure rollback. Calling an exporter directly does not create that enclosing journal. Sharding still builds the full graph in memory, and safety checks do not eliminate all filesystem race windows.

A generation is installed exclusively and accepted only through the app-owned `OLW-Publications/current.json` commit pointer. Before-pointer process crashes leave the previous generation current; after-pointer crashes do not roll it back. Failed/uncommitted workspaces are retained, not automatically promoted. There is still **no single transaction across extraction, external APIs, and all vault state**, nor a power-loss/filesystem durability certification. Do not repair corrupt installed generations in place. Use the [recovery procedure](migration.md) and the [immutable publication contract](immutable-publication.md).

## Synthesis and acceptance

- `SYNTHESIS_MODE=single`: one structured producer request per span-catalog window (one window for an ordinary source); chunking, diagnostic-aware retries and review batches add requests.
- `SYNTHESIS_MODE=two_pass` with `GROUNDING_REVIEW=required`: per window, an unchecked span-range draft, then a complete refinement over the same window (`task=expand`, `LLM_EXPAND_MODEL` when set). Only the refinement is reviewed and published; a rejected refinement never falls back to the draft. With `GROUNDING_REVIEW=off`, two_pass is the legacy unchecked skeleton + per-concept expansion path (quote evidence), marked `unchecked`.
- Grounded citations (interface `evidence-span-range-v1`): each claim names ONE contiguous `evidence_span_range {start_id, end_id}`; the application materializes every intervening offered span ID and builds the exact quote/offsets through the strict catalog (≤6 expanded IDs, ≤4000 chars, offered window, exact source revision). Free `evidence_span_ids` lists from a producer are rejected, never widened. Persisted evidence keeps its expanded `span_ids`.
- Grounding review (receipt `grounded-review-v2`, unit identity `review-unit-identity-v2`): every published statement AND label is a unit — summary paragraphs, source tags, key points, questions, concept titles, aliases, tags, section headings, body points/prose, claims (with cited text and `source_ref`), relations as the published sentence `<subject> <relation> <target label>`, MoC titles, summary paragraphs, tags and member labels (`map_tag`, `map_member`). Units carry their subject (and heading); keys hash exact text (whitespace is content) plus context; claim keys bind the exact quote and offsets. Identities (slugs, relation target labels) are normalized before review; a merge that changes a unit (e.g. a relation whose target is defined in another window) triggers a bounded final admission review against the origin windows. Receipts are verified for catalog version, unit-set digest, per-unit kind/verdict and support span IDs of the current revision, even on the unchanged-signature path.
- Chunked sources merge losslessly: every window's summary paragraph, key point, question, section, claim, relation, alias and tag is kept (exact duplicates only removed); a same-slug concept from a later window is merged, its differing title kept as a reviewed alias.
- Retained last-good output (changed source not admitted, or a receipt that does not satisfy the current review configuration) stays published but is labelled with an Obsidian warning callout, and its evidence is not shown as current verified evidence. User-edited pages are never rewritten to add the label.
- Identity binding (`core/entity_identity.py`, `synth/identity.py`, `VaultRegistry._reuse_is_bound`, checkpoint `integrity-v4-identity-evidence`). A model `canonical_id`, a copied canonical slug or an owned slug is an untrusted matching proposal. **Names never prove identity**: equal titles, a self-alias, a reviewed alias naming the incumbent, a shared (even rare) word, a missing definition, or definitions in another language are absence of evidence. A proposal binds only on one positive basis, decided BEFORE any registry mutation:
  - *Sole owner*: the source is the entity's only owner (a revision may update its own entity, including its title/definition).
  - *Continuity*: a co-owner whose own `(title, summary)` snapshot is unchanged since its binding was established — the incumbent (earliest owner), or an owner bound by an assessment. A co-owned incumbent that rewrites its definition (e.g. Python language → snake) must re-prove against the next owner's definition; otherwise it splits off to its own ID/slug, the other owners keep the entity and slug, and other sources' links are not retargeted.
  - *Identical definition*: equal non-empty normalized `(title, summary)` snapshot as the candidate's definition (the incumbent among the OTHER owners; an orphan row's own definition). Same-run reservations use the same rule; empty summaries never match.
  - *Explicit identity assessment*: when no other basis applies, the pipeline sends the proposal to the review model (`task="identity"`, same model as review) with the subject's reviewed source spans, the actual candidate ID and both definition snapshots (SHA-256). The closed schema forces the assessor to echo ID and snapshots; the app validates echo, vocabulary (`same`/`different`/`insufficient`) and offered support spans. Records (and the producer's original hint) are stored in the synthesis receipt (`receipt["identity"]`, `identity-assessment-v1`). The registry accepts only an approved receipt's `same` record of the receipt's own source revision, with support spans, bound to exactly this candidate ID, the candidate's CURRENT snapshot and the proposal's own snapshot. A changed snapshot on either side invalidates it (re-asked on the next build); terminal verdicts are not re-asked.
  - Anything else stays a distinct entity under its own title (definition-derived slug suffix on a collision); facts, relations and MoC membership are kept. An identity verdict never rejects a synthesis, and support/citation review is unchanged: unsupported output is rejected before any assessment.
  - Direct API callers (`upsert_synthesis`, `update_synthesis`, `assign_canonical_ids`) get no implicit approval: without an assessment record only ownership or an identical definition binds. `registry.identity_requests(synthesis)` (pure) lists what needs an assessment; `registry.binding_evidence()` reports the basis of each `(source, entity)` binding.
  - Publication: the pipeline passes the binding bases to the plan; a page with several contributing sources refuses (visible `unbound` error) any contribution without a basis, even under an equal title. A contribution with a `same` assessment may be published under the page title although its own label differs (e.g. `阿司匹林` under `Aspirin`); its claims show `reviewed under “…”`.
  - Labels decide naming only (page titles, relation displays, MoC member labels): case/whitespace-insensitive, typographic spacing/connectors (`FlashAttention` = `Flash Attention` = `Flash-Attention`) and both halves of an explicit cross-script bilingual title match. Semantic punctuation is preserved: `C++`, `C#`, `C`, `F#`, `F*`, `.NET`, `3.11` are distinct labels.
  - MoC analogue (the one remaining automatic, non-assessed match): a MoC slug collision across sources is reused only with ownership, or the same exact title, same detected language AND a shared member that the proposing source itself defines (a member identity-bound by the rules above); externally referenced member slugs never count. Two same-titled topics that share an identity-bound member therefore merge without an assessment; their summaries stay separate paragraphs and every member/tag is a reviewed unit.
  - Other analogues audited: within-source merges (same slug in one source or across chunk windows) are the producer's own local identity; embedding semantic dedupe merges only concepts already sharing a registry ID or an explicit `confirmed_merges` mapping; cross-lingual similarity and `equivalences.json` create links, never merges.
  - Tradeoffs: legitimate cross-source reuse costs one assessor request per unbound proposal (cached in the receipt; a rebuild is a zero-call no-op). `GROUNDING_REVIEW=off` never assesses: its output reuses entities only by ownership or identical definition. A co-owned entity's wording update needs a fresh assessment. A failed assessment (provider error, contract violation) is a visible `identity:` build error, keeps the concept distinct and is retried on the next build without a producer call. The first build after upgrading re-evaluates every cached binding: legacy co-owners without evidence are assessed once (reviewer calls only) and kept or split; review-off caches split. A sole owner may still rewrite its own entity; other sources' relations to its slug stay label-bound (a relation is live only while its reviewed display names the page), not identity-assessed. A `same` verdict is a model judgment, not proof of truth.
- Publication identity (`synth/publication.py`). A receipt binds units to the subject/label they were reviewed under in ONE source; the final publication plan binds them to the page that publishes them. Pre-review normalization never collapses concepts by raw model IDs; one source cannot claim one existing ID for two concepts.
  - Cross-source merges keep every contributor's title (as an alias), summary paragraphs, sections, claims (each with its own evidence) and typed relations; MoC merges keep every summary paragraph, tag and member. Each contribution must carry an identity basis (above) and be bound to the page title (one of its own reviewed labels, attested by another contributor's reviewed labels, or a `same` assessment of this entity); claims bound through another subject show `reviewed under “…”` on their evidence line. An unbound admitted contribution is a visible build error and is labelled `unbound`.
  - A relation is a live link only if its reviewed display label names the actual target page (title, bound alias, or literal slug); otherwise the assertion is published unlinked under "Unresolved Relations" (target existing or created later). Inverse backlinks are app-derived structure in a separate "Derived Backlinks" section, derived only from live forward edges and carrying the forward edge's status (stale/unchecked/unbound) — never inserted into `relations`. MoC members are `reviewed` only through a bound `map_member` unit; app-added members (similarity assignment, confirmed equivalences) are typed "app-derived membership; not reviewed".
  - `GROUNDING_REVIEW=off` output is published with an `unchecked` warning callout and `[n*]` claim markers, including on mixed pages.
  - The renderer re-verifies the plan after its own normalization; any later retitle/retarget is reported as an error and re-planned, never published silently.
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

`build --dry-run` validates current publication without a writable SQLite open, then checks all source bodies against current-generation (or legacy fallback) `state.json` and prints new/changed/unchanged/deleted filenames, force behavior, and content-only synthesis candidates. It makes no provider calls or vault writes, including on an absent vault. It cannot predict schema/cache compatibility, queue eligibility, or render changes; JSON state may lag SQLite. `ingest --plan` is local-only; `--preview` deliberately performs real extraction but does not persist it.

## Extraction and admission

Supported routing is a capability, not a promise that every URL yields complete evidence.

| Input | Current routing and limits |
| --- | --- |
| Generic web | Scientific preflight for recognized landing pages, then defuddle.md, optional discovered-document/LiteParse route, trafilatura, local defuddle CLI, applicable specialist/archive fallbacks. Ordering and early success matter: not every fallback runs on every request. |
| PDF | Bounded document download with MIME/signature checks; PyMuPDF first, optional LiteParse fallback. An image-only or protected PDF may fail or lack usable text. |
| DOCX / other supported binary documents | DOCX has a `python-docx` route; other document formats rely on the optional parser's actual capabilities. An extension match alone is not successful extraction. |
| Legacy Word `.doc` | Word 97–2003 uses isolated LibreOffice DOC→DOCX conversion plus `python-docx` when available. Encryption, unsupported Word versions and conversion failures fail closed; dependency unavailability may use a diagnostic-labelled LiteParse compatibility route. Images and embedded content are not fully extracted. See [native route and limits](legacy-word-extraction.md). |
| Scientific papers / JATS | Official accessible HTML/XML or discovered public document URLs. Discovery is candidate-bounded, and selected document redirects must satisfy the applicable host restriction. XML parsing uses defusedxml. |
| YouTube | Transcript services/subtitles and configured ASR fallbacks depend on availability and credentials. Title/channel metadata is not a transcript; explicit unavailable-transcript placeholders are rejected at the public admission boundary. |
| Podcasts | Cache/publisher transcript first, including one explicitly labelled public transcript link for a uniquely matched episode; feed discovery, remote ASR, or local transcription when configured and media is obtainable. Descriptions/show-only pages are rejected as incomplete. RSS, transcript artifacts, and media are distinct steps. |
| Public Drive file viewers | Only exact supported file-viewer routes with a matching advertised anonymous document download. Identity, redirects, MIME/signature, size and time are checked. Private files and confirmation pages fail; no permission bypass. |
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

Public builds preserve legacy notes, original sources, human policy, annotations, and installed generations at their existing paths. Rendering occurs only in a fresh private workspace; its internal hash/backup machinery does not authorize writing into human files or installed generations. Edit retained legacy notes or separate `03-Raw-Annotations/entities/<canonical-id>.md` annotations. No move-aside step or `reviewed: true` flag is needed to keep those files intact. Manual generation edits are corruption and cause refusal.

`olw fix VAULT --dry-run` (also the default) scans the validated current projection, or legacy pages before the first publication, without writes or model calls. `olw fix VAULT --apply` requires source inputs and invokes `run_pipeline(..., refresh_render=True)` to publish a **complete new immutable projection through normal synthesis/review gates**. It is not a deterministic in-place repair command and does not execute each listed text edit or resolve human-review findings. Changed/incompatible inputs may require producer/reviewer model calls and incur cost. Existing human files and old generations remain unchanged. Result JSON uses `mode: publish`, `generation`, and `summary.pages_published`.

`fix --restore` is retained for compatibility but is always refused, even for trusted backups. Inspect/copy backup content into a new separate annotation, or republish via `build --refresh-render` when current publication is valid. Republish is not backup restoration and does not bypass corrupt-generation validation. See [recovery guidance](migration.md#single-backup-inspection-and-republishing).

## Query is a separate path

Synthesis retrieves candidates from SQLite FTS5/optional ANN. `olw query` builds its own wiki page/graph context and ranks lexical matches plus personalized PageRank. A result limit does not imply bounded corpus reads. Citation validation checks path resolution, not whether the cited text entails the answer. Saved answers default to `05-Queries/`.
