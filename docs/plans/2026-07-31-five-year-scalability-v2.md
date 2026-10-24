# Five-Year Vault Scalability v2 Implementation Plan

> **Historical plan — not active instructions or a capacity guarantee.** Baseline results and targets below belong to this plan's original context. Use the [active documentation index](../README.md), especially [scaling](../scaling.md) and [migration/recovery](../migration.md), for current behavior. Retained as design/delivery history.

## Objective

Implement the roadmap in `docs/scalability-roadmap.md` for a steady-state vault of
5,000 sources, 50,000 concepts, and 20,000 MoCs without weakening extraction,
synthesis, evidence, review protection, rollback safety, or Obsidian output quality.
The real vault is never used as a test fixture.

## Baseline

- Base revision: `2674c443236267be1fc0facf9968185a90c7790b`
- Baseline lint: clean
- Baseline tests: 877 passed
- Host budget at start: 8 CPUs, 23 GiB RAM, about 9 GiB available
- Implementation/test worker budget: at most 2 concurrent read-only agents
- Live extraction/synthesis budget: 1 worker for E2E verification

## Non-negotiable invariants

1. Unchanged source knowledge remains available after a transient synthesis failure.
2. Reviewed pages retain their human-authored body.
3. A failed render restores every renderer-owned path it changed, but never overwrites
   a concurrent human edit.
4. Existing JSON state/cache files migrate once and remain readable during rollback.
5. Incremental mode never uses an all-pairs concept or orphan-to-MoC loop.
6. Candidate context and evidence text are bounded before entering any LLM prompt.
7. A model may reuse a canonical concept only by its stable registry ID; a new concept
   receives a deterministic stable ID at the parser/registry boundary.
8. Normal builds do not scan every source Markdown file, cache JSON file, or rendered
   page. Explicit full audits and migrations may do so and must be observable.
9. Generated graph/index/report artifacts shard above configured thresholds. Mermaid
   output is disabled above its safe threshold rather than emitting unusable files.
10. Every remote request path shares the same source-run limiter. Nested Pass 1 chunks,
    Pass 2 expansions, repairs, and resynthesis cannot each create independent capacity.

## Architecture

### 1. Vault registry

Add a WAL-mode SQLite registry at `04-Wiki/.llmwiki/registry.db` with indexed tables:

- `source`, `source_version`, `synthesis`
- `concept`, `concept_alias`, `moc`, `membership`
- `embedding`, `ann_bucket`
- `dirty_queue`, `operation`, `ingest_hash`, `health_finding`, `meta`

The registry stores source hashes, canonical URLs, artifact versions, serialized
syntheses/canonical models, float32 embeddings, queue state, and generated-path dirtiness.
Schema migrations are transactional and versioned in `meta`.

Legacy migration imports `state.json`, per-source synthesis cache JSON, embeddings JSON,
and operations JSON exactly once. It records completion only after a committed import.
Normal registry startup does not repeat filesystem scans.

### 2. Version contract

Every source/synthesis/embedding record carries the versions that affect its meaning:

- extractor version
- prompt version
- synthesis schema version
- synthesis model ID
- embedding model ID and dimensions
- source/embedding input hash

An incompatible version does not silently reuse stale data. It inserts an explicit dirty
queue record with the invalidation reason. Compatible cache data is reused.

### 3. Bounded retrieval and stable IDs

Replace the full concept slug prompt with a `ConceptRetriever`:

- SQLite FTS5 lexical candidates
- bounded local ANN candidates over persisted embeddings
- exact cosine reranking only inside the bounded candidate pool
- deterministic union/deduplication capped by `candidate_limit` (50-200)

Prompt candidates are rendered as stable ID, canonical slug, and title. Synthesis output
uses `canonical_id` when reusing a candidate. New concepts receive a deterministic ID and
are indexed with aliases. Invalid or unknown IDs cannot redirect a concept to an arbitrary
path.

### 4. Local ANN index

Use deterministic multi-table sparse random-hyperplane LSH stored in `ann_bucket`:

- float32 vectors remain authoritative in `embedding`
- multiple independently seeded tables reduce boundary misses
- exact and Hamming-distance-one buckets produce a bounded candidate pool
- exact cosine reranks candidates
- entity kind/model/dimension are part of every key

Incremental semantic dedupe, cross-lingual linking, and orphan-to-MoC assignment receive
candidate maps from the ANN index. Global reconciliation remains an explicit offline mode;
normal builds cannot fall back to all-pairs behavior.

### 5. Shared request limiter and evidence windows

Create one limiter per pipeline run and pass it through:

- single-pass synthesis
- chunked Pass 1
- two-pass Pass 1
- Pass 2 expansion
- parse repair/retry
- concept resynthesis

Pass 2 receives deterministic evidence windows selected from paragraphs that match the
concept title, aliases, rationale, claims, and candidate terms. Windows preserve source
order, include bounded context around matches, and fall back to bounded head/middle/tail
coverage when lexical evidence is sparse. The prompt records omitted-character counts.

Structured parse failures are persisted in `dirty_queue` with bounded diagnostics and a
next-attempt time. Success atomically completes the queue item.

### 6. Durable ingest index and retry queue

Normalize requested/canonical URLs and atomically upsert URL/content hashes. Existing
source Markdown is scanned only during the one-time migration. Subsequent duplicate checks
are indexed lookups. Extraction workers are separately bounded from synthesis workers;
default extraction concurrency remains conservative and E2E uses one worker.

Failed URLs retain extractor chain, bounded error text, attempt count, and next-attempt
time. Resume claims pending work transactionally, so interruption cannot duplicate writes.

### 7. Incremental renderer

Introduce a render plan containing dirty source/entry/concept/MoC IDs, stale paths, and
index/graph shard IDs. Normal rendering:

- iterates only dirty entity sets
- does not glob whole content directories for stale cleanup
- backs up only paths that may change
- journals preimages lazily before each write/delete
- restores only journaled paths on failure while preserving concurrent edits

Explicit full render/audit mode retains complete reconciliation behavior.

### 8. Sharded derived artifacts

Above configured thresholds:

- directory indexes become bounded hash-prefix shards plus a small manifest page
- graph JSON becomes a manifest plus bounded node/edge shards
- source dependency graph follows the same policy
- Mermaid graph export is omitted with an explicit manifest reason
- Dataview views remain fixed-size query-driven pages

Incremental render updates only affected shards. Below thresholds, current filenames and
formats remain backward compatible.

### 9. Incremental maintenance

Persist findings keyed by path and finding kind. After a normal render, rescan only written
or deleted paths and dependency-invalidated paths. `olw health --full` performs a complete
audit and refreshes the index; default health output can read the indexed snapshot and
reports whether a full audit is pending.

Metrics and operations keep bounded live rows; older rows move to archive tables instead
of disappearing or forcing whole-file rewrites.

## TDD delivery slices

1. Registry schema, migration, source/synthesis round-trip, version invalidation.
2. Indexed ingest dedupe, retry queue claim/complete/reopen behavior.
3. Stable concept IDs, FTS retrieval, bounded prompt candidate rendering.
4. Float32 embedding round-trip, ANN recall fixture, bounded candidate evaluation.
5. ANN-wired dedupe/cross-lingual/MoC assignment with no all-pairs fallback.
6. Shared limiter across nested synthesis and deterministic evidence windows.
7. Lazy write journal rollback with concurrent edit preservation.
8. Dirty render plan, bounded index shards, graph/source-graph thresholds.
9. Incremental health finding updates and explicit full audit.
10. Synthetic 5k/50k/20k capacity gate and real-URL temporary-vault E2E.

Each slice follows RED -> GREEN -> focused regression -> integration suite.

## Capacity gates

The synthetic fixture must prove:

- 5,000 indexed sources upsert and resolve without source-tree scans
- 50,000 concepts are searchable with <= configured prompt candidates
- ANN exact-rerank evaluations never exceed the configured pool
- 20,000 MoCs are addressable without orphan x MoC iteration
- dirty queue claims are bounded and resume after process reopen
- one changed source produces a bounded render plan and bounded shard writes
- prompt candidate characters, evidence characters, graph shard records, and request
  concurrency all remain within configured budgets

Wall-clock assertions are avoided except for a generous smoke ceiling; behavioral counters
are the primary proof.

## Verification

1. Focused RED/GREEN tests per slice.
2. `python -m ruff check obsidian_llm_wiki tests`.
3. Full `pytest tests/new -q --tb=short`.
4. Capacity test with real SQLite/FTS/ANN data.
5. Real public URLs in a fresh temporary vault:
   - a normal web article
   - an arXiv paper/PDF path
   - a second related source to exercise candidate reuse/incremental rendering
6. Validate/health/query smoke checks against the temporary vault.
7. Inspect generated registry, queue, shards, frontmatter, links, and evidence.
8. Independent read-only review of the frozen diff for logic, data integrity, regressions,
   resource bounds, and missing coverage.
9. Remediate confirmed findings, rerun gates, and obtain a fresh scoped review if fixes
   change reviewed code.
10. Commit and push only after the last review and verification pass are complete.
