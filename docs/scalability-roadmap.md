# Five-Year Scalability Roadmap

> **Historical design record — not active operating instructions.** The completion claims, schema versions, and capacity gates below reflect the earlier roadmap, not current end-to-end certification. Changed builds still perform corpus-wide work. Use [current scaling limits and proposals](scaling.md), [implementation](implementation.md), and [migration/recovery](migration.md) instead. This record is retained for design history.

## Scope and conclusion

This document defines and tracks a target vault of **5,000 sources, 50,000 concepts, and 20,000 MoCs**. The v2 runtime implements the required storage, bounded retrieval, ANN candidate generation, path-incremental rendering, sharded derived artifacts, and incremental maintenance model.

The target is a v2 storage/indexing architecture, not a collection of tuning flags. Markdown and JSON remain portable artifacts; `.llmwiki/registry.db` is the authoritative incremental runtime index.

## Superseded v1 constraints

| Constraint | Current path | Why it fails at target scale |
| --- | --- | --- |
| Full concept index in synthesis prompts | `core/pipeline.py`, `synth/prompts.py` | 50k slugs alone exceed practical context budgets. |
| Pairwise semantic dedupe and cross-lingual links | `synth/dedupe.py`, `synth/embedding.py` | 50k concepts implies roughly 1.25B pair comparisons per global pass. |
| Orphan-to-MoC assignment | `synth/dedupe.py` | 50k orphan candidates × 20k MoCs can reach 1B comparisons. |
| Full render transaction | `render/obsidian.py` | Repeatedly scans/copies/re-writes tens of thousands of pages and grows backups. |
| JSON cache/state/embeddings | `core/cache.py`, `core/state.py`, `synth/embedding.py` | Whole-file rewrites, no indexed dirty queue, and expensive JSON vector storage. |
| Sequential ingest dedupe | `cli/ingest.py` | Full source-file scans per input trend toward quadratic I/O. |

## Delivery sequence

### P0 — required before sustained bulk ingestion

1. **Introduce a SQLite registry.**
   - Tables: source, source_version, synthesis, concept, concept_alias, moc, membership, embedding, dirty_queue, operation.
   - Every record carries content hash, extractor version, prompt/schema version, model ID, and timestamps.
   - Use indexed dirty queues rather than reconstructing state from filesystem scans.

2. **Replace full-corpus prompt context with retrieval.**
   - Keep the canonical concept registry outside prompts.
   - Retrieve 50–200 lexical/ANN candidates per source.
   - Require model output to reference candidates by stable ID or explicitly create a new concept.

3. **Replace all-pairs similarity with ANN candidate search.**
   - Store float32 vectors in SQLite-compatible vector storage, a local ANN index, or a dedicated vector store.
   - Query top-k candidates only for new/changed concepts and MoC centroids.
   - Schedule global reconciliation as an offline maintenance job, never in normal builds.

4. **Make rendering path-incremental.**
   - Record changed source, entry, concept, MoC, index shard, and graph shard paths in a journal.
   - Back up only paths that will change; do not snapshot the whole vault.
   - Treat generated views/graphs as derived artifacts with threshold-based sharding.

### P1 — required for quality and operational safety

5. **Bound two-pass synthesis.**
   - Use evidence-selected source windows in Pass 2, not the entire source per concept.
   - Apply one shared request/rate limiter across all passes, retries, and resynthesis.
   - Persist structured parse failures in the dirty queue and retry with model-specific repair prompts.

6. **Create a durable ingest hash index.**
   - Atomically upsert `(normalized_url, canonical_url, content_hash)`.
   - Use bounded parallel extraction workers.
   - Keep failed URLs in a retry queue with extractor diagnostics and next-attempt time.

7. **Version embeddings and caches.**
   - Cache metadata must include embedding model, dimensions, input hash, schema, extractor, prompt, and synthesis model versions.
   - Invalidate on any incompatible version change.

### P2 — maintenance and user experience

8. **Shard graph and static reports.**
   - Disable Mermaid export beyond a configurable node/edge threshold.
   - Emit paginated/sharded JSON graph artifacts and rely on Dataview or query-driven reports.

9. **Incremental maintenance.**
   - Maintain health findings incrementally; reserve full health scans for scheduled audits.
   - Archive operations/metrics outside the capped live history.

10. **Capacity tests and gates.**
    - Add synthetic 5k/50k/20k fixtures that verify no O(N²) path runs in incremental mode.
    - Enforce budgets for prompt tokens, candidate counts, render paths, graph size, and request concurrency.

## Acceptance criteria for v2

- A single changed source does not scan all source Markdown, all cache JSON, or all rendered pages.
- Synthesis receives a bounded candidate set and bounded evidence text.
- Similarity and MoC assignment inspect only ANN candidates, not all concepts/MoCs.
- A normal incremental build writes only affected pages plus bounded index shards.
- Cache/model/schema changes create explicit dirty records and observable migration work.
- Bulk ingestion resumes safely after interruption without duplicate extraction or synthesis.

## Implemented v2 runtime

| Requirement | Implementation |
| --- | --- |
| Transactional registry and migrations | `core/registry.py` — SQLite WAL schema v3, legacy JSON import marker, stable concept IDs, source ownership, FTS5, versioned synthesis rows, leased dirty/retry queues, render manifests, ingest hashes, and maintenance cursor. |
| Bounded candidate retrieval | `synth/retrieval.py` — FTS5 plus optional ANN union, candidate count and character budgets, canonical-ID prompt contract. |
| ANN without optional native dependencies | `core/registry.py` — deterministic multi-table sparse random-hyperplane LSH over float32 vectors with exact cosine reranking capped by `ann_candidate_pool`. |
| Bounded two-pass prompts | `synth/evidence_windows.py`, `core/request_limiter.py`, `synth/quality.py` — evidence windows and one shared request/rate limiter across extraction, expansion, retries, and resynthesis. |
| Durable ingest | `cli/ingest.py` — one-time legacy hash backfill, indexed URL/content dedupe, bounded extraction prefetch, and exponential retry records. |
| Incremental rendering | `render/obsidian.py` — model fingerprints, registry-selected dirty paths, timestamp-insensitive no-op writes, and lazy touched-path rollback/backups. |
| Sharded derived artifacts | `render/graph_export.py`, `render/dataview.py`, `render/obsidian.py` — stable hash-bucket graph shards, bounded static report shards, and bounded directory index shards. Mermaid becomes a summary beyond the configured monolith threshold. |
| Incremental maintenance | `cli/health.py` — persistent round-robin page batches by default; `olw health --full` remains the explicit full audit. |
| Capacity gates | `tests/new/test_scalability_capacity.py` and `tests/new/test_scalability_*` exercise the exact 5k/50k/20k fixture and enforce candidate, queue, index, graph, report, evidence, and concurrency budgets. |

### Runtime budgets

The defaults are deliberately conservative for a small server:

- synthesis and extraction concurrency: `COMPILE_CONCURRENCY=3` / `olw ingest --parallel 3` (hard-capped to eight extraction workers);
- lexical/ANN prompt candidates: `CANDIDATE_LIMIT=100`, `CANDIDATE_CHAR_BUDGET=16000`;
- ANN exact rerank pool: `ANN_CANDIDATE_POOL=1000`;
- Pass 2 evidence: `EVIDENCE_WINDOW_CHARS=15000`;
- index/report shards: `INDEX_SHARD_SIZE=2000`, `REPORT_SHARD_SIZE=2000`;
- graph shards/monolith cutoff: `GRAPH_SHARD_SIZE=5000`, `GRAPH_MONOLITH_THRESHOLD=10000`;
- routine health maintenance: `olw health --batch-size 1000`.

The deterministic capacity gate completes with every graph node/edge shard at or below 5,000 records, every index/report shard at or below 2,000 records, FTS candidates at or below 100, and dirty-queue claims at or below the requested worker batch.
