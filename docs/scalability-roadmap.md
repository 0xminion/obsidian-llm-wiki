# Five-Year Scalability Roadmap

## Scope and conclusion

This document assesses a target vault of **5,000 sources, 50,000 concepts, and 20,000 MoCs**. The current implementation is suitable for a small personal vault but is not safe at that steady state: normal builds contain full-corpus prompt construction, pairwise similarity passes, and full-tree rendering.

The target must be treated as a v2 storage/indexing project, not a tuning exercise.

## Verified current constraints

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
