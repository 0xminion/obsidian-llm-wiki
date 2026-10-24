# Scaling: measured paths, current limits, and a proposed design

Active architecture reference. **There is no end-to-end million-note certification.** Historical 5k-source/50k-concept/20k-MoC targets and synthetic fixtures are not guarantees about extraction, LLM quality, latency, RAM, disk, or Obsidian responsiveness.

## Current cost model

| Path | What is bounded or avoided | Work still proportional to the corpus |
| --- | --- | --- |
| Unchanged CLI build | Reads metadata stubs rather than every source body; successful checkpoint skips synthesis loading/merge/render. | Directory enumeration/stat, source signature/index comparisons, and version checks. Not O(1), and not a generated-output integrity scan. |
| One changed source | Only changed/incompatible sources need inference. FTS/ANN candidate prompts and evidence windows have budgets. | Load/deserialize live syntheses, merge concepts/MoCs, traverse relationships, construct/fingerprint render candidates, and update whole-file state where applicable. **Changed builds are still O(corpus), not dependency-local.** |
| Generated file publication | Fingerprints select changed paths; timestamp-insensitive equality avoids gratuitous rewrites; touched paths are backed up. | Building the plan can be corpus-wide even when only a few files are written. Shard boundaries can broaden affected artifacts. |
| Registry point updates | Indexed source/entity/ownership/search-row and ANN-entity operations avoid particular full-table scans. | Migration, aggregate rebuilds, retained history, and some global reconciliation remain distinct workloads. SQL `LIMIT` alone does not prove bounded work. |
| Embeddings | Float32 storage, LSH candidates, bounded exact reranking, cached compatible vectors. | Initial/backfill/model-namespace work can cover the corpus. Candidate caps trade recall for cost; they are not a proof of semantic correctness. |
| Query | Returned/context pages can be limited. | The current wiki page/graph retrieval path can load the full corpus. PageRank is not replaced by the synthesis FTS/ANN path. |
| Health | Routine page batch size; explicit `--full` for exhaustive scan. | Index reconciliation and complete scans remain separate costs. Bounded returned findings do not imply bounded total work. |

Metadata containment also inspects immediate `.llmwiki/` names and legacy cache JSON paths during startup. This deliberately conservative preflight adds O(cache entries) filesystem work, including on no-op builds and registry opens. The final 5,000-source no-op measured 5.100 seconds, slower than the intermediate pre-hardening result; the earlier 0.275-second result is not a final-code claim. Before large batches, replace repeated complete cache-name preflight with equally safe indexed/per-access validation and measure it; do not remove containment guards to win a benchmark.

The current `sources/`, `entries/`, `concepts/`, and `mocs/` layout is primarily flat. Sharding derived indexes/graph JSON does not partition all note storage or eliminate full-corpus Python objects. Whole-file `state.json` and render-page hash metadata are additional scale considerations.

## Evidence levels must stay separate

1. **Unit/operation-count regression:** proves a specified invariant on a fixture—e.g. an indexed replacement avoids scanning unrelated FTS rows. It is not a filesystem or provider throughput test.
2. **Synthetic registry/ANN/graph fixture:** exercises selected tables/algorithms. Name actual inserted counts, vector dimension, graph density, and queries; do not label it an entire compiled vault.
3. **Synthetic filesystem pipeline:** [`scripts/benchmark_pipeline.py`](../scripts/benchmark_pipeline.py) creates real source Markdown, uses actual discovery/registry/merge/render, and measures initial, unchanged, and one-source-changed runs. Synthesis is a deterministic fixture; embeddings are disabled. It does **not** measure web extraction, real inference, or summary quality.
4. **External extraction/parser integration:** a real PDF parse or public page fetch establishes that input's extraction path, not all formats/sites or a bulk-ingestion SLA.
5. **Live representative corpus:** needed for semantic quality, realistic note/edge/MoC distributions, provider behavior, resource growth, and desktop usability. This remains a separate acceptance exercise.

Do not combine different test sizes into a fabricated end-to-end capacity claim. Record results with environment, commit/dirty-tree identity, exact command, source count, actual generated note count, content sizes, topology, optional services, wall time, peak RSS, disk/backup growth, and errors. The [verification report](verification.md) preserves baseline, intermediate failures, scoped tests, and actual benchmark receipts; its final aggregate section is reserved for the integrating reviewer. These intermediate measurements must not be promoted to final capacity certification.

### Reproduce the synthetic pipeline benchmark

From the repository with the Python environment activated:

```bash
export RESULTS="$(mktemp -d "${TMPDIR:-$HOME}/olw-benchmarks.XXXXXXXX")"
python scripts/benchmark_pipeline.py --sources 10 --output "$RESULTS/pipeline-10.json"
python scripts/benchmark_pipeline.py --sources 100 --output "$RESULTS/pipeline-100.json"
python scripts/benchmark_pipeline.py --sources 1000 --output "$RESULTS/pipeline-1000.json"
python scripts/benchmark_pipeline.py --sources 5000 --output "$RESULTS/pipeline-5000.json"
```

Each process creates a new temporary vault and leaves its path in the JSON. Keep runs in separate processes for meaningful per-process peak RSS. The RSS field is explicitly Linux KiB; the script uses Unix `resource` and is not a portable Windows harness. Check free disk before larger sizes; these commands create real files and backups. Do not run a million-source case by extrapolation or assume multiplying a small-run result predicts it.

## Proposed local architecture: bounded dirty/dependency batches

**Not implemented merely by the settings below; requires code, migrations, and acceptance evidence.**

### Identity-based file partitions

Use immutable IDs as storage keys, decoupled from editable display titles, aliases, and URLs. For example, a proposed layout could be `concepts/<id-prefix>/<id>.md` with title/aliases in frontmatter and an ID→path registry. Partition sources, entries, and MoCs similarly; keep human-facing discovery in indexes/search rather than one giant directory.

A migration must assign IDs to existing source and MoC records, resolve collisions, preserve old paths/redirects and all wikilinks, retain reviewed content, and be resumable. Do not just move current files: the current loader is nonrecursive and references are slug-based.

### SQLite as the local dependency and publication index

Keep local SQLite for a single-machine compiler, with indexed tables for:

- Immutable entity identity, revisions, source ownership, aliases, claims/evidence, and explicit relationships.
- Reverse dependencies: source→concept/entry; concept→dependent concepts/MoCs/views; entity→index/graph shard.
- Dirty work with payload generation, due time, lease generation/owner, retry reason, and bounded batch claims.
- Per-entity render input revision, produced content hash, current file signature, and human-ownership state.
- An outbox/journal that makes DB intent, atomic file replacement, completion, and restart reconciliation explicit.

A changed source should enqueue only its own work plus the transitive dependencies whose inputs actually changed. Process a bounded batch, commit progress, and resume. Revision-aware dependencies must remove obsolete ownership when a source changes/deletes; otherwise selective compilation retains stale claims. Keep expensive global audits/index rebuilds explicit and restartable rather than on every build.

Use keyset pagination, covering/leading-key indexes, FTS rowid identity, and entity-keyed vector deletion. Verify SQLite query plans and VM step growth; a small return value can still hide a large scan/sort. Avoid loading all IDs into Python just to select a batch. Reconcile external edits via an incremental file watcher/change journal plus an explicit periodic full audit; filesystem watchers alone are not a durable truth source.

### Stable output shards and bounded publication

Assign fixed/stable hash buckets or another measured partitioning scheme to indexes/graphs. Rebuild only dirty buckets. Bound both records and serialized bytes. A “2,000 records” page can still be huge if summaries are huge. Track cumulative backups and archives as well as live files; retention must be explicit and recoverability tested.

Preserve user-owned text separately or with a tested three-way merge contract. Do not adopt edited files simply because their path matches an entity. Publication must recheck the file revision observed when planning to avoid clobbering edits made during inference.

## Million-note architecture: separate archive, index, and Obsidian working set

At a million notes, the design target should **not** be “open every generated file in one desktop vault and hope.” Separate:

1. **Canonical archive:** partitioned immutable source/revision and compiled entity storage, with backups and verifiable provenance. It may remain on local disk initially; choose object storage or a service only when measured deployment needs justify it.
2. **Search/dependency index:** persistent lexical/vector/relationship indexes and bounded retrieval APIs. SQLite may suit a local workload; higher write concurrency or remote clients may justify a separate database/search service. No database choice alone certifies million-note performance.
3. **Selective Obsidian working set:** materialize a chosen project/topic/recent/review-needed subset, its required dependencies, and navigational links. Mark unmaterialized targets clearly or expose an explicit “fetch note” workflow rather than silently generating broken links.
4. **Human-edited overlay:** preserve edits and reviewed decisions across eviction/re-materialization. Define synchronization/conflict rules before automating working-set pruning.

The working-set size should be measured on the user's desktop, plugin set, sync backend, and storage. Do not claim a universal Obsidian file-count limit or responsiveness guarantee.

## Acceptance before a scale claim

- Compare no-op, one-source update, delete-last-owner, many-owner shared concepts, renames, and large batch changes at increasing corpus sizes.
- Measure **work**, not only output: source reads/stats, DB VM steps, rows deserialized, relationships traversed, prompt bytes, vector evaluations, planned/written/deleted paths, peak RSS, wall time, and disk growth.
- Exercise sparse/disconnected graphs and high-degree hubs; a friendly ring topology misses dangling-node and fanout costs.
- Crash after each persistence boundary; verify last-good output, dirty generations, queues, checkpoints, user edits, and restore consistency.
- Test same-title/different-definition concepts, multiple languages, many aliases, long filenames, duplicate URLs, source reverts, incomplete extraction, and corpus-wide version changes.
- Benchmark representative desktop navigation/search/sync with the actual working set. Separately evaluate summaries, unsupported claims, quotations, identity decisions, and relationship accuracy with human review.

Until these exist, describe individual measured paths and remaining costs—not “fully indexed builds,” “safe semantic auto-merge,” or “million-note ready.”
