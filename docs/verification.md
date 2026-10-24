# Verification report and acceptance handoff

**Final local acceptance passed for the exercised Linux workflows.** The complete Python matrix, independent review closures, package/plugin checks and measured limits are recorded below. This report preserves earlier failures rather than presenting the audit as green from the start. It is not a security certification or proof of model correctness.

## Target and receipt locations

- Baseline pin: `d25cd80e3101252fad6aadd149831fa6ce9d505d`.
- Integration branch: `audit/incremental-integrity`. Earlier receipts used HEAD plus a staged diff because fixes were still uncommitted. The final production hash below and the delivery commit/manifest identify the delivered source.
- Evidence root for this audit host: `/home/linuxuser/.hermes/cache/scratch/` (abbreviated **SCRATCH** below). These are machine-local audit artifacts, not repository files or permanent public links. Archive them separately if retention is required.
- Synthetic tests/rehearsals use disposable vaults, not user data. Inference fixtures or HTTP MockTransport are explicitly distinguished from live service calls below.

## Baseline and intermediate trajectory

| Stage | Actual result | Receipt / interpretation |
| --- | --- | --- |
| Pinned baseline | **947 passed, 1 skipped in 31.33s** | `SCRATCH/wiki-audit-evidence/baseline-pytest.txt`; baseline only, not final acceptance. |
| Input-contract integration | **1,037 passed, 4 failed, 1 skipped** | `input-suite.txt` in the same evidence directory. |
| Acceptance integration | **1,305 passed, 16 failed** | `acceptance-suite.txt`; in-flight fixture/implementation mismatches were not hidden. |
| Pre-review suite | **1,371 passed, 6 failed** | `pre-review-suite.txt`. |
| Intermediate green candidate | **1,379 passed in 42.78s** | `integrated-green-candidate.txt`; subsequently challenged by independent A/B probes. **Not the final green claim.** |
| Later language/CLI integration | **1,380 passed, 5 failed** | `source-language-suite.txt`; includes old success-on-failure expectations and language expectations before reconciliation. |
| CLI agent's later whole configured suite | **1,426 passed, 1 failed** | `SCRATCH/wiki-audit-cli-full-suite.xml`; remaining metadata-only title regression was still red then. This does not describe later parent fixes. |

These are successive slices, **not counts to sum**. See the [issue ledger](audit-issues.md) for root causes and regression selectors.

## Independent verification: original failing snapshots

### A — real CLI/provider serialization, deterministic HTTP

`SCRATCH/wiki-verification-a/REPORT.md`, `snapshot-hashes.json`, `reviewed.diff`, per-run vaults/requests, and `results.json` record the tested intermediate source.

- Main end-to-end harness: **35 checks, 9 failed assertions**.
- Supplemental metadata/security: **14 checks, 2 failed assertions**, both Chinese identity divergence (A1).
- Minimal independent tests: **9 failed** on the frozen snapshot and the then-current live tree.
- Selected repository tests: **420 passed, 1 failed**. The existing refresh/review test immediately retried inside queue backoff; its failure was not proof of overwritten reviewed bytes.

A exercised actual Typer CLI, httpx request serialization, parser/validation, persistence, registry, and renderer. HTTP transport responses and DNS were deterministic; real sockets were forbidden. It did **not** test a paid/live LLM or live publisher access.

Reproduce the historical failure or replay current fixes deliberately:

```bash
cd /home/linuxuser/.hermes/cache/scratch/wiki-verification-a
PY=/home/linuxuser/projects/obsidian-llm-wiki-audit/.venv/bin/python
# Historical snapshot; failure is expected and retained as evidence:
VERIFY_REPO="$PWD/snapshot" "$PY" verification_a.py
# Current checkout; the harness forbids real sockets and creates fresh vaults:
env -i PATH=/usr/bin:/bin HOME="$PWD" TMPDIR="$PWD" \
  VERIFY_REPO=/home/linuxuser/projects/obsidian-llm-wiki-audit \
  "$PY" -m pytest -q -p no:cacheprovider --tb=short independent_tests.py
```

The parent replay regenerated the original A directory's aggregate `results.json` and `metadata-results.json`; those files now describe the replay, not the initial failure. The unchanged historical FAIL report, snapshot, `run.log`, and JUnit receipts preserve the initial findings. Later closing reviews copied harnesses into separate evidence directories. Do not overwrite original aggregate evidence when replaying without first copying/archive-labeling it. Changing `VERIFY_REPO` is essential: rerunning a frozen failure does not test a fix.

### B — real pipeline/render/SQLite, mocked inference

`SCRATCH/wiki-verification-b/REPORT.md` and **`final/`** identify the authoritative frozen target; earlier snapshots are retained but are not its final result.

- **22 independent probes: 12 passed, 10 failed**, no collection/runtime errors.
- **178 selected repository tests passed**; not the full suite.
- **40 actual generated Mermaid files parsed** with Mermaid **11.12.2**, and a deliberately malformed control failed. This establishes syntax, not SVG or Obsidian GUI rendering.
- **16 registries** passed SQLite integrity and foreign-key checks despite semantic-staleness failures. Structural database integrity is not freshness.

B mocked model/embedding inference and deliberately injected late-render failure; actual source loading, parsing/validation, merge, state, SQLite, Markdown, graph export, backups and cleanup ran with live sockets forbidden. It demonstrated the full-render symlink escape separately from the **direct-export** hostile manifest deletion. Source-symlink pre-read was a containment failure, not demonstrated model exfiltration.

Historical replay:

```bash
cd /home/linuxuser/.hermes/cache/scratch/wiki-verification-b
PYTHONDONTWRITEBYTECODE=1 /home/linuxuser/projects/obsidian-llm-wiki-audit/.venv/bin/python \
  -m pytest final/verify_b.py -q --tb=short -p no:cacheprovider \
  --basetemp="$PWD/final/rerun"
```

This uses B's frozen target; the integrating reviewer must bind a separate fresh replay to the final production source and retain its hash manifest. The frozen report itself must not be rewritten to pretend it passed.

## Follow-up scoped receipts (not final aggregate)

- CLI agent: **41 new CLI cases passed**; related CLI/source/ingest plus optional integration **84 passed, 1 skipped** (LiteParse CLI missing at that run). Five selected original A CLI probes **passed, 4 deselected** against then-current live source. Evidence: `SCRATCH/wiki-audit-cli.md`, `wiki-audit-cli-focused.xml`, `wiki-audit-cli-independent.xml`.
- Documentation consistency run: **221 passed in 13.13s**, 15 modules covering CLI outcomes/planning, definition/MoC identities, languages/parser remaps, explicit equivalence, per-chunk acceptance/task bounds, leases/recovery, surviving owner aggregates, containment, backlink work, disclaimers/skeletons and renderer B fixes. Real network blocked, no credentials, no user vaults. Evidence: `SCRATCH/wiki-audit-docs-rehearsal/{regressions.log,regressions.xml,regression-command.json,source-hashes.json,run_regressions.py}`. Production source hashes were unchanged during that run. This is a targeted repository run, **not** a new independent full A/B pass.
- Cross-source language/slug follow-up: `tests/new/test_source_quality_gate.py::test_bilingual_alias_keeps_canonical_slug_and_source_links` **1 passed in 0.24s** under the same offline guard; `SCRATCH/wiki-audit-docs-rehearsal/cross-source.xml`. Kept separate from the 221-test run.
- Parent-reported dependency slice: **29 dependency/PDF/extractor tests passed** after replacing deprecated `fitz` import with `pymupdf as fitz` to avoid stdout/NDJSON warning contamination. Final integrating receipt should name its exact log and source identity; do not add this count to 221.

## Synthetic filesystem pipeline benchmark receipts

Raw: `SCRATCH/wiki-audit-evidence/release-pipeline-{10,100,1000,5000}.json`. Harness: [`scripts/benchmark_pipeline.py`](../scripts/benchmark_pipeline.py); [commands and cost boundaries](scaling.md#reproduce-the-synthetic-pipeline-benchmark).

These runs include the final metadata-containment and shared-MoC pipeline fixes. The subsequent resume-only CLI fix does not change this direct-pipeline benchmark path. Each size ran in a separate process. Environment in JSON: Python **3.12.3**, SQLite **3.45.1**, Linux **6.8.0-139-generic**, x86_64/glibc 2.39. Times below include discovery; discovery is also recorded separately in raw JSON.

| Sources | Markdown files counted | Initial (s) | Unchanged (s) | One changed (s) | Peak RSS (Linux KiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 41 | 0.308 | 0.039 | 0.110 | 39,956 |
| 100 | 311 | 1.725 | 0.126 | 0.342 | 42,452 |
| 1,000 | 3,011 | 17.826 | 0.989 | 2.689 | 63,680 |
| 5,000 | 16,291 | 99.861 | 5.100 | 15.346 | 151,816 |

Fixture: small synthetic experiment sources, one concept per source, no explicit concept relations or MoCs, embeddings disabled. The Markdown count includes source/entry/concept/navigation/report pages outside `.llmwiki`, **not that many independent concepts**. Real discovery, compile orchestration, merge, registry/state, and filesystem/graph publication ran; `_synthesize_source` was replaced by deterministic structured output. **No extraction, model wire latency, or live synthesis quality was measured.**

At every size, unchanged work made 0 fixture synthesis calls and wrote 0 reported paths; a single-source edit made 1 fixture synthesis call and wrote 3 reported paths. This selective publication does **not** mean work is dependency-local: the changed-corpus path still loads/merges/fingerprints the full corpus (**O(N)**). Single runs on a shared host are not SLOs. No crash recovery, rich shared-topic graph, ANN recall, or desktop capacity is certified by these numbers.

## Separate query and production registry benchmarks

Raw: `SCRATCH/wiki-audit-evidence/release-query-scale.json`, `release-registry-scale.json`; original methodological report: `SCRATCH/wiki-audit-scale.md`. Harness: [`benchmarks/benchmark_scale_paths.py`](../benchmarks/benchmark_scale_paths.py).

| Entities/pages | Full graph read/parse (s) | Retrieve 10 (s) | Query peak RSS (KiB) | Cumulative registry upserts (s) | Registry common-term search (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.206 | 0.055 | 24,036 | 0.837 | 0.005 |
| 10,000 | 1.779 | 0.526 | 39,268 | 10.123 | 0.036 |
| 50,000 | 10.538 | 2.850 | 117,184 | 52.104 | 0.203 |

Query mode reads actual small Markdown notes with only one edge in the graph; **every page** is read even though 10 results are returned. Registry mode runs production `upsert_synthesis`, ownership and FTS for one concept/source, not direct SQL row loading and not a complete vault build. Its cumulative times above sum the recorded growth batches; raw batch timings remain available. Later sizes in each mode inherit the process high-water RSS, so those RSS values are not independent fresh-process measurements. No extraction, live model, embedding service, or Obsidian rendering was measured in either mode.

Reproduce separately, with an existing scratch directory:

```bash
python benchmarks/benchmark_scale_paths.py query --sizes 1000 10000 50000 \
  --scratch "$TMPDIR" --output "$TMPDIR/query-benchmark.json"
python benchmarks/benchmark_scale_paths.py registry --sizes 1000 10000 50000 \
  --scratch "$TMPDIR" --output "$TMPDIR/registry-benchmark.json"
```

**No 1-million-note capability claim is supported.** Shards, bounded prompts, improved local operations, and a 50k-page query fixture do not prove million-note end-to-end latency/memory, semantic quality, extraction throughput, or usable desktop navigation. The [proposed architecture](scaling.md#million-note-architecture-separate-archive-index-and-obsidian-working-set) is future work.

## Live service and dependency boundaries

- `SCRATCH/wiki-audit-evidence/live-embedding.json` records **real local Ollama** `qwen3-embedding:0.6b`: 3 synthetic strings, each 1,024 dimensions; English/Chinese cosine **0.7257936070752519**, unrelated cosine **0.11755076400883328**, elapsed **5.922216795996064s**. This demonstrates that local embedding request only, not translation equivalence, identity permission, broad cross-language recall, or synthesis quality.
- No chat model was installed in the local Ollama instance according to the integrating reviewer's discovery; **live synthesis quality was not tested**. A fixture-backed end-to-end path must not be labeled live LLM acceptance.
- A's extraction used real routing with deterministic HTTP responses. Pipeline/path benchmarks did not fetch sources. Live extraction/PDF/podcast/YouTube results, if added by the parent, must identify input, extractor/dependency, whether transport was real, content/provenance checks, and access/completeness limits; one success cannot certify other sites/formats.
- Dependency integration removed unused `youtube-transcript-api`, changed the stale calendar ceiling to `yt-dlp>=2026.10.21`, and regenerated `uv.lock`. Parent receipts `lock-update.txt` and `locked-sync.txt` record `uv lock --upgrade` / `uv sync --locked --extra dev --extra pdf --extra docx --extra liteparse --extra youtube`; parent verified installed yt-dlp **2026.10.21**. Use the checked-in exact lock for reproducibility, not an arbitrary future-year upper bound. CI configuration was **not changed**; ordinary floating pip resolution in CI remains a reproducibility boundary.

## Documentation rehearsal

`SCRATCH/wiki-audit-docs-rehearsal/rehearse.py` extracted and executed the actual Bash/Python snippets from [migration](migration.md) and [equivalence policy](data-model.md#human-confirmed-equivalences), changing only the sample vault path/environment to a newly created synthetic fixture. `sitecustomize.py` blocked sockets, the environment/HOME were isolated, and `.env` contained no credentials. Successful run directory: `migration-bnrjaqkl/`; `migration-latest.json` and per-step `.sh`/`.log` files retain commands and results.

Verified:

- Whole-vault snapshot and rehearsal copies: **7 files**, matching SHA-256 manifests.
- `build --dry-run`: **2 unchanged sources**, 0 new/changed/deleted, 0 content-only synthesis candidates; paths, content bytes and mtimes unchanged; **no registry created**.
- Explicit no-provider migration: schema **6**; **2 sources, 2 syntheses, 2 concepts, 0 MoCs**; integrity `ok`, **0** foreign-key violations; first import true, second false.
- Actual canonical-ID discovery, owner/definition listing, validated pair creation and reversed duplicate assertion: **1 unique pair**, no guessed IDs. The provider-calling build at the end of the policy section was deliberately **not run** in this offline snippet rehearsal; fixture pipeline tests separately cover application.
- Simulated upgrade on the synthetic original followed by documented whole-vault rollback: exact original **7-file** manifest restored; altered/upgraded state retained in `.before-recovery-*`.

Corrections: documentation's stale claim that build dry-run creates/migrates SQLite was removed. Explicit registry migration remains a separate write step; old artifact versions are never relabeled as current. Two initial scratch fixture setup mistakes (wrong section class name, then a string passed positionally as `points`) were corrected to keyword `BodySection(heading=..., prose=...)` and the entire rehearsal rerun on a **new** vault. The invalid fixture was skipped by legacy import, illustrating why migration markers/counts need inspection. The documented migration commands themselves required no executable change.

Not established: migration from every historical physical schema, cross-filesystem/power-loss durability, real-vault data preservation, provider-backed regeneration, or rollback while other writers are active.

## Final local acceptance

The final production snapshot contains 88 Python modules. SHA-256 of the sorted path-to-SHA256 JSON (`json.dumps(mapping, sort_keys=True).encode()`) is **`4bdd705f6b15973ddfbb736c41673a85e8ed4e1034bddba2772174f9436a64b1`**. `production-source-hashes.json` contains the individual hashes; the delivery manifest records the resulting commit. All runs below exercised this production code or the explicitly scoped, unchanged components identified by the independent reports.

### Complete Python matrix

Each environment was installed from `uv.lock` with `dev,pdf,docx,liteparse,youtube` extras. The real LiteParse PDF test was included, not skipped.

| Python line | Passed | Failed / errors / skipped | Seconds | Receipt |
| --- | ---: | --- | ---: | --- |
| 3.11 | 1,762 | 0 / 0 / 0 | 75.543 | `shipping-py311.xml` / `.txt` |
| 3.12 | 1,762 | 0 / 0 / 0 | 92.160 | `shipping-py312.xml` / `.txt` |
| 3.13 | 1,762 | 0 / 0 / 0 | 70.792 | `shipping-py313.xml` / `.txt` |

Command shape: `PATH="ENV/bin:$PATH" ENV/bin/python -m pytest tests/ -q --tb=short --basetemp=ISOLATED_SCRATCH --junitxml=RECEIPT`. Exact commands were executed from the checkout. Counts overlap across interpreters; they are not additional unique tests. Ruff for `obsidian_llm_wiki tests scripts benchmarks`, `git diff --check`, and `uv lock --check` passed.

An earlier 3.11 run exposed a genuine WAL-negotiation race despite SQLite's busy timeout. The fix retries only BUSY/LOCKED errors during WAL negotiation, with bounded attempts/deadline, and closes the connection on terminal failure. The independent 3.11 startup/error matrix and five fresh concurrent-migration reruns passed before this final complete matrix.

### Independent closure

- **A:** original workflow 35/35, metadata/security 14/14, minimal regressions 9/9, extended checks 25/25, and original final findings 2/2 passed. Closing review then found one earlier resume-reader bypass (CA-01); that finding was fixed rather than hidden.
- **CA-01 approval:** `SCRATCH/wiki-ca01-approval/REPORT.md` independently approves closure. Six unchanged original resume probes, 96 selected operations/ingest tests, 45 new public-API/caller-order/process assertions, and the original 35/14 workflows passed. Forty-one instrumented boundary/control executions observed zero outside access or provider/network calls. Only CLI/operation-store production files changed after the prior closing-A snapshot.
- **B:** `SCRATCH/wiki-closing-review-b/REPORT.md` approves its integrated scope. The 22 original adapted probes, 16 extensions, 3 destructive boundary probes, 4 independent arrival/deletion lifecycle cases, and 222 selected regressions passed. It independently checked source bindings, external sentinels and authoritative SQLite/page content. The later CA-01 change does not alter these pipeline/registry/render modules.
- B parsed **182 newly generated/retained Mermaid files** (62 distinct hashes) with Mermaid 11.12.2 and rejected a malformed control. This is syntax validation, not GUI/SVG acceptance.

Historical FAIL reports remain historical. Each closure names its tested source and scope; no report is a blanket security certification.

### Packaging, plugin and dependency receipts

- `uv build --wheel` succeeded. The final wheel installed into a separate environment outside the checkout. `wheel_smoke.py` exercised real discovery/registry/merge/render twice with deterministic inference: first compiled one source; the second skipped it, made no additional provider call and wrote no generated paths. Installed-package import path, SQLite checks and actual output are in `verified-wheel-smoke.json`. Installed `olw --help` also succeeded.
- Final wheel SHA-256: **`71f2170a7ebc2b8f670fed1574e90c185cafa9ff12df2d49cbe5e7314a641b46`**. This is an audited unreleased snapshot; it retains the project's 3.0.0 package version and is not a PyPI release.
- Plugin: **6 tests passed**, TypeScript check and production build passed on Node 26.12.1. An independent in-memory build was byte-identical to the 7,312-byte artifact. No native Obsidian GUI session or minimum-version desktop acceptance was performed.
- Python `pip-audit` reported **no known vulnerabilities** in the installed dependency snapshot (`final-pip-audit.json`). Uninstalled optional ASR dependencies are outside that receipt.
- npm initially reported the moderate moment advisory GHSA-4p3w-j4w9-5jqw through the development Obsidian API package. Updating/pinning Obsidian 1.14.4 and the lock resolved the report; `npm-audit-remediated.json` reports **zero known vulnerabilities**. Tests/typecheck/build were rerun after the update. Obsidian is externalized from the plugin bundle.
- Real public extraction of `https://www.python.org/doc/essays/blurb/` returned 1,693 characters through trafilatura; extracted-content SHA-256 matched provenance (`final-live-extraction.json`). No synthesis or durable extraction writes occurred. One public page is not a publisher/access-completeness guarantee.
- `test_audit_source_graph_rollback.py` explicitly injects a late source-graph failure and verifies exact restoration of previously rendered files; it passed within every final matrix run.

### Remaining limits and next priorities

The correctness and recovery workflows above pass locally. **The million-note architecture is not implemented or certified.** Changed builds still merge/fingerprint the corpus; query loads the full graph; metadata/cache-name preflight adds linear filesystem work. The final safety-hardened 5,000-source no-op measured **5.100 seconds**, not the earlier intermediate 0.275 seconds. Reduce repeated preflight safely, persist dependency-local work/query indexes, and introduce ID-based partitions/selective Obsidian materialization before making a million-note claim.

No real chat-model quality evaluation was available: local Ollama had embedding models only, and no paid endpoint was used. Exact quotes and structural gates do not prove semantic entailment. Native Windows/macOS and Obsidian GUI behavior remain outside execution evidence; the operation store still uses POSIX locking. Path checks do not eliminate hostile check/use races or hard-link attacks. SQLite, Markdown and caches do not form a whole-vault power-loss transaction. Optional outsourced/media extraction and disabled-by-default scholarly fallback retain the documented transport/provenance limits. CI configuration was not changed; floating CI dependency resolution remains distinct from the locked local matrix.
