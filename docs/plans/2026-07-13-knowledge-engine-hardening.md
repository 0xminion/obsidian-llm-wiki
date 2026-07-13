# Knowledge Engine Hardening Implementation Plan

> **For Hermes:** Execute in staged batches. Every code task adds targeted tests before integration; run `scripts/run_tests.sh` only if this repository provides it, otherwise activate `.venv` and run `python -m pytest tests/ -q` plus `ruff check obsidian_llm_wiki tests`.

**Goal:** Make the compiler provenance-rich, safe for scientific/document ingestion, graph-aware at query time, maintainable by humans, and operable through CLI JSON plus a thin local Obsidian bridge.

**Architecture:** Python remains the only extraction, synthesis, graph, maintenance, and rendering engine. New additive modules own provenance, safe downloads, graph retrieval, maintenance, sessions, and command output. The optional `obsidian-plugin/` package is a local UI adapter that executes `olw --json`; it must never duplicate compiler logic.

**Tech stack:** Python 3.12, Typer, httpx, PyMuPDF, optional LiteParse CLI, JSON state files, deterministic Markdown rendering; TypeScript/esbuild only for the thin Obsidian adapter.

---

## Acceptance contracts

1. A scientific landing page with an official same-publisher accessible full-text link chooses verified full text over an abstract that a generic parser can extract.
2. Every rendered source page persists immutable retrieval provenance: requested, resolved, extracted URLs; extraction chain; format/MIME; retrieval time; content SHA-256; bounded diagnostics.
3. No document path can exceed configured fetch byte limits, follow unbounded candidate links, or emit unbounded LiteParse output.
4. `--plan` is no-network; `--preview` extracts but writes nothing; `--json` emits one structured result per source.
5. Concepts do not render `## Related Concepts`; relation frontmatter and `## Cross-References / 关联图谱` remain.
6. Review protection, backups, maintenance fixes, contradiction records, source aliases, and merge revision state preserve human edits by default.
7. Query answers have deterministic retrieved-page citations and a lexical/PPR retrieval explanation.
8. Thin Obsidian plugin invokes the CLI and displays structured events only; it has no compiler implementation.

## Phase 1 — source correctness and document safety

### Task 1: Provenance model and source-page persistence

**Files:**
- Modify: `obsidian_llm_wiki/core/models.py`
- Modify: `obsidian_llm_wiki/render/obsidian.py`
- Modify: `obsidian_llm_wiki/ingest/sources.py`
- Test: `tests/new/test_provenance.py`

Add an immutable `SourceProvenance` dataclass with requested/resolved/extracted URLs, extraction chain, content type, format, retrieved timestamp, content hash, and bounded diagnostics. Add it to `SourceDoc` with a default factory so existing extractors remain source-compatible. Persist/reload it in source frontmatter and verify write → reload round trips exactly.

### Task 2: Safe fetch boundary and document router

**Files:**
- Create: `obsidian_llm_wiki/ingest/documents.py`
- Modify: `obsidian_llm_wiki/config.py`
- Modify: `obsidian_llm_wiki/ingest/liteparse.py`
- Modify: `obsidian_llm_wiki/ingest/extractors/pdf.py`
- Modify: `obsidian_llm_wiki/ingest/extractors/__init__.py`
- Test: `tests/new/test_documents.py`

Create one dispatcher for local paths, direct URLs, and discovered documents. Stream downloads with `MAX_DOCUMENT_BYTES`, validate MIME plus file signatures where supported, limit candidate links, retain final resolved URL, and bound LiteParse timeout/stdout/stderr. Dispatch PDF/DOC/DOCX/EPUB/PPT/PPTX/XLS/XLSX consistently.

### Task 3: Scientific candidate selection

**Files:**
- Modify: `obsidian_llm_wiki/ingest/web.py`
- Modify: `obsidian_llm_wiki/ingest/extractors/scientific.py`
- Test: `tests/new/test_scientific_selection.py`

Add same-publisher public-document preflight for known scientific pages. Rank official accessible HTML above official PDF above landing-page extraction, require a substantive-content threshold, and never bypass authentication/paywalls. Record selection decisions in provenance. Regression-test a successful abstract landing page with a valid full-text citation link.

### Task 4: Real LiteParse contract fixture

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/fixtures/minimal.pdf`
- Create: `tests/integration/test_liteparse_cli.py`
- Modify: CI workflow as discovered

Add a separately marked integration test that runs only when the `lit` executable is installed. CI installs the optional extra and exercises the real CLI against a harmless fixture. Unit tests remain fully mocked and hermetic.

## Phase 2 — human-safe maintenance

### Task 5: Reviewed pages and bounded backups

**Files:**
- Create: `obsidian_llm_wiki/core/review.py`
- Create: `obsidian_llm_wiki/core/backups.py`
- Modify: `obsidian_llm_wiki/render/obsidian.py`
- Test: `tests/new/test_review_protection.py`, `tests/new/test_backups.py`

Honor `reviewed: true`: preserve curated body content while allowing explicit metadata/provenance updates. Back up every automated rewrite to `.llmwiki/backups/`, retain a bounded count, and expose restore through the maintenance CLI.

### Task 6: Maintenance scan and fix command

**Files:**
- Create: `obsidian_llm_wiki/core/maintenance.py`
- Create: `obsidian_llm_wiki/cli/fix.py`
- Modify: `obsidian_llm_wiki/cli/health.py`
- Test: `tests/new/test_maintenance.py`

Convert health findings into typed, machine-readable findings. Implement `olw fix --dry-run` and explicit `--apply` for deterministic repairs only: broken relation targets, orphan/MoC assignment candidates, tag normalization, aliases, and empty generated stubs. Never auto-merge pages or overwrite reviewed content.

### Task 7: Contradiction and revision-aware merge records

**Files:**
- Create: `obsidian_llm_wiki/core/contradictions.py`
- Modify: `obsidian_llm_wiki/core/state.py`
- Modify: `obsidian_llm_wiki/core/pipeline.py`
- Test: `tests/new/test_contradictions.py`

Persist source revisions and contradiction records with `detected`, `review_ok`, `pending_fix`, `resolved`, and `suppressed` states. Surface them in health/fix output. Do not ask the LLM to silently resolve factual conflicts.

### Task 8: Aliases, schema policy, and adaptive granularity

**Files:**
- Create: `obsidian_llm_wiki/core/schema.py`
- Modify: `obsidian_llm_wiki/ingest/sources.py`
- Modify: `obsidian_llm_wiki/synth/prompts.py`
- Modify: `obsidian_llm_wiki/config.py`
- Test: `tests/new/test_schema.py`, `tests/new/test_granularity.py`

Propagate source aliases/tags. Support an editable schema file scoped per task. Select synthesis granularity from source type and length with explicit thresholds, while preserving user overrides.

## Phase 3 — query engine

### Task 9: Graph index and hybrid retrieval

**Files:**
- Create: `obsidian_llm_wiki/query/graph.py`
- Create: `obsidian_llm_wiki/query/retrieval.py`
- Modify: `obsidian_llm_wiki/cli/query.py`
- Test: `tests/new/test_retrieval.py`

Build a deterministic wiki-link/relations graph. Implement lexical retrieval, seeded PageRank for sparse graphs, and graph-first PageRank once graph maturity thresholds are met. Return a structured retrieval trace.

### Task 10: Grounded citations, snippets, profiles, and sessions

**Files:**
- Create: `obsidian_llm_wiki/query/context.py`
- Create: `obsidian_llm_wiki/query/sessions.py`
- Modify: `obsidian_llm_wiki/cli/query.py`
- Modify: `obsidian_llm_wiki/config.py`
- Test: `tests/new/test_query_grounding.py`, `tests/new/test_query_sessions.py`

Extract compact deterministic page snippets before prompting. Enforce citations to retrieved source paths, store query sessions with provenance, allow query-only profiles/instructions, and provide an explicit save-answer workflow.

## Phase 4 — operational interfaces

### Task 11: Per-task models and provider checks

**Files:**
- Modify: `obsidian_llm_wiki/config.py`
- Modify: `obsidian_llm_wiki/providers/llm.py`
- Create: `obsidian_llm_wiki/cli/providers.py`
- Test: `tests/new/test_task_models.py`, `tests/new/test_provider_preflight.py`

Add model overrides for ingest, maintenance, and query; preserve old unified configuration. Add provider preflight/model-list probes and structured URL/auth/rate-limit diagnostics.

### Task 12: Extract-only UX, structured run history, and cancellation

**Files:**
- Modify: `obsidian_llm_wiki/cli/ingest.py`
- Modify: `obsidian_llm_wiki/core/metrics.py`
- Create: `obsidian_llm_wiki/core/operations.py`
- Test: `tests/new/test_ingest_modes.py`, `tests/new/test_operations.py`

Implement `--plan`, real `--preview`, `--json`, durable source-level operation records, retry states, and cooperative cancellation. Keep old options compatible where possible.

### Task 13: Thin Obsidian bridge

**Files:**
- Create: `obsidian-plugin/package.json`
- Create: `obsidian-plugin/manifest.json`
- Create: `obsidian-plugin/src/main.ts`
- Create: `obsidian-plugin/src/cli-client.ts`
- Create: `obsidian-plugin/src/types.ts`
- Create: `obsidian-plugin/tests/cli-client.test.ts`

Expose command-palette actions for ingest, preview, health, fix, and query. Spawn/configure `olw`, stream JSON events, support cancellation, and show result/history panels. No extraction, graph, rendering, or LLM algorithm belongs in this package.

## Phase 5 — rendering contract and compatibility

### Task 14: Remove duplicate Concept relationship section

**Files:**
- Modify: `obsidian_llm_wiki/render/obsidian.py`
- Modify: render/golden tests identified by search

Remove body-level `## Related Concepts`. Keep `relations:` frontmatter and the typed cross-reference diagram. Update golden output and migration-safe health rules so existing older pages remain readable.

## Phase 6 — final quality gates

### Task 15: End-to-end fixtures and review loop

**Files:**
- Create or modify: focused integration tests and documentation as proved necessary

Run synthetic source → extraction → provenance persistence → synthesis stub → render → health/fix → query smoke paths. Run linter and full test suite. Launch independent agents for spec/integration, bug/regression, and security/data-integrity review; remediate every confirmed high/medium finding; run a post-fix review. Commit only after fresh verification.

## Explicit non-goals

- No web auth/paywall challenge bypass.
- No copying green-dalii source code.
- No second TypeScript compiler.
- No automatic destructive merge/contradiction resolution.
- No telemetry or external data collection.
