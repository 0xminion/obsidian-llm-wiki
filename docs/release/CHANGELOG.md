# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.3.0] - 2026-04-27

### Added
- **Module decomposition**: Split 3 monolithic files into focused packages with backward-compatible re-exports:
  - `cli.py` (1,252 lines) → `pipeline/cli/` (6 modules: ingest, compile_cmd, review_cmd, quality, manage, _helpers)
  - `lint.py` (1,207 lines) → `pipeline/lint/` (4 modules: checks, fixes, models, runner)
  - `compile.py` (1,580 lines) → `pipeline/compile/` (4 modules: core, semantic, structural, watch)
- **CircuitBreaker** (`pipeline/utils.py`): Thread-safe circuit breaker for LLM failure protection. Wired into compile agent, insight pre-generation, and batch filename generation.
- **Structured logging with correlation IDs** (`pipeline/log.py`): `batch_id`, `source_hash`, `stage` propagated via `contextvars`. Includes `stage_timer()` context manager for automatic stage timing.
- **Language detection module** (`pipeline/language.py`): Extracted from inline heuristics across plan/create modules.
- **74 new tests** (total: 820):
  - 52 adversarial parser tests (`tests/test_adversarial_parsers.py`)
  - 22 enrich coverage tests (`tests/test_enrich.py`)
  - 5 E2E smoke tests in `tests/test_integration.py`
- **Shared utility extraction**: `frontmatter_list_items()` deduplicated from compile modules into `pipeline/utils.py`.
- GitHub Actions CI for static checks, tests, wheel build, and installed CLI smoke test.
- Central note schema definitions shared by template generation and lint validation.
- Packaged vault assets under `pipeline/assets/` so wheel installs can initialize complete vault scaffolding.
- QMD query mode contract: `auto`, `vec`, and `lex` modes.
- Structured JSONL telemetry helper for machine-readable stage events.
- Release checklist documenting artifact-based verification.

### Fixed
- **Exception narrowing** across 11 files: replaced all bare `except Exception` with specific types (`ConnectionError`, `TimeoutError`, `OSError`, `ValueError`, `json.JSONDecodeError`). Prevents silent suppression of programming errors.
- **Security**: `config.py` `resolved_extract_dir` now uses `tempfile.gettempdir()` + `mkdir(mode=0o700)` instead of hardcoded `/tmp` paths.
- **CI**: `tests.yml` syntax check updated for new package paths (was referencing deleted `pipeline/compile.py`).
- **Ruff**: Fixed 14 lint errors (unused imports, unused variables, ambiguous variable names) introduced during decomposition.
- Entry template generation now matches validator-required sections for comparison and procedural notes.
- Batch validation checks new concept files in `04-Wiki/concepts/` instead of `04-Wiki/entries/`.
- OpenRouter HTTP-error test now uses a file-like body, eliminating unraisable cleanup noise.
- Static-check noise from unused test imports.

### Changed
- **Architecture docs** updated to reflect new package structure, test count (820), and line count (~15,800).
- **README** architecture tree updated to match actual package layout.

## [0.2.0] - 2026-04-23

### Added
- **Unified LLM Client** (`pipeline/llm_client.py`): Multi-provider abstraction supporting Ollama, OpenRouter, and Hermes. Configurable via `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_TIMEOUT`. Replaces scattered provider-specific code.
- **Semantic Compile Operations** (`pipeline/compile.py`): New compile pass with embedding-based cross-linking, concept merging, and MoC rebuild — replaces the previous Hermes-subprocess-based compile agent.
- **19 new unit tests** (`tests/test_compile_semantic.py`) covering cross-link generation, concept merging, MoC rebuild, embedding batching, and frontmatter preservation.
- **`--fast` flag** for `pipeline query`: Direct LLM Q&A without Hermes agent overhead (sub-5s vs 40-60s).
- **`run_qmd_query` now accepts `concepts_dir` parameter**: Removes hardcoded vault path, enables non-default vault paths to function correctly.
- **`_replace_wikilink_in_dir` helper**: Centralized reference rewriting for concept merge cleanup across entries, concepts, MoCs, and sources.
- **Edge validation in `_merge_concepts`**: `_remove_concept_from_edges()` removes stale edges after merge.
- **`raise_on_error` parameter** in `LLMClient.generate()`: Opt-in exception raising with `LLMGenerationError` for better error transparency.

### Fixed
- **H1 (CRITICAL)** — `pipeline/compile.py` now `import math` — previously crashed with `NameError` during `NoteIndex.similarity()` on every semantic compile.
- **H2/H3** — Regex for `LINK`/`MERGE` semantic commands now uses `(.+?)\s*\|\s*(.+?)` instead of `\S+`, correctly parsing filenames with spaces, Chinese characters, and special characters.
- **H4** — `_merge_concepts` now performs reference cleanup across all directories (`entries_dir`, `concepts_dir`, `mocs_dir`, `sources_dir`) and removes stale edges from `edges.tsv`.
- **M1+M6** — `compile_pass` CLI command now acquires `PipelineLock`, preventing concurrent access corruption across `ingest` and `compile`.
- **M3** — `LLMClient.generate()` returns empty string on failure like before, but `generate_or_raise()` (used by semantic compile) throws `LLMGenerationError` with details instead of silently swallowing errors.
- **M4** — `_run_semantic_compile` now correctly reports `agent_succeeded = False` when any semantic operation returns zero changes, preventing false-positive success output.
- **M5** — MoC frontmatter regex now uses `partition('\n---\n')` to only split on the document's first yaml-delimiter occurrence, preventing corruption when YAML values contain `---`.
- **M7** — `NoteIndex.embed_all()` now deduplicates batch data with `stable_hash()` instead of raw `text → vector` dict keys, preventing silent collision/overwrites when two notes share identical title+preview.

### Changed
- **Compile pass architecture**: Moved from Hermes subprocess agent to direct LLM calls (3× faster, no shell quoting issues, deterministic timeouts).
- **LLM client error handling**: `LLMClient.generate()` logs errors but returns `""`. Callers that need failure detection use `generate_or_raise()`.
- **README**: Updated to reflect unified LLM client configuration (`LLM_PROVIDER`), `--fast` query flag, new compile architecture, and test count (637).

### Deprecated
- `pipeline/create/agent.py` — Hermes subprocess agent for creation stage. Path still tested but superseded by `templates.py` with direct LLM calls. Will be removed in 0.3.0.
- `generate_entry_insights_legacy()` in `pipeline/create/templates.py` — same rationale.

### Removed
- None (dead code deferred to 0.3.0 for backward compatibility).

## [0.1.0] - 2026-02-XX

### Added
- Initial release: 3-stage pipeline (Extract → Plan → Create) with Obsidian vault integration.
- CLI commands: `ingest`, `compile`, `lint`, `validate`, `query`, `reindex`, `stats`, `tags`.
- SQLite-backed deduplication, dead letter queue, and lint caching.
- Multi-language support: English and Chinese content native.
- Semantic concept search via `qmd` with Qwen3 embedding model.
- Automated wikilink extraction and typed edge generation.

[0.2.0]: https://github.com/0xminion/obsidian-llm-wiki/compare/v0.1.0...v0.2.0
