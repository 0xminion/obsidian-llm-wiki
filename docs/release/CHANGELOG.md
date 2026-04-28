# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Graph diagnostics**: `pipeline graph-doctor [VAULT] --json` reports unresolved wikilinks, stale edges, malformed edge rows, and duplicate Obsidian stems across sources, entries, concepts, and MoCs.
- **Vault migration system**: `pipeline migrate [VAULT] --yes --json` applies idempotent schema/assets migrations and writes `06-Config/schema-version.json` (current schema version: `1`).
- **Adversarial golden corpus**: `pipeline fixture --adversarial --overwrite --json` creates edge-case fixtures for packaged CLI smoke tests and regressions.
- **Semantic status fields**: `CompileResult` now exposes `semantic_status` and `semantic_degraded_reason` so degraded/empty semantic LLM output is visible to automation.
- **Scalable semantic candidate blocking**: semantic cross-link and merge paths now block candidates by tags, title tokens, and bounded fallback windows before expensive similarity/LLM work.
- **Agentic skill installation guide** in the root README, pointing to `skills/obsidian-ingest.md` and Hermes profile installation commands.

### Fixed
- Hardened filename/path handling with strict safe note stems and resolved-path containment checks across vault writes, templates, review approval, and semantic link insertion.
- Review approval no longer trusts arbitrary `pending_reviews.file_path` rows; writes are collection-scoped and plan-atomic.
- Review approval rolls back already-replaced files if a later atomic replace fails.
- YouTube URL handling now validates real YouTube hostnames and passes canonical URLs to downstream tools.
- curl-based extraction now DNS-pins with `--resolve` and fails closed if a safe public pin cannot be established.
- Secret-bearing curl headers, including AssemblyAI authorization, are sent through stdin config instead of argv.
- Empty successful agent stdout in non-fast query mode is treated as failure and does not archive the query.
- QMD disable flag now honors `USE_QMD_MCP=false|0|no|off`.
- QMD result path conversion now handles vault-relative paths such as `04-Wiki/concepts/foo.md`.
- QMD embeddings are consumed by compile when available instead of disabling local embeddings and losing semantic signal.
- Duplicate reports are rewritten when duplicates disappear.
- Edge cache is invalidated after direct `edges.tsv` rewrites.
- MoC frontmatter/link generation is sanitized and idempotent.

### Verified
- `ruff check .`
- `pyflakes pipeline tests`
- `pytest -q` → 852 passed
- installed wheel smoke: `init`, adversarial fixture, `graph-doctor`, `migrate`

## [0.3.0] - 2026-04-27

### Added
- **Direct Semantic Compile Pass** — Replaced the 600-second Hermes subprocess fallback with Python-driven embedding similarity + LLM validation. Compile pass now completes in ~3 minutes instead of hanging for 10 minutes. Falls back to Hermes subprocess only on direct-LLM failure.
- **Module decomposition**: Split 3 monolithic files into focused packages with backward-compatible re-exports:
  - `cli.py` → `pipeline/cli/`
  - `lint.py` → `pipeline/lint/`
  - `compile.py` → `pipeline/compile/`
- **CircuitBreaker** (`pipeline/utils.py`): Thread-safe circuit breaker for LLM failure protection.
- **Structured logging with correlation IDs** (`pipeline/log.py`).
- **Language detection module** (`pipeline/language.py`).
- GitHub Actions CI for static checks, tests, wheel build, and installed CLI smoke test.
- Central note schema definitions shared by template generation and lint validation.
- Packaged vault assets under `pipeline/assets/` so wheel installs can initialize complete vault scaffolding.
- QMD query mode contract: `auto`, `vec`, and `lex` modes.
- Structured JSONL telemetry helper for machine-readable stage events.
- Release checklist documenting artifact-based verification.

### Fixed
- Exception narrowing across extraction/compile/config paths.
- `config.py` temporary extract directories use `tempfile.gettempdir()` + restricted permissions instead of ad hoc `/tmp` paths.
- CI/static-check failures after module decomposition.
- Entry template generation now matches validator-required sections for comparison and procedural notes.
- Batch validation checks correct generated file collections.
- OpenRouter HTTP-error test cleanup noise.

### Changed
- Architecture docs updated for package structure, packaged assets, CI, and QMD mode contract.

## [0.2.0] - 2026-04-23

### Added
- **Unified LLM Client** (`pipeline/llm_client.py`): Multi-provider abstraction supporting Ollama, OpenRouter, and Hermes.
- **Semantic Compile Operations**: embedding-based cross-linking, concept merging, and MoC rebuild.
- `--fast` flag for `pipeline query`.
- `run_qmd_query` accepts `concepts_dir` for non-default vault paths.
- `_replace_wikilink_in_dir` for concept merge cleanup.
- `raise_on_error` / `generate_or_raise()` for transparent LLM failure handling.

### Fixed
- Missing `math` import in semantic similarity.
- Semantic command parsing for filenames with spaces/CJK/special characters.
- Merge reference cleanup across entries, concepts, MoCs, and sources.
- Compile command locking.
- LLM empty/failure handling in semantic paths.
- MoC frontmatter parsing when body contains YAML delimiters.
- Embedding cache key collision from identical note text.

### Changed
- Compile pass moved from Hermes subprocess to direct LLM calls.
- README updated for unified LLM client and query fast path.

### Deprecated
- `pipeline/create/agent.py` remains for compatibility but is superseded by template mode.

## [0.1.0] - 2026-02-XX

### Added
- Initial release: 3-stage pipeline (Extract → Plan → Create) with Obsidian vault integration.
- CLI commands: `ingest`, `compile`, `lint`, `validate`, `query`, `reindex`, `stats`, `tags`.
- SQLite-backed deduplication, dead letter queue, and lint caching.
- Multi-language English/Chinese support.
- Semantic concept search via QMD.
- Automated wikilink extraction and typed edge generation.

[0.2.0]: https://github.com/0xminion/obsidian-llm-wiki/compare/v0.1.0...v0.2.0
