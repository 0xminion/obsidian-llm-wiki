# PRD: Obsidian LLM Wiki v0.3.0

## Executive summary

Obsidian LLM Wiki is a local-first pipeline for converting raw research inputs into an Obsidian-native knowledge graph. It extracts source content, plans notes, creates structured wiki artifacts, maintains typed graph edges, and exposes diagnostics/migration commands for unattended operation.

The product contract is: **one command turns inbox content into a reviewable, linked, maintainable vault without letting LLMs write arbitrary files.**

## Users

1. Individual researchers maintaining a personal Obsidian PKM vault.
2. AI agents that need a deterministic ingestion interface.
3. Technical users who want source provenance, graph health, and release-grade verification rather than a loose pile of generated markdown.

## Problem statement

The system must:

1. Ingest web articles, X/Twitter URLs, YouTube videos, podcasts, PDFs, and pre-extracted markdown clippings.
2. Preserve raw source provenance while producing concise entry notes.
3. Maintain evergreen concept notes and topic Maps of Content.
4. Keep a typed graph (`edges.tsv`) aligned with actual Obsidian wikilinks.
5. Support bilingual English/Chinese notes without slugifying CJK titles into nonsense.
6. Give human operators review gates, diagnostics, migrations, and machine-readable status.
7. Be safe against path traversal, stale graph artifacts, SSRF-style fetches, and secret leaks.

## Pipeline architecture

```text
01-Raw / 02-Clippings
        ↓
Stage 1: Extract
        ↓
Stage 2: Plan
        ↓
Stage 3: Create
        ↓
Compile + lint + diagnostics
        ↓
04-Wiki + 06-Config artifacts
```

### Stage 1: Extract

- Deterministic Python source routing.
- Type-specific fallback chains.
- URL/content deduplication via SQLite store.
- Dead letter queue for failed extraction.
- SSRF-resistant validation and DNS-pinned curl fail-closed behavior.
- Secret headers passed through stdin config, not argv.
- JSONL telemetry with sensitive URL/query values redacted.

### Stage 2: Plan

- Dedup check against existing source notes.
- QMD MCP semantic concept search where available.
- Local keyword fallback if QMD is disabled or unreachable.
- Deterministic plan generation for obvious cases.
- Direct LLM planning only for uncertain sources.
- Optional human review queue.

### Stage 3: Create

- Template-based note creation.
- Bounded LLM insights for summary/core insight sections.
- Python-owned frontmatter, paths, wikilinks, collisions, MoC updates, and concept stubs.
- Safe note stems and resolved-path containment checks for every file write.
- Distinct entry/source stems to avoid ambiguous Obsidian links.
- Batch validation and no-stub rules.

### Compile pass

| Operation | Type | Product contract |
|---|---|---|
| Cross-link suggestions | Semantic | Add missing links only after validation. |
| Concept merge proposals | Semantic | Merge near-duplicates with reference rewrites. |
| MoC rebuild | Semantic + deterministic | Topic hubs reflect current related notes. |
| Wiki index rebuild | Deterministic | Index reflects entries, concepts, MoCs, sources. |
| Edge rebuild | Deterministic | `edges.tsv` reflects resolvable vault links and provenance. |
| Duplicate report | Deterministic | Report is rewritten even when duplicates disappear. |
| Semantic status | Structured result | Empty/failed LLM output is degraded/failure, not silent success. |

## Product features

### Ingestion

- URL inbox: `01-Raw/*.url`.
- Markdown clipping inbox: `02-Clippings/*.md`.
- Review mode: `pipeline ingest --review`, `pipeline approve`, `pipeline reject`, `pipeline review-status`.
- Resume mode: `pipeline ingest --resume`.

### Vault maintenance

- `pipeline compile` for graph/index/MoC refresh.
- `pipeline lint` and `pipeline validate` for health and output quality gates.
- `pipeline stats`, `pipeline reindex`, and `pipeline tags` for dashboards and derived artifacts.
- `pipeline telemetry` for recent redacted operational events.

### Diagnostics and repair readiness

- `pipeline doctor` / `pipeline config-doctor` validate installation/config with redacted output.
- `pipeline graph-doctor` reports unresolved links, stale edges, malformed edges, and duplicate stems.
- `pipeline migrate` applies idempotent schema/assets migrations and records schema version.
- `pipeline fixture --adversarial` creates edge-case golden corpus fixtures.
- `pipeline release-check` checks package/release metadata hygiene.

### Query

- `pipeline query --ask ... --fast` uses direct LLM calls for low-latency answers.
- Default query mode uses the configured Hermes/agent command for deeper tool-assisted Q&A.
- Empty successful agent stdout is treated as failure and does not archive the query.

## Data model

| Model/artifact | Purpose |
|---|---|
| `ExtractedSource` | URL, title, author, content, source type, hash. |
| `Manifest` | Stage 1 extraction output. |
| `Plan` / `Plans` | Stage 2 note creation plan. |
| `pending_reviews` | SQLite-backed human review queue. |
| `Edge` | Typed graph relationship. |
| `CompileResult` | Structured compile metrics including semantic status/degradation. |
| `schema-version.json` | Vault migration state, current schema version `1`. |

## Acceptance criteria

- [x] Handles URLs, YouTube, podcasts, PDFs, X/Twitter, web articles, and markdown clippings.
- [x] Produces Source, Entry, Concept, and MoC notes.
- [x] Keeps `07-WIP/` untouched.
- [x] Uses safe filename/path helpers and containment checks for untrusted names.
- [x] Review approval is plan-atomic and rolls back replace failures.
- [x] Network extraction validates hosts, pins curl DNS, and fails closed on unsafe pins.
- [x] Secret-bearing HTTP headers do not appear in subprocess argv.
- [x] QMD can be disabled explicitly with `USE_QMD_MCP=false`.
- [x] QMD embeddings are consumed by compile when available.
- [x] Compile exposes semantic degradation rather than silent zero-change success.
- [x] Graph diagnostics exist as `pipeline graph-doctor`.
- [x] Vault migrations exist as `pipeline migrate`, schema version `1`.
- [x] Adversarial golden corpus exists via `pipeline fixture --adversarial`.
- [x] Installed-wheel smoke verifies `init`, fixture generation, graph doctor, and migration.
- [x] Current source test suite: 852 passing tests.

## Non-goals

- Multi-user collaboration.
- Real-time sync service.
- Web UI.
- Letting an LLM/agent directly perform arbitrary vault writes in the default path.
- Treating vector search as the canonical store; Obsidian markdown and deterministic indexes remain the source of truth.

## Verification contract

Release readiness requires all of:

```bash
ruff check .
pyflakes pipeline tests
pytest -q
python3 -m pip wheel . -w /tmp/obsidian-llm-wiki-wheel --no-deps
python3 -m venv /tmp/obsidian-llm-wiki-venv
/tmp/obsidian-llm-wiki-venv/bin/pip install /tmp/obsidian-llm-wiki-wheel/*.whl
/tmp/obsidian-llm-wiki-venv/bin/pipeline init /tmp/obsidian-llm-wiki-vault
/tmp/obsidian-llm-wiki-venv/bin/pipeline fixture /tmp/obsidian-llm-wiki-vault --adversarial --overwrite --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline graph-doctor /tmp/obsidian-llm-wiki-vault --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline migrate /tmp/obsidian-llm-wiki-vault --yes --json
```
