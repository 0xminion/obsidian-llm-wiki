# Documentation

Structured project documentation lives here so the repository root stays clean.

## Current system docs

- [Architecture](architecture/ARCHITECTURE.md) — data flow, components, safety boundaries, graph/index semantics, QMD behavior.
- [Product requirements](product/PRD.md) — current product contract, users, feature set, acceptance criteria, non-goals.

## Operational commands

| Command | Purpose |
|---|---|
| `pipeline doctor [VAULT] --json` | First-run/config diagnostics with redacted settings. |
| `pipeline config-doctor [VAULT] --json` | Config-focused alias for doctor. |
| `pipeline graph-doctor [VAULT] --json` | Graph integrity: unresolved links, stale edges, malformed edges, duplicate stems. |
| `pipeline migrate [VAULT] --yes --json` | Idempotent schema/assets migration; writes `06-Config/schema-version.json`. |
| `pipeline fixture VAULT --adversarial --overwrite --json` | Golden adversarial corpus for smoke/regression tests. |
| `pipeline telemetry [VAULT] --json` | Recent redacted structured pipeline events. |
| `pipeline release-check --json` | Version/docs/release metadata hygiene. |

Automation-facing commands should emit one clean JSON object when `--json` is provided. Do not interleave human prose with machine output.

## Graph semantics

Graph nodes are first-class notes from `sources`, `entries`, `concepts`, and `mocs`. Edges are derived from resolvable wikilinks and serialized to `06-Config/edges.tsv` as:

```text
source<TAB>target<TAB>type<TAB>description
```

Source notes are intentionally included so entry→source provenance survives compile, stats, lint, and graph-doctor flows.

## Safety model

- Titles, LLM filename suggestions, MoC names, review DB rows, aliases, and source URLs are untrusted input.
- Note paths must be produced through the safe filename/path helpers and checked with resolved-path containment.
- Review approval validates a full plan before final writes, writes temp files first, and rolls back already-replaced files if a later replace fails.
- Network extraction rejects unsafe URL targets, pins curl DNS with `--resolve`, and fails closed if a public pin cannot be established.
- Secrets must not appear in argv, telemetry, docs, logs, or review summaries.

## Release docs

- [Changelog](release/CHANGELOG.md)
- [Patch notes](release/PATCH_NOTES.md)
- [Release process](release/RELEASE.md)

## Quality history

- [Audits](audits/)
- [Code reviews](reviews/)

## Agent skill

The repository skill is [`../skills/obsidian-ingest.md`](../skills/obsidian-ingest.md). The root README contains the Hermes profile installation commands.
