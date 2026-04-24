# Documentation

Structured project documentation lives here so the repository root stays clean.

## Core docs

- [Architecture](architecture/ARCHITECTURE.md)
- [Product requirements](product/PRD.md)

## Operational UX

- `pipeline doctor [VAULT]` / `pipeline config-doctor [VAULT]` — first-run/config diagnostics with redacted config.
- `pipeline fixture VAULT` — deterministic example vault for demos and snapshot tests.
- `pipeline review-status [VAULT]` — pending review queue summary without side effects.
- `pipeline telemetry [VAULT] --json` — recent structured extraction/pipeline events.
- `pipeline release-check` — version/docs/release metadata hygiene.
- Commands that automation commonly consumes support `--json` (`stats`, `doctor`, `review-status`, `approve`, `reject`, `telemetry`, `fixture`, `release-check`).

## Graph semantics

Graph nodes are first-class notes from entries, concepts, MoCs, and sources. Edges are derived from wikilinks whose targets resolve to an existing note, then serialized to `06-Config/edges.tsv` with an edge type and description. Source notes are intentionally included so entry→source provenance links survive compile/stat/report flows.

## Release docs

- [Changelog](release/CHANGELOG.md)
- [Patch notes](release/PATCH_NOTES.md)
- [Release process](release/RELEASE.md)

## Quality history

- [Audits](audits/)
- [Code reviews](reviews/)
