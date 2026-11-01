# Query currentness contract

Ordinary `olw query` is **captured-generation Q&A**, not a claim that an immutable
publication is the latest state of continuously editable human inputs.

## Admission

- Resolve and validate one committed immutable generation. Never inspect staging
  as evidence, migrate a registry, rewrite receipts, or repair damaged history.
- `core.publication_inputs.input_freshness(vault_root, manifest)` supplies the
  shared safe input mapping/observation. Missing inspection is `unverifiable`
  and fails closed. The query lane does not guess how to rebase absolute paths.
- Require matching source inventory/revisions and policy input state before
  retrieval, immediately before provider use, and after response preparation
  before accepting the result for session/save/output. A changed generation or
  drifted/unverifiable input returns exit 1, empty answer/citations and a reason.
- These are observations, **not an atomic latest-at-commit guarantee**. Editors
  can still change human inputs after observation. Successful output, saved
  answers and sessions carry an application-added captured-generation qualifier.
- Legacy Markdown without a captured publication is not eligible evidence.
  Build it first. There is no implicit historical-query mode.

## Lifecycle data

`WikiPage` retains `orphaned`, `orphaned_reason`, `publication_status`,
`status_reasons` and per-source `source_statuses`. A read-only adapter derives
source/owner status from the committed semantic publication plan and verifies
approved review receipts against captured source bytes and current reviewer
configuration. A resealed unsupported registry schema is refused.

`eligible_graph` filters *before* ranking and `max_results`. `build_context`
independently refuses non-eligible pages and carries lifecycle/source status
outside the bounded prose snippet. Raw graph inspection retains historical
pages for diagnostics; direct consumers must apply `eligible_graph` before
retrieval and must independently observe inputs before accepting an answer.

Orphaned, failed-retained, unchecked, unadmitted, unbound and unmapped pages are
excluded. A mixed-owner page is wholly excluded when any owner's contribution
is non-admitted; a separate healthy source/concept can still answer. Generated
navigation/aggregate pages without a verified source ownership mapping are not
query evidence. No history is deleted. Source file text is not inferred from
warning prose, so a short snippet cannot discard the authoritative status.

JSON has a top-level `freshness` result and the same final status in
`retrieval_trace.freshness`; `retrieval_trace.pages` records per-page eligibility
and reasons. Failed requests never store a successful session. Saved-answer
collisions preserve the human winner and do not create a new session.

Prior session answers are reused only for the same generation and an eligible
retrieved evidence subset. Otherwise the old answer is excluded from the
provider messages; its history is not deleted merely because a query refused.

## Test integration

The shared helper is integrated. Run ordinary pytest **without** an input adapter:

```
python -m pytest tests/new/test_query_currentness.py tests/new/test_query_currentness_lifecycle.py tests/new/test_integrated_input_inventory.py
```

The former explicit `query_input_test_adapter.py` remains as historical test-lane
support, not production verification; it installs nothing when the real helper
exists. Fresh-process and parent integration gates use the real shared helper.

Source inventory checks include newly added source files. Capture records even
empty discovered source files, so an empty file can later acquire content without
escaping revision checks. Generated source navigation files are excluded using
the same inventory rules as the source loader.

API-only captured bodies without live source files remain queryable as snapshots.
Their final JSON freshness status is `captured_only`, with `unobserved_sources`,
and an application-added answer qualifier explicitly disclaims live/upstream
freshness. They are not labeled `observed_current` or reused as observed-current
session evidence. Existing observable file drift still causes refusal.

Scripted providers prove lifecycle plumbing, not semantic answer accuracy.
New tests exercise fresh queries/no-op, source add/edit/delete without build,
policy edits, failed review/backoff, source deletion/orphans, one-result short
snippets, mixed-owner staleness, generation changes during query, late input
edits, session reinjection, unknown schema/corruption and answer collisions.
