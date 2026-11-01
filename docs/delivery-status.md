# Delivery status and validation limits

## Integration scope

This update integrates the tested immutable-publication and grounding pipeline, conservative document/podcast acquisition, reader-currentness checks, regression tests, and operating documentation. Integration into the repository is explicitly requested; **a merge is not a claim that the live semantic acceptance gates passed**.

The compiler keeps provenance, exact citation bytes, aliases, source-language content, human annotations, existing source files and historical publications. Required grounding review remains fail-closed. No acceptance threshold was weakened, no failed response was repaired into a pass, and no production vault was migrated as part of delivery.

Start with the [repository quick start](../README.md), [migration guide](migration.md), [immutable-publication contract](immutable-publication.md), and [bounded attribution context v4](fragment-context-v4.md).

## Verified engineering results

The frozen 443-file application snapshot delivered before this integration had:

| Check | Actual result |
|---|---|
| Complete local Python suite | 3,680 passed; 0 failures/errors/skips; exit 0 |
| Context-family regressions | 227 passed |
| Provider/component adapter checks | 57 passed |
| Fresh wheel/noneditable installation | 113 production modules imported and matched the source/wheel bytes |
| Installed CLI command tree | 15 help commands passed |
| Packaging acceptance | 46 unique tests passed; plugin typecheck/build checked |
| Delivered-source CLI smoke | Help, build/ingest help, dry-run and ingest plan passed; English/Chinese source fixture bytes preserved; no model/network calls |

These counts describe their exercised snapshots, not a new CI matrix result. Integration changes only documentation, one behavior-equivalent test-dictionary literal for Ruff, and a trailing blank test-file line; application source bytes remain identical to that tested snapshot. Current GitHub Actions results must be read for the current PR/commit rather than inferred from historical runs.

## Final live comparison: blocked, not passed

One final subscription-backed comparison was authorized with the same frozen inputs and requested model, a maximum of 31 native attempts, two preflights, 29 semantic requests, zero retries and no fallback.

Actual outcome:

- Both preflights passed.
- The first semantic request stopped the run: **3 native attempts total**, process exit **1**, stop reason `http_error`.
- The native capture contains a completed response, but the local adapter returned HTTP **502** with `{"message":"Codex upstream request failed","type":"upstream_error","upstream_status":null}`.
- **0 of 29 semantic results were successfully delivered.** The other 28 requests were not submitted. Missing results receive no correctness credit; this is not 29 demonstrated wrong model verdicts.
- The deeper SDK/transport cause is not established from the retained exception/timing metadata. A captured native-only response was not substituted for successful downstream delivery.
- No retry, fallback, resampling, account/model switch, shared-service change or credential reset followed the failure. The owned server stopped, its temporary bearer was removed, and source/input preservation and cleanup were checked.

Task-local capture/controller fixtures were separately exercised with 52 and 18 passing controls respectively. These mocked protocol checks are not semantic predictions and do not turn the failed real run into a pass. Experiment controllers and raw captures are intentionally outside the application repository; no credentials or private evaluation bodies/maps are published here.

## Remaining limits

- Complete live semantic, heldout and eight-source publication/coverage acceptance remain unestablished.
- One original generic historical Chinese fixture body is missing; that specific historical replay remains incomplete. It was not replaced with guessed content. Separate current multilingual controls and actual source-bound historical-v3 reconstruction do not erase that missing historical denominator.
- A prior stopped experiment recorded an unattributed auth-store metadata change; this was not called credential-byte preservation. The final experiment's recorded stat components matched, with auth-store writes forbidden.
- Native Obsidian GUI operation, physical power-loss, hostile filesystem races and current million-note capacity are not certified. Changed builds still perform corpus-wide work; see [scaling](scaling.md).
- No package release, tag, production-vault migration or operational service deployment is implied by this merge.

Earlier verification reports are retained as historical scoped evidence. Their headings saying “final acceptance” apply to those older snapshots, not to the unresolved live gates documented here.
