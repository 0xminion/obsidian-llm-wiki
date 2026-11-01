# Immutable application callers — isolated candidate integration

This supplements (and supersedes the caller limitations in) `immutable-publication.md`. It is not approval for a production migration or a claim that the old full suite is green.

## Public caller routing

| Caller | Read/write boundary |
|---|---|
| `build`, `core.pipeline.run_pipeline` | Full human-source inventory, then existing semantic pipeline in an exclusively created private workspace. No writable legacy/published registry for inventory. |
| `ingest`, `--resume` | Mutable registry, operation JSON, resume markers, leases, failures and lock under `04-Wiki/OLW-Publications/operations/`. Complete new source files install with no-replace semantics. Publication uses the same private build pipeline. |
| `recompile`, `recompile_single_source` | Validate source filename and full corpus, force only the requested source, publish a complete generation. It is not a registry-only mutation. |
| `fix --dry-run` | Scan the validated current projection, or legacy pages if no publication exists; no writes. |
| `fix --apply` | Rebuild a complete new projection through the existing semantic/review pipeline. Never invoke the legacy in-place patcher. Existing generations and legacy pages remain unchanged. |
| `health` | Full read-only scan; registry source-hash/render-ownership inspection deserializes a guarded closed SQLite byte snapshot into memory. No writable SQL, maintenance cursor or checkpoint. |
| `validate`, `build --dry-run`, `current`, graph/query | Resolve current before reading. Corrupt generations refuse, not repair. An operational-only namespace is not a pending generation; an actual staged/installed first attempt is. |
| `query --save-answer` | Only new files under `05-Queries/`; complete temporary bytes are linked with no overwrite. Sessions remain in `OLW-Publications/query-state/`, never a generation or the human legacy metadata directory. Query profile policy stays a read-only human input. |
| `metrics`, `load_metrics`, `load_all_metrics` | Resolve the validated current snapshot (legacy fallback is read-only). |
| `identity-migration --dry-run` | Resolve a valid current immutable generation before inventory. Keep the legacy schema/WAL/lock/constructor guards. Verify installed consumers rather than interpreting archived stage paths as current paths; compare current human source/policy hashes before certifying continuity. No bounded model-call estimate is invented. |

`core.application` centralizes read resolution, source inventory, operational-state location, read-only registry inspection and exclusive source installation. Low-level legacy rendering/registry primitives are still internal writable implementation tools; applications must not pass a published generation as a writable workspace.

## Supported domain writers and raw primitives

`orphan_page` and `mark_orphaned_concepts` refuse attempts to update legacy or installed pages in a vault with an immutable publication namespace. Use the normal build pipeline after removing the source: it marks orphaned concepts in a fresh private staging workspace and publishes a new generation. `MetricsCollector.save()` likewise refuses immutable-vault and installed-generation destinations before opening a registry or writing metrics; use the build pipeline for publication metrics. It does not silently redirect direct saves. These domain APIs retain legacy behavior in a non-immutable vault, and the pipeline's private workspace remains writable. Namespace ownership collisions and symlink paths are refused, not adopted.

These are API boundary checks, not operating-system immutability or a security sandbox. Raw filesystem writes and internal rendering, registry, cache, state, backup and atomic-write primitives can still mutate paths supplied by a caller; they are not supported public publication APIs. Path checks do not prevent hostile concurrent ancestor replacement. Direct writers must not target installed generations or human metadata.

### Domain guard contract and ordering

* With no `04-Wiki/OLW-Publications` namespace, both domain writers keep their existing mutable legacy behavior. Human legacy preservation by this guard applies once that namespace exists; it is not a universal prohibition on modifying all legacy notes.
* An existing namespace must have the exact publisher `owner.json` value. Missing, malformed, unowned or symlinked ownership evidence refuses before page/registry writes. This also covers an operations-only or pending namespace without `current.json`.
* The writable exception is the private layout `04-Wiki/OLW-Publications/.staging/<work>/04-Wiki/...` under an owned namespace. Every enclosing `04-Wiki` boundary is checked, so nesting a new wiki beneath an installed generation does not authorize it. This is layout/ownership validation, not an unforgeable capability or authentication check.
* `orphan_page` checks before reading page contents, backing up, starting a render transaction or writing. `mark_orphaned_concepts` inherits the check through each actual `orphan_page` call. Unsafe identities/paths that already return `False` retain that behavior; immutable destinations raise `RuntimeError` with build-pipeline guidance.
* `MetricsCollector` validates the expanded lexical input (environment variables and `~` included) before `Config.vault` can resolve away symlinks. `save()` rechecks that spelling and both output destinations before `VaultRegistry` initialization. The namespace path is checked before `exists()` or other following metadata access. A collector created before publication or before an ancestor replacement is revalidated, not grandfathered in.
* Refusal leaves original human files and installed generations at their paths with their bytes, inode and modes intact. Regression snapshots also compare mtime/ctime, uid/gid and available xattrs. Read-induced atime and parent-directory timestamps are not covered. Hardlink alias attacks, namespace forgery/removal, raw primitive calls and check/use races are not claimed prevented.

Actual caller chain: `run_pipeline` → `publish_pipeline` creates an exclusive private workspace → `_run_pipeline_workspace` calls `orphan_page` for deleted owners and calls `MetricsCollector.save()` in its `finally` block. Tests wrap these real calls (rather than replacing them), including successful build, delete, no-op and injected render failure. Existing immutable-caller controls exercise public build, ingest/resume, fix/recompile and metrics routes. Local scripted providers are test doubles, not live-provider quality evidence.

The domain-writer lane's executable regressions are in `tests/new/test_domain_writer_guards.py`; its receipt runner and hash manifest are under `.domain-writer-evidence/resume-01/`. The fresh final gates are 245 focused cases and 58 caller/lifecycle controls, each with actual rc 0 and reconciled JUnit (no skips/failures/errors), plus changed-file Ruff rc 0. These selected gates do not certify the full repository suite or the parent's combined-tree/query lane. Original interrupted receipts, including the older 193-pass run and later environment-symlink failure, remain historical evidence, not final success credit.

## Explicit compatibility changes

* Flat `.llmwiki/state.json` is not needed for normal source inventory. Current-generation cache/registry remain the accepted semantic state. Existing flat metadata is never automatically migrated in place.
* Ingestion does **not** move/delete successful clippings or inbox queue files. The operational index recognizes duplicates. Existing source filenames are never reused by replacement; a collision selects a new suffix atomically.
* `--resume` retries failed extraction as well as interrupted operations, with explicit retry reset only for resumed items. With no remaining extraction work it still invokes publication unless `--skip-synthesis` was requested. Old flat operation files are not silently adopted; an attempted legacy resume refuses before writes and directs the operator to inspect old records read-only and explicitly submit their URLs. An old ingest lock also refuses; it is not deleted automatically.
* `fix --apply` means **publish a regenerated projection**, not “apply each listed text edit.” Result JSON uses `mode: publish`, `generation`, and `pages_published`. Human-review findings are not automatically resolved. Changed inputs can require normal producer/reviewer calls; `fix --dry-run` itself makes none.
* Health's old incremental maintenance cursor is intentionally no longer written. `--full`/batch arguments remain accepted, but the reported scan is a full observational scan. Source-hash mismatch and unmanaged-generated-page findings remain available without a writable registry.
* The publication namespace is excluded from legacy graph, health and validation scans, so operational failure ledgers cannot become retrieved wiki evidence before the first build.

## Refused destructive options

* Mutating `ingest --refresh`: refused **before operational writes**. Use `--preview`, review the complete extraction, and save it under a new input filename. `--refresh --preview` remains observational.
* `fix --restore`: refused before writes even for a trusted old backup. Inspect/copy the backup to a separate annotation manually, or publish a new projection via `build --refresh-render`.
* `health --write`: refused; redirect the report outside the vault explicitly.
* Query answer destinations outside `05-Queries`, any publication destination, and any existing answer (even with `--force`): refused. Choose a new filename. A concurrent destination creation wins at installation.
* Unknown/corrupt publication namespaces or manifests: fail closed. No automatic repair, deletion, namespace adoption or pruning.

## Obsidian bridge

The plugin adds **OLW: Open current generated wiki**. It calls `olw current VAULT --json`, accepts only the exact generation-qualified index path matching the returned generation ID, waits for successful process completion, then calls Obsidian's `openLinkText`. No human homepage is rewritten. The CLI independently validates the manifest entrypoint.

The plugin was compiled, typechecked and tested with the **existing locked dependencies copied into the isolated candidate**, without a global install. Mocked Obsidian lifecycle tests are not native UI verification. No native Obsidian executable/display was available; **native UI remains unverified**.

## Safety and evidence boundaries

New tests exercise actual Typer commands plus a fresh-process CLI roundtrip using local scripted producer/reviewer/query doubles. No real providers, credentials, private evaluation fixtures or production vaults are used. Recursive preservation compares file bytes, inode, mode, mtime/ctime, uid/gid and supported xattrs for human paths and retained published files. Read-induced atime is excluded; creating new files may change parent directory metadata.

The final report/receipts live under `.caller-evidence/`. That directory retains the RED and intermediate failed receipts as well as final focused JUnit, actual statuses, source hashes and changed-file Ruff output. The old caller suite and the parent's previous full-suite failure inventory are separate gates; no blanket obsolete-fixture exemption is asserted. Path-ancestor TOCTOU hardening, power-loss durability, native Obsidian behavior and combined-tree acceptance remain separate boundaries.
