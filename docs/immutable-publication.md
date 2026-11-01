# Immutable-generation publication (isolated implementation candidate)

**Historical lane report:** the status, caller limitations, suite failures, and demo path below record the initial implementation lane. For current CLI routing, including ingest, fix/recompile, health, current resolution, and the bridge, use the superseding [application-caller contract](immutable-application-callers.md), [quick reference](../README.md#common-commands), and [recovery guide](migration.md). Neither document certifies live-model semantic quality or production migration acceptance.

**Status: functional vertical implementation, not release-ready.** The old flat-layout suite is not green. See the evidence report for exact receipts and remaining failures. This document describes the implemented build path, not production migration approval.

## Ownership and format

`core.pipeline.run_pipeline` now invokes `core.immutable_publication.publish_pipeline`. The former pipeline is `_run_pipeline_workspace` and is used only on a new private staging vault by this entrypoint. Identity matching, bilateral assessments, source support review and prospective publication planning are retained.

```text
VAULT/
  04-Wiki/
    concepts/, entries/, mocs/, index.md       # legacy human-editable notes: untouched
    sources/                                  # human inputs: read only during build
    .llmwiki/schema.yaml, equivalences.json    # human policy: read only
    OLW-Publications/                          # publisher-owned namespace
      owner.json
      writer.lock                             # persistent flock inode; never unlinked
      current.json                            # sole authoritative commit pointer/manifest
      generations/<unique-id>/                # complete installed immutable projection
        concepts/, entries/, mocs/, sources/, index.md, views/, ...
        .llmwiki/
          registry.db                         # closed SQLite snapshot; readers deserialize in memory
          cache/, cache-history/, publications/
          identity-publication.json           # semantic-plan receipt, NOT another live pointer
          source-inputs/                      # captured input bytes, separate from rendered source views
      .staging/<unique-id>/                    # private build vault; failed attempts retained
      query-state/query-sessions.json         # mutable query sessions, not inside generations
  03-Raw-Annotations/entities/<canonical-id>.md
```

`current.json` has format `immutable-generation-v1`, `status`, generation ID, complete relative file SHA-256 inventory, input/config fingerprint, source-input hashes, semantic plan ID, errors, entity-to-page mapping, previous generation ID and the current entrypoint path. Logical entity IDs remain the semantic authority; new generation names are presentation/version identities.

Existing namespaces without this application's owner record cause a collision failure. Generation installation uses Linux `renameat2(RENAME_NOREPLACE)`. There is **no exchange operation and no destructive install fallback**. Systems lacking that syscall/filesystem capability fail rather than use ordinary replacing rename.

## Build and commit

1. Read input bodies and policy without opening/migrating the legacy registry for CLI inventory.
2. Acquire an advisory lock for competing publishers. Editors of human paths need not honor it.
3. Validate the current generation, including its complete hash inventory and the semantic plan's effective cache/SQL/page coherence. If fingerprint and successful status match, return a zero-provider-call no-op without changing the pointer or published files.
4. Create a fresh private vault. Seed from a validated previous generation, or closed legacy metadata on first migration; never copy legacy human output pages into writable output locations.
5. Capture current raw source bytes under `source-inputs`, and run synthesis/review/identity/registry/cache/render work privately. Render source views into the generation, never over input sources.
6. Preserve the admitted historical source revision in a stale retained source view. The failed newer raw input stays separately captured; warnings and retry/backoff stay visible.
7. Qualify generated wikilinks to exact vault-root generation paths, append stable annotation/legacy links, update only the private renderer inventory, and recheck cache/registry/page consumers against the original semantic plan.
8. Recheck observed source/policy bytes, flush files and directories, and exclusively install the new directory. Flush the prepared manifest, atomically replace **only the app-owned `current.json`**, and fsync its parent. No public reader accepts a first uncommitted generation. Readers with an existing current generation keep using it while a new workspace is pending.

Before-pointer crashes leave the old accepted generation authoritative. After-pointer process crashes never run a compensating rollback; restart validates and retains the accepted generation. Installed-but-uncommitted directories and failed workspaces are retained, not automatically recovered into current or pruned. This is process-crash evidence, **not a power-loss/filesystem-controller durability certification**.

## Human editing

- Edit `03-Raw-Annotations/entities/<canonical-id>.md`, original sources, policy, or retained legacy notes.
- Generated entity views link to their stable annotation path whether or not a template exists.
- `build --annotation-templates` prepares complete template bytes privately and installs them with no-replace semantics. It never opens an existing annotation for writing. A human file created at the final installation seam wins.
- Whole legacy pages with matching presentation names are linked as **legacy pages retained in place**. No claim is made that arbitrary legacy prose has been attributed or merged into reviewed generation content.
- All old generations remain. They are immutable by the publication protocol, not filesystem-enforced read-only files. Manual changes are corruption and are rejected, not repaired by overwriting them.
- File-preservation acceptance records bytes, inode, ctime, mode, mtime, uid/gid and binary user xattrs. Read-induced atime changes are excluded. Creating a new namespace/annotation can change parent-directory metadata. Native cross-user chown, every ACL/filesystem type and power loss are not certified here.

## Public entrypoints exercised

```sh
olw build VAULT --annotation-templates
olw current VAULT --json
olw query VAULT --ask 'What does this entity do?' --json
olw build VAULT                         # unchanged input => zero model calls
olw build VAULT --dry-run
olw validate VAULT
olw health VAULT --json
olw identity-migration VAULT --dry-run --json
```

`current` prints a validated generation entrypoint and an `obsidian://open?path=...` URI without creating or replacing a human homepage. The graph/query path indexes only current generation pages, not staging, prior generations, legacy notes or annotations. Query sessions are outside the immutable generation. Query `--save-answer --force` is refused inside the publication namespace.

Health uses a read-only full scan for immutable vaults; `health --write` is explicitly refused rather than modifying a generation. Validate and build dry-run resolve current and reject corrupt generations. The legacy identity preview remains available; its added `publication_workflow` section inventories retained notes, current generation, annotation targets/existing files and storage/call-cost limitations. **Its legacy identity section still addresses legacy metadata, so a new vault with no legacy registry reports that missing evidence and exits 3 despite a valid new generation.** No apply command or automatic cleanup is added.

## Costs and known integration boundaries

- Every changed publication copies the full prior generated tree, registry, cache and accumulated evidence, then performs a complete render. This trades storage and IO for non-overwriting ownership. There is no pruning/GC design in this candidate.
- No-op is zero model calls and no published writes, **not** zero IO: input bodies and generation files are hashed, the registry is deserialized and consumer coherence is verified. Scale optimization is not established by the focused tests.
- Initial legacy adoption requires a closed metadata snapshot. Uncheckpointed legacy WAL/journal state is refused. Concurrent replacement of app-owned namespace ancestors, malicious edits to app-owned manifests, symlink/path TOCTOU attacks and a legacy writer running a different lock protocol require additional hardening/audit. The demonstrated editor guarantee concerns the human file paths that this publisher never mutates.
- The existing native Obsidian plugin has not been migrated or GUI-tested. Native URI navigation and rendered links are statically checked only; `native_ui_verified` is explicitly false. This is not native Graph View/Properties/plugin UI acceptance.
- The low-level legacy renderer, legacy cache/registry callers, maintenance `fix`/restore and ingestion/operations workflows are not comprehensively migrated. Do not use them as independent writers on an immutable vault; only the exercised build/query/current/health/validate paths have this contract. Production enablement remains blocked on that integration work and the full-suite failures.

## Test-contract changes and retained assertions

Original semantic tests were not deleted, weakened, xfailed or bulk-retargeted. New tests explicitly assert that generated files live under the committed generation, edits go to annotations, and all original legacy/source/policy/annotation bytes and recorded metadata remain in place. The original intent-to-replace seam is still exercised: a compiler child pauses after `record_replace` and before `os.replace` on a **private staged generated page**, while a separate editor ignores the lock and edits legacy/annotation/source/policy files. The child then replaces its private file and hard-exits. Continued FD writes and two restarts preserve the human paths.

The out-of-tree `test_adapted_receipt_guards.py` imports the original 18 exact-output receipt negative tests and tamper functions unchanged; only their admitted-cache fixture resolves the committed generation instead of the flat metadata path. Its passing receipt does not erase the intentionally retained original suite failures.

## Reproducible synthetic CLI demo

From the candidate checkout, using the authorized interpreter and no installation:

```sh
python /home/linuxuser/.hermes/cache/scratch/wiki-immutable-publication/evidence/run_demo.py
```

This creates a unique scratch vault, isolates HOME/XDG/TMP before child imports, asserts the candidate import origin, prohibits network, scripts producer/reviewer/query responses, and invokes the actual CLI in separate processes. It saves exact commands, stdout/stderr, statuses and model-double call counts. The fixtures test implementation plumbing and preservation, not real model semantic quality.
