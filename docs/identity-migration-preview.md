# Identity migration preview (`olw identity-migration --dry-run`)

**Read-only inspection, not migration apply.** The command inventories legacy
schema-6 bindings and verifies existing schema-7 publication artifacts. It never
constructs the writable registry, plans a new generation, recovers a journal,
rewrites citations, calls a provider or implements `--apply`.

```bash
olw identity-migration "$VAULT" --dry-run
olw identity-migration "$VAULT" --dry-run --json
olw identity-migration "$VAULT" --dry-run --json --scratch "$WORK/preview"
```

`--dry-run` is required. Optional `--scratch` writes only the report, to a
specified directory outside the vault. Without it, the report is stdout only.
There are no lock, metrics, cache, schema, WAL/SHM or recovery writes.

## Supported inputs and classifications

The preview reads source bodies, synthesis caches, generated pages and render
ownership hashes. It loads `registry.db` **as bytes into an in-memory SQLite
connection**, not through `VaultRegistry(...)` and not via an immutable live DB
connection. It does not upgrade older databases.

For schema **7**, the preview checks:

- `identity-publication-plan-v1` marker and content-addressed archived plan;
- registry generation, source/review payloads, assignments, policy checkpoint,
  source/policy raw-file hashes and generated output/cache hashes;
- `grounded-review-v2` approved receipts and known publication binding bases;
- `identity-assessment-v2-bilateral` proofs where the binding requires assessment;
- known `identity-context-v2-bilateral` subject/owner context versions and exact
  historical `(filename, revision)` content availability. Candidate context
  aggregates in plan v1 have no separate version field;
- local or bilateral per-occurrence reference proof, separate from page-name
  wikilink existence.

A schema-7 binding is `resolved` only when these applicable checks agree. Valid
co-owned bindings from the actual committed plan can be resolved; a shared
name, copied ID, generic definition or ownership alone is not enough. Missing
history is never reconstructed from today's source, even when the text matches.
Unknown versions, absent plans, checksum drift and unsupported evidence are
reported rather than silently promoted. `publication.verified` concerns the
publication checkpoint; inspect binding statuses and blockers too. It is not a
semantic re-review by a model or an authenticity guarantee against an owner
rewriting all evidence and checksums.

For schema **6**, existing inventory behavior remains: ambiguous co-owners are
`needs_assessment`; missing/drifted sources and stale receipts are `unresolved`.
A sole-owner `resolved` observation is **legacy bookkeeping continuity only**,
not approval to merge/reuse an identity in a new publication. Old unilateral
assessments are counted as historical records, never accepted as bilateral proof.
Unsupported registry schemas cannot produce resolved bindings.

## Pending means blocked, even if the marker says committed

**Any existing `.llmwiki/publication-stage` blocks certification.** This includes
pending manifests, interrupted staging without a manifest, malformed/binary
manifests, and committed manifests awaiting cleanup. The report sets
`recovery_required: true`, lists `pending_publication_recovery`, and leaves
bindings unresolved. A pending marker without a stage also blocks.

This deliberately rejects the ambiguous state where `identity-publication.json`
says committed but the journal is pending: the next writer may roll it back.
The preview does **not** delegate certification to `publication_current`, which
in the audited candidate did not check this journal. A binary attachment or
other failed recovery does not make a pending stage clean. Preview never calls
recovery, retries it, decodes/restores staged files or removes a stage. Preserve
the complete state and obtain a matching, verified writer recovery procedure.

If SQLite itself is busy or holds uncheckpointed WAL, the command refuses before
inventory (exit 2); that is not a clean pending-generation report either.

## Exit codes and report contract

| Code | Meaning |
| --- | --- |
| **0** | Inventory completed without detected blockers; **not apply authorization**. |
| **2** | Refused, observed concurrent inconsistency, or usage error. No safe inventory. |
| **3** | Inventory completed with blockers, including pending recovery or unverifiable publication. |

JSON stdout is a single object. Successful inventories and blocker reports share
`preview_version: identity-migration-preview-v1`, `mode: dry-run`,
`status: preview` (or `inconsistent`), and these fields:

- `registry`: observed schema, in-memory strategy, DB hashes before/after, sidecars
  and observable locks. No database connection to the filesystem is opened.
- `publication`: recorded marker status, plan ID, checkpoint verification,
  recovery-required flag and observed journal status when schema 7 is inventoried.
- `sources`: registered/present/missing/unregistered filenames.
- `bindings`: source, canonical ID/slug, status/reason, owners, receipt versions,
  source revision comparisons, claim/citation counts, cache agreement,
  historical context availability and publication basis/plan provenance.
- `identity_references`: per-occurrence receipt records, original text/slug,
  target ID, recorded status, observed resolved/unresolved status and proof
  availability. Unresolved occurrences retain their original text and provenance.
- `references`: name-only rendered wikilink observations; **not identity proofs**.
- `cache_history`: archived cache paths, SHA-256 and byte counts. This is an
  inventory of archives, not validation or promotion of their payload versions.
- `summary`: binding statuses, registered source count, claim counts and counts
  of claims carrying catalog-span or quote citations. These are not distinct
  span-ID counts and do not count duplicate rendered footnotes as new claims.
- `human_edited_pages`, `blockers`, `notes`, `inconsistencies`, `vault_unchanged`.
- `assessment_estimate`: `status: unavailable`, `lower: null`, `upper: null`,
  measured `needs_assessment_bindings` and `model_calls_made: 0`. A request graph,
  retry policy and complete assessment budget are not enumerated by this reader;
  binding counts are not a defensible call estimate.
- `apply: {implemented: false, handled_by: "future plan build"}`. The historical
  field wording is retained for compatibility; the schema-7 build exists, but
  **this command still has no migration apply or recovery operation**.

Refusal objects have `mode`, `status: refused`, `vault`, empty `bindings`,
`apply`, `registry` (when observed), and `refusal: {reasons, action}`. Diagnostics
also go to stderr; they do not contaminate JSON stdout. Usage failures before
inventory may be ordinary CLI diagnostics rather than JSON.

## Read-only strategy and live filesystem limits

The preview refuses a nonempty WAL, any rollback journal, observable live SHM
connection, live compile-lock PID, unsafe symlink, missing vault, corrupt or
oversized DB, or unsupported SQL/payload shape it cannot traverse. **Do not delete
sidecars**: they may contain committed data. Let the owning application close
and checkpoint normally after stopping writers.

After inventory it rehashes the main DB, checks sidecar metadata and compares
entry types, sizes, mtimes and inodes under `04-Wiki`. An observed change yields
`inconsistent`, exit 2. Repeated quiescent previews are identical except for
`generated_at`.

These checks are **not an atomic filesystem-plus-SQLite snapshot or a writer
fence**. Use a quiescent, verified disposable snapshot when correctness matters.

- Uncoordinated editors can change bytes between observations, or change and
  restore bytes/metadata before the next observation. A post-check edit is not
  fenced. SHA-256 markers are not signatures or independent-principal authority.
- Linux `F_GETLK` lock observation is process-relative: same-process SQLite
  connections and unsupported VFS/filesystem locking are not universally covered.
  On unsupported platforms an existing SHM file whose lock cannot be inspected
  is refused. Offline checks are not a complete busy-writer detection protocol.
- Symlinks observed before reading are refused; adversarial ancestor replacement
  between filesystem operations is not an atomic containment guarantee.
- Tests compare paths, bytes, mtimes and inodes. **atime is not preserved or
  guaranteed**: ordinary reads can update it. No power-loss/fsync guarantee,
  all-interruption-boundary guarantee or live-vault acceptance is claimed.
- Registry deserialization is bounded to 2 GiB; archive scanning and hashing
  whole output trees add I/O. Large-vault scalability is not measured here.
- Human-edited pages, staged files and recovery-conflict data are observed, not
  reconciled. `resolved` is an observation of stored proof, not a migration plan.

## Backup → preview → review

Follow [migration.md](migration.md) to stop writers and snapshot the **whole**
vault, including any sidecars. Run the preview on that rehearsal copy; handle
exit 2 before trusting inventory, and review every exit-3 blocker. Preserve
pending stages and unknown-version artifacts rather than deleting them to make
preview pass. Re-run after an independently authorized recovery or build.

The test suites retain the original 14 legacy cases and separately build real
schema-7 fixtures through public `olw build` with an honest context-aware local
HTTP transport double. They cover committed plan/source/citation agreement,
repeat preview, constructor/provider/network prohibitions, pending/committed
marker disagreement, interrupted and binary/corrupt journals, missing history,
unknown versions/proof, source/review drift, separate-process busy/committed-WAL
refusal, real `olw` exit codes 0/2/3 and human/JSON output. Recursive vault and
isolated HOME snapshots prove no tested path, byte, inode or mtime changed.
These synthetic results are not production migration acceptance.
