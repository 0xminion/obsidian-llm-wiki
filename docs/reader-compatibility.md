# Compiler and default-reader compatibility

## Outer publication gate

`_compiler_versions` retains its existing artifact, identity-assessor,
identity-plan and review-receipt fields and also records
`identity_context.CONTEXT_VERSION`. That value is an independent input to the
actual `_pipeline_checkpoint`; it is not covered by artifact versions.

An absent stamp or an older stamp without this field cannot take the outer
no-op. It triggers the existing private revalidation/publication path. It does
not necessarily require producer or reviewer calls. Once the new stamp is
published, unchanged builds regain the persistent no-op. No old generation,
human file, receipt validator or pipeline implementation is rewritten.

The finite regression matrix varies 15 version inputs over both review modes
(`required`, `off`) and synthesis modes (`single`, `two_pass`):

| Version surface | Expected compatibility effect |
| --- | --- |
| Extractor, synthesis prompt/schema, feedback contract | Artifact and inner checkpoint change; outer stamp changes |
| Grounded two-pass contract | Changes only for required/two-pass |
| Citation catalog/interface, review unit identity | Artifact and inner checkpoint change; outer stamp changes |
| Review prompt | Changes for required review; inactive for off |
| Review receipt | Extra outer gate, including off |
| Identity assessor prompt/record | Assessor and inner checkpoint change; outer stamp changes |
| Identity context | Inner checkpoint and outer stamp change |
| Identity plan | Extra outer gate |
| Registry schema | Not a stamp input: current-publication reader refuses before no-op |

Each probe asserts that an actual inner-checkpoint change cannot leave the
outer stamp unchanged. This is a finite audited surface, not a claim that any
future unversioned code change automatically invalidates caches. Changes to
reviewer internals should continue to flow through its public fingerprint and
verifier interfaces rather than duplicating receipt layouts in readers.

## Default graph policy

`build_graph_from_vault(wiki_dir, config=None)` resolves an independent,
read-only review policy using the existing `load_review_policy` helper. It uses
the caller's wiki parent as the vault; it never derives policy from a receipt or
an archived origin. The helper constructs no `Config`, provider or registry.

Precedence matches CLI vault selection:

- If the selected vault has `.env`, it is an explicit env file and its values
  override the process environment.
- Without that file, ordinary implicit resolution applies: process values
  override CWD `.env` values.
- A caller-supplied config remains authoritative; readers do not replace it
  with local configuration.

Historical receipt integrity is checked independently, then current-policy
compatibility is checked through `verify_review_receipt`. Valid old-policy
pages remain inspectable but cannot be used by `build_context` as current
evidence. Unavailable or invalid default policy is marked `unverifiable`, as
is default review-off policy (it cannot establish admission-review
compatibility). Explicit config and CLI review-off behavior are unchanged;
unchecked receipts still fail the strict integrity gate.

The low-level classifier also treats omitted config as unverified history,
not current eligibility. Lifecycle rejection, orphan status and mixed-owner
exclusion remain intact. Generic retrieval can still return historical
candidates for inspection; current CLI filtering and context admission retain
their separate guards.

Policy compatibility is not live-input freshness. API-only source snapshots
remain subject to the existing captured-only qualification; readers do not
claim upstream freshness or an atomic latest-at-commit guarantee.

## Focused verification

New tests: `tests/new/test_compiler_reader_compatibility.py` and
`tests/new/test_default_reader_policy.py`. They use real public pipeline,
graph, retrieval and context APIs, with only scripted provider transport.
Snapshots compare bytes, inode, mode, size, ownership, timestamps and xattrs
(excluding read-sensitive atime). This establishes persistent preservation,
not absence of write-capable syscalls.

The sibling evidence directory contains the RED/ GREEN JUnit receipts,
status-preserving clean-environment runner, unchanged copied independent audit
child, exact v6/baseline/candidate replay, preimages, patch and hash manifest.
The parent owns combined-suite and new reviewer-context implementation
validation; this lane changes no reviewer or receipt implementation.
