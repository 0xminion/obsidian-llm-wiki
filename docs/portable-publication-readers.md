# Portable publication readers

## Compatibility contract

Normal readers use `current_publication()`: the envelope inventory, committed
semantic plan and consumers are checked, and the registry must have the exact
canonical supported schema token (`7`) plus every required table/column type
from the versioned writer DDL. Registry bytes are deserialized into anonymous
memory; no `VaultRegistry`, migration, disk SQLite connection or sidecar is
used. Resealing hashes does not make an unsupported schema supported.

Diagnostic identity preview alone may inventory an unsupported version with
blockers. It cannot certify that publication or its bindings. Existing statuses
remain: 0 clean preview, 2 refused unsafe/corrupt/unreadable snapshot, 3 completed
inventory with blockers. Semantic corruption still refuses rather than falling
back to a weaker envelope-only reader.

## Logical inputs and exact-copy rehearsals

New envelopes keep `format: immutable-generation-v1` and add
`source_inputs_format: vault-relative-v1`. `source_inputs` records sorted pairs
of vault-relative locator and SHA-256 (or null for an absent input). Allowed
locators are exactly:

- `.env`
- `.llmwiki/schema.yaml`
- `04-Wiki/.llmwiki/schema.yaml`
- `04-Wiki/.llmwiki/equivalences.json`
- `04-Wiki/sources/<validated source basename>.md`

The four policy anchors are mandatory, including absent ones. A committed source
must have a recorded logical input. Paths containing traversal, noncanonical
spellings, duplicate logical identities or other namespaces are rejected.
Live file reads check the complete mapped set for symlinks before content reads.

For known old envelopes without `source_inputs_format`, the `.env` locator
anchors ONE lexical absolute origin. Every remaining policy/source locator must
belong to that same origin and the allowed logical namespace. Only the relative
suffix is used beneath the copied vault. The historical origin is never resolved,
stat'ed, or opened; an original vault still present, modified or deleted makes
no difference. Inconsistent or incomplete mappings fail closed, not by basename
fallback. Unknown metadata formats require a compatible reader/migration.

A no-op legacy build replays the old fingerprint with only the two actual Config
location fields bound to that validated lexical origin. It does not normalize
arbitrary strings. New fingerprints exclude only those location fields. No-op
builds do not rewrite pointers or immutable generations, including old ones.
Changed inputs require a new generation through the normal writer.

Archived semantic-plan paths, exact quotes, URLs, source snapshots and proof
hashes are historical evidence. No path substitution or literal rewriting is
applied to those bytes. Copy the entire quiescent vault and run preview against
the copied root; no preparatory edit of `.env` or old generation files is needed.

## Shared freshness API

```python
from obsidian_llm_wiki.core.publication_inputs import input_freshness
result = input_freshness(vault_root, manifest)
# {"status": "current" | "drifted" | "unverifiable",
#  "changed_paths": ["04-Wiki/sources/source.md", ...], "reasons": [...]}
```

This API does no writes, provider calls or Config construction. It describes
observable file-input freshness only. Its inventory check includes newly added
source files, not just recorded hashes; capture includes empty discovered files
and excludes generated source navigation using the canonical loader rules.
An absent recorded source remains a captured snapshot: the helper adds
`input_scope: captured_only`, `unobserved_sources`, and an explicit reason.
This does not establish that an upstream URL/API is current. Query results carry
`freshness.status: captured_only` and an explicit human/saved-answer qualifier.
Callers must first validate the generation (registry/schema/context/consumer
agreement), and independently enforce their review-policy/admission requirements.
The helper does not claim to verify policy supplied only by process environment.
Invalid mappings or unsafe paths produce `unverifiable`, not `current`.

Query callers should gate before provider/state writes and recheck before
accepting or saving a response. That integration belongs to the query lane.

## Limits

- Path-based containment retains the existing filesystem TOCTOU limitation;
  these tests do not establish hostile concurrent symlink-race protection.
- File hashing is observational, not a multi-file atomic snapshot. Existing
  immutable generation/race controls remain separate from input freshness.
- No native Obsidian UI, power-loss durability, live providers, private evals or
  production vaults were exercised.
- Tests use synthetic evidence and deterministic local model doubles; provider
  quality and truth of source assertions are outside this compatibility gate.
