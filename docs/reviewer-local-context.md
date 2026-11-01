# Bounded context-binding reviewer contract

This document records the original v1/v2 contract and its receipt integration.
For current fragment offers and their preserved historical interpretation, see
[bounded linked-attribution context v4](fragment-context-v4.md).

## What is enforced

`grounding-review-prompt-v4` / `grounded-review-v3` replaces the prompt-only
local-context policy with exact app-offered records in the real reviewer request
and receipt. `review-context-v1` participates in `reviewer_fingerprint`, which
already participates in the inner artifact versions and outer compiler stamp.
The source text, supplied claim, evidence quote, citation IDs and anchor offsets
are not widened, repaired or rewritten.

For a claim, `context_binding` contains:

- `policy`, `source_file`, full `source_sha256`, `catalog_version`;
- `unit_key` (the existing exact text/subject/source-ref/anchor identity),
  `cited_span_ids`, `citation_start`, `citation_end`;
- `visible_intervals`: sorted, nonoverlapping half-open character intervals;
  adjacent intervals are coalesced without changing source text;
- `offered_context`: zero, one or two records, each with `context_id`,
  `char_start`, `char_end`, `byte_start`, `byte_end`, `text`, `text_sha256`.

Character coordinates are Python `str` indexes. Byte coordinates refer to UTF-8
encoding of that exact source **body**, not a Markdown file's frontmatter or an
original document container. All ranges are half-open. Context IDs are `ctx-`
plus full SHA-256 of canonical JSON binding metadata and the exact interval
record. They inhabit a different namespace from factual citation span IDs.

## Selection policy

Select at most **384 characters / 1,536 UTF-8 bytes on each side** of the exact
citation envelope. Stop at the containing visible interval's edge or the nearest
raw blank-line paragraph boundary (`\r?\n[ \t]*\r?\n`). Omit whitespace-only
records. Never jump a visibility gap, skip an oversized neighbor in favor of a
more distant one, or interpret a catalog span as a paragraph. Long paragraphs
are clipped by raw offsets; text is never normalized or reconstructed. At most
1,024 disjoint visibility intervals are representable by this policy.

The runtime adapter verifies the canonical catalog, exact source revision,
window text/spans/IDs/offsets and claim anchors **before provider dispatch**.
The pure constructor verifies source hashing and coordinate/visibility inputs.
Receipt validation rebuilds expected offers from independent source/claim
identity and compares the complete closed binding. Foreign sources, wrong
hashes, invalid offsets, arbitrary distant/full-source records, oversized text,
unknown policy versions and changed unit/citation bindings fail closed.

The window-wide SOURCE SPANS block remains available for non-claim units in a
mixed batch. It is **not** authorized context or evidence for a claim. The claim
rule authorizes only its own `cited_text` plus its explicit offered-context
records. This is a structural admission contract, not a mechanism proving which
visible text an LLM internally used.

## Model declaration and factual support remain different

Every response unit must include the new **required** field
`model_context_uses: [{context_id, purpose}]`. `purpose` is one of `referent`,
`attribution`, `local_qualifier`. Non-claims must return `[]`; claims may also
return `[]`. Extra fields, malformed lists, duplicate pairs, unsupported purposes
and IDs not offered to that exact unit are rejected. Both the outer response
schema and nested use objects remain closed.

These are **model-declared uses**, not app-observed reasoning. A hash establishes
byte/identity consistency, not semantic use or truth. A valid declaration does
not establish that the declared purpose was semantically appropriate.

Only citations can support factual assertions. `support_span_ids` remains a
nonempty subset of the claim's own cited IDs for supported claims. Context IDs
cannot be cited as factual support. Context may resolve references, attribution
and governing local qualifiers; it may not supply uncited independent facts,
roles, quantities, causal links, list members or unrelated entities. Missing or
ambiguous attribution/reference resolution must remain `insufficient`. Existing
whole-output rejection, digest, subject, heading, tag, relation, map and
citation gates remain in place.

## Pure offset API for independent callers

```python
from obsidian_llm_wiki.synth.review_context import (
    build_context_binding, validate_context_binding,
)

inputs = dict(
    content=exact_body, source_file=source_identity, source_sha256=body_sha256,
    catalog_version=explicit_catalog_name, unit_key=exact_unit_key,
    cited_span_ids=exact_ids, citation_start=quote_start, citation_end=quote_end,
    visible_intervals=[[shown_start, shown_end]],
)
binding = build_context_binding(**inputs)
assert validate_context_binding(binding, **inputs) == ""
```

The pure API accepts explicitly named offset catalogs without assuming their
segments are sentences. Its caller must validate that catalog's ID-to-citation
anchor mapping independently; the pure API cannot authenticate a mapping it was
not given. `review_window` deliberately accepts only the canonical application
catalog. Do not inject a differently segmented catalog into that adapter, invent
new citation bounds, or label such a catalog canonical. Disjoint citation
coverage, invalid intervals or unknown policies are unrepresentable/refused.

## Receipts, history and reuse

Current receipts preserve the app-offered binding and validated model declaration
per unit. The builder validates outcomes before returning a receipt; the same
pure integrity checks run on persisted cache/registry reload. JSON cache and
model containers already preserve arbitrary receipt dictionaries, so no producer,
provider, pipeline or model-schema refactor is required.

The only additional production integration is in `core/identity_migration.py`:
replace its hardcoded v2 receipt version with the review module's current version
and explicit known-version set. Without this small dependency, a valid v3 receipt
passed admission but was called an unknown version by the read-only preview.
Historical recognition and current-policy compatibility remain separate gates.

Explicit known historical `grounded-review-v2` outcomes keep their old
`kind/verdict/span_ids` shape and remain integrity-readable without adding new
fields. `grounded-review-v3` with known `review-context-v1` also remains readable
if the current policy changes. Unknown receipt/policy versions fail closed.
Historical integrity never establishes current-policy compatibility: current
reuse independently requires the current receipt version, context policy and
reviewer configuration. Old-policy generations remain immutable; builds publish
a newly reviewed generation rather than rewriting history.

A v3 claim's exact context cannot be authenticated from a whole-source digest
alone. The old `source_sha256`/empty-body shortcut returns an explicit
`exact source content required` reason for such claims. The existing pipeline
falls back to reading the source and can reuse a valid current cache without
inference; unchanged publication fast no-ops still use zero provider calls.
This is an intentional stronger contract, with a possible additional source read
on inner stat-only reuse. No fake content is reconstructed from the receipt.

## Evidence and limitations

Focused tests exercise EN/CN source → real pipeline → captured request → validated
review → receipt → cache/registry/publication → reload/no-op. Separate-process
transition evidence runs frozen baseline code and candidate code through the real
HTTP provider adapter with only `httpx.MockTransport` scripted; live connections
are forbidden. It checks old receipt readability, current-policy refusal,
config-supplied graph eligibility, recompile and no-op, byte/mtime/inode
preservation of historical generations, and byte/mtime preservation of human
source input.

Scripted answers test plumbing and rejection mechanics, **not semantic accuracy**.
No live model, benchmark/private labels or production vault was used. Parent
component evaluation and combined full regression remain required. Config-less
graph eligibility and identity-context outer-version stamping belong to the
other implementation lane, not this module.
