# Separately anchored quotation fragments

Current producer claims use `evidence_span_ranges: [{start_id, count}, ...]`.
Each range is contiguous; ranges must be ordered and disjoint. A claim is bounded
by **3 fragments, 6 span IDs total and 4000 quoted characters total**. The parser
still accepts historical `evidence_span_range` start/count or start/end objects,
but never both fields together, a free ID list, or model-authored quote/offsets.
All producer objects remain closed and the strict-schema call opt-ins are unchanged.

A single range materializes the old `EvidenceSpan` and retains its old serialized
shape. Multiple ranges materialize `EvidenceFragments`, serialized as
`{"version":"evidence-fragments-v1","fragments":[...]}`. Each fragment independently
stores quote, source filename/revision, catalog version/IDs, character offsets,
UTF-8 byte offsets (`start_byte`, `end_byte`) and verification state. There is no
outer quote or outer start/end hull. Raw text, including internal CRLF and Chinese,
is sliced without normalization. All admission readers rederive each anchor.

Consumers use `evidence_fragments()` explicitly. Fragment-aware identities prevent
deduplication, publication subjects, merge paths or preview counts from silently
collapsing different fragment sets. Rendering gives each fragment its own quote
and anchor. Cache and registry JSON use the shared serializer/parser; query and
preview apply the historical receipt verifier, not just outer checksums.

Review packets have `cited_fragments` instead of `cited_text` for multiple fragments;
`cited_span_ids` is their factual union, never gap IDs. Unit identity includes the
versioned ordered fragment records (exact quotes, source/revision, IDs, offsets).
The reviewer must judge every unit and the whole claim, without inferring an
unstated relation from co-citation. Support IDs outside the cited union still fail.
Scripted tests exercise these contracts; they do **not** prove semantic accuracy.

## Version and history policy

- Citation interface: `evidence-fragment-ranges-v1`.
- Producer epoch: `synthesis-v9-fragment-ranges-feedback-refine`.
- Compiler schema epoch: `synthesis-schema-v5-fragments` (new evidence union).
- Receipt: `grounded-review-v4`; historical v2/v3 receipts remain readable.
- Review prompt: `grounding-review-prompt-v7-fragments`.
- Historical single-fragment unit keys retain `review-unit-identity-v2` exactly;
  multi-fragment keys include the explicit `evidence-fragments-v1` discriminator.
- Cache envelope schema 2 and SQL schema are unchanged: their JSON payload accepts
  the versioned union. Current artifact fingerprints force private revalidation;
  historical publication admission is not revoked.

The bounded lane adapter currently wraps independently derived local v2 contexts
in `fragment-local-context-v1` (at most two bounded records per fragment). It never
uses the first-start/last-end hull as a citation. Exact wrapper/child records and
model context-use declarations are validated. The separately owned pure v3 context
API is intentionally **not** modified here; the integration owner should replace
this isolated adapter with that API and rerun combined-context controls.

## Verification

`tests/new/test_evidence_fragments.py` exercises parser → review request → receipt
→ serializer/reload → renderer/publication, actual pipeline cache/registry/query
and stable no-op. Negative controls cover hidden, stale, wrong-source, reordered,
overlapping, duplicate, excessive fragments/spans/chars, malformed counts and second
fragment tampering (including resealed unit digests).
`tests/new/test_fragment_controls.py` adds raw EN/CN LF/CRLF byte checks, strict cache
shapes, second-context tampering and actual resealed cache/SQL/plan/generation preview
and query rejection. Historical single-range parser negatives are retained.

Current-schema tests were adapted only where their expectation or positive wire
fixture named the superseded single-range producer schema/epoch. Required fields,
recursive closedness, 1..6 integer counts, strict provider wire checks, no-op
fingerprints, old endpoint overflow rejection and historical admission assertions
remain enforced. The former `test_wire_schema_claim_cites_one_closed_range_not_a_list`
is now `test_wire_schema_claim_cites_bounded_closed_fragment_ranges`; free ID lists
remain forbidden by unchanged negative tests.
