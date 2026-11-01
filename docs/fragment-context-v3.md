# Fragment-linked local context v3

Historical v3 specification. The current fragment selector is documented in
[bounded linked-attribution context v4](fragment-context-v4.md); explicit v3
construction and validation retain the behavior below.

## API boundary

`build_fragment_context_binding(*, content, source_file, source_sha256,
catalog_version, unit_key, cited_fragments, visible_intervals)` is pure and
returns JSON-compatible deterministic data. Each input fragment has exactly
`span_ids: list[str]`, `start_offset: int`, and `end_offset: int`. The application
adapter must validate **each** fragment against its actual catalog before this
call. This module checks raw source integrity, input types, offsets, limits and
visibility; it does not pretend to establish catalog membership or entailment.

`validate_fragment_context_binding(binding, **expected)` returns `""` on success
and a diagnostic string on rejection. Supply the **same complete constructor
inputs independently**, including raw content and actual visible intervals. Do
not reconstruct expected citation identities or visibility from an untrusted
binding. A source hash without its body is insufficient to check context bytes.

Citations are ordered, nonempty, disjoint character intervals. Touching citations
are permitted but are never merged. IDs must be distinct both within and across
fragments. Limits are three fragments, six span IDs total, and 4,000 cited
characters total. There is no fabricated bounding citation over a hidden gap.
Boolean offsets are rejected rather than treated as integers. Visibility must be
an ordered list of 1–1,024 nonempty in-bounds intervals; overlap is invalid,
touching intervals coalesce, and every citation must fit in one visible island.

## Binding and record shapes

The binding has exactly `policy`, `source_file`, `source_sha256`,
`catalog_version`, `unit_key`, `cited_fragments`, `visible_intervals`, and
`offered_context`. Its policy is `review-context-v3`.

Each output citation retains the input fields and adds `byte_start`, `byte_end`,
`text_sha256`, `fragment_index` (zero-based), and `fragment_id`. Fragment IDs bind
source identity/revision, catalog, unit, normalized visibility, and the **complete
ordered list of citations** with character/byte offsets and text hashes.

Every context record contains:

- `fragment_index` and `fragment_id`: its explicit owning citation;
- `relation`: `same_paragraph`, `adjacent_quote_introduction`, or `local_frame`;
- `side`: `before` or `after`;
- `char_start`, `char_end`, `byte_start`, `byte_end`;
- exact raw `text`, `text_sha256`, and `context_id`.

Context IDs bind the complete binding header, **all** citation identities, and
that context record. A change to the second citation therefore changes even the
first citation's context IDs. Validation reconstructs exact offers; recomputing
all public checksums around a changed offer cannot authorize it.

## Structural selection, not semantic inference

For each citation, select at most one local context record before and one after,
within its containing visible island. Each side is at most 384 characters and
1,536 UTF-8 bytes; the whole binding is at most six records, 2,304 characters and
9,216 UTF-8 bytes. Citation boundaries are hard limits for other fragments'
context. Context records cannot overlap any selected citation or one another.
If two fragments compete for context in their intervening gap, citation-order
first ownership wins; the later fragment cannot duplicate that offered interval.
The limits include all raw separators and any prefix before the citation.

Normally the nearest blank-paragraph separator clamps each side. The left side
alone may cross **one** blank separator to offer the complete nearest single
physical prose line or a local ATX heading (levels 2–6). The preceding line's
start must be established by source offset zero or a fully visible preceding
paragraph separator inside the same bounded prefix. The citation must start on
the first physical line of the following block. Never recover a truncated line,
search earlier blocks, cross a hidden separator, or search forward for a frame.

A plain prose line before an explicit first top-level Markdown blockquote is
labelled `adjacent_quote_introduction`. Colon punctuation is **not** required.
A local heading, or a plain prose line before ordinary prose, is labelled
`local_frame`. This label states only the exact local structural adjacency; an
unrelated nearest paragraph may still be offered. It does not identify a speaker
or establish that the relation is meaningful. The reviewer must decide whether
any permitted interpretation is warranted and may use none of the offered text.

Wrapped preceding blocks, multiple blank separators, indented code, list/container
introductions, document-title headings (`#`), nested quote destinations and
unrecognized Markdown block shapes are not recovered across the separator. This
is deliberately not a general Markdown parser or an NLP classifier. Existing
same-paragraph slices can still be bounded/truncated raw local text; **only**
cross-blank-line framing requires the complete predecessor.

No text is normalized or reconstructed: Chinese, combining characters, Markdown,
LF and CRLF retain their original character and UTF-8 byte coordinates. A
structural offer and its digest certify consistency, not semantic use or truth.

## Permitted uses and historical compatibility

Context can resolve a referent, attribution, local framing/scope or qualifier.
It cannot supply uncited factual premises—actions, quantities, causal relations,
or other independent facts. `validate_model_context_uses` accepts these v3 offers
without a new declaration vocabulary: only `referent`, `attribution`, and
`local_qualifier` are allowed. Local framing is declared as `local_qualifier`;
`framing` and `independent_fact` are not new accepted purpose tokens. Missing,
foreign and duplicate use IDs are rejected. An empty use list remains valid.
Call the binding validator before validating declarations.

Known policies now include v1, v2 and v3. The **old** single-fragment
`build_context_binding` and `validate_context_binding` APIs still handle v1/v2
only; relabeling their envelope v3 fails closed. Their algorithms, field shapes,
context IDs and serialized digests are unchanged. The captured legacy fixture
contains original-builder compact serialization and SHA-256 results for 12 input
cases under both historical versions, including raw CRLF/Unicode and clipped
visibility.

`FRAGMENT_CONTEXT_POLICY_VERSION` is v3. `CONTEXT_POLICY_VERSION` deliberately
remains v2 in this isolated pure-API change, preserving the old builder's default
and existing caller receipt headers. **Parent integration must switch the caller
receipt/reviewer policy and fingerprint epoch atomically with its v3 adapter.**
Do not merely bump the old constant while still calling the single-fragment
builder. Historical admission must dispatch v1/v2 bindings to the old validator
and v3 bindings to the fragment validator, without relabeling old publications.
No caller, receipt, cache, prompt, render or query module is changed here.

## Executable checks

- `test_fragment_context_v3.py`: first multi-fragment RED→GREEN, raw explicit
  frame links, exact second-fragment tamper rejection, frozen v1/v2 serialization.
- `test_fragment_context_integrity.py`: independent-input identity, byte/character
  boundaries, caps, disjointness, hidden gaps, full citation ID binding, resealed
  tamper, unknown envelope rejection and model declarations.
- `test_fragment_context_structure.py`: complete-or-absent structural boundaries,
  partial visibility, local-only selection, title/container rejection, no
  cross-citation contamination and nonaliasing.

These are offline engineering contract tests, not measured model accuracy,
speaker inference, source coverage or full application integration results.
