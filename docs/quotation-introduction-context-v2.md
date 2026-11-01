# Quotation-introduction context policy v2

`review-context-v2` extends the existing bounded local-context offer, not the citation or approval contract. The motivating auditor-owned source (not a benchmark) is:

```markdown
Inez, explaining the Willow pump, said:

> “It must remain idle during calibration.”
```

The canonical citation stays exactly `> “It must remain idle during calibration.”`. Its IDs, source revision, character offsets and quote bytes are unchanged. The introduction and separator are offered separately in `context_binding.offered_context`.

## Narrow structural selection

The v1 same-paragraph selection remains the default local rule. Only its left blank-line boundary has this v2 exception:

- Within the containing visible interval and existing 384-character / 1536-UTF-8-byte per-side caps, recognize one complete, single physical introduction line ending in `:` or `：` (optional trailing horizontal whitespace).
- It must be followed by exactly one blank-line separator (LF or CRLF, optionally containing spaces/tabs), then the first explicit top-level blockquote line: zero to three spaces, `>`, horizontal whitespace, nonblank text other than a second `>`.
- The citation must start on that first blockquote line. A citation later in a multiline blockquote does not authorize backward search through the quote.
- The entire introduction, separator and any prefix before the citation must fit. Establish its start at source offset zero or a complete preceding paragraph separator within that bounded visible prefix. Do not recover a prefix cut by a window or cap, even if the full source contains it.
- Select only that immediately preceding line, never earlier paragraphs. The right-side paragraph boundary remains unchanged. Disjoint visible intervals are never bridged. Adjacent visible intervals can still coalesce under the existing rule.
- Conservatively exclude wrapped introductions, multiple blank lines, indented code, nested quotes, heading/list/container-style introductions and prefix Markdown syntax (`#`, `>`, `*`, `+`, `-`, backtick, tilde, pipe, brackets or ordered-list markers). Inline Markdown elsewhere in the line is retained verbatim. This is a bounded structural selector, **not a general Markdown parser or paragraph search**; it makes no claim to resolve arbitrary containers or document grammar.

Punctuation selects a candidate only. It does not identify a speaker, establish truth, prove the relation of the adjacent text, or supply independent factual support. Model-declared use remains a separate `model_context_uses` list, including a valid empty list. Only `referent`, `attribution` and `local_qualifier` purposes are accepted. Missing or ambiguous semantics must remain insufficient under the unchanged review instructions. Context IDs cannot replace citation support IDs.

Raw LF, CRLF, Unicode and Markdown are sliced from original source coordinates, not normalized or reconstructed. Character/byte bounds and hashes remain independently validated against the exact source body.

## Versions and historical integrity

The explicit allowlist contains only `review-context-v1` and `review-context-v2`. An explicit v1 constructor/validator invocation uses its original paragraph-clamped algorithm, with original v1 IDs/digests. Historical v3 receipts carrying v1 bindings remain integrity-readable, without rewriting or relabeling them. Unknown policy versions fail closed, even if a process's current-version constant is changed to an unknown value.

Every claim binding must match its receipt's context-policy version. Current-policy reuse additionally requires the current version and reviewer fingerprint. The fingerprint already incorporates the context policy, so this policy-only change invalidates inner cache compatibility and the outer fast no-op without changing prompt, unit identity, citation catalog or receipt schema versions.

## Executable checks and limits

- `test_quotation_context_boundary.py`: real canonical LF/CRLF requests plus single-newline control; exact source/quote preservation; bounded raw offsets, local-only selection, unseen/truncated gaps, known-v1 replay, policy mismatch/unknown rejection, allowed declarations and model insufficiency; raw LF/CRLF receipt/cache reload.
- `test_quotation_context_upgrade.py` and `quotation_context_process.py`: separate-process version-only old-policy build → current build → stable repeated build. The constant-only scenario is explicitly labelled a simulation; it does not pretend to import old code. The sibling evidence harness additionally imports hash-verified actual archived v1 source.
- The pipeline checks capture real scripted-model requests and verify receipt/cache/registry equality, rendered quote/claim presence, graph current-policy exclusion before upgrade, old generation bytes/metadata preservation, human-source preservation and a zero-call repeat with an unchanged receipt.

These are **engineering contract checks with offline scripted model answers**, not measured semantic correctness, speaker-identification accuracy, benchmark coverage or truth. No private references or live inference are used.

### Existing out-of-scope acceptance blockers

The strict CRLF disk-input publication test is intentionally not weakened or marked xfail. With both the archived v1 source and this candidate, `load_source_file` normalizes disk newlines before the immutable staging comparison, which rejects the raw CRLF `SourceDoc` with `stale source snapshot before staging: willow.md`. The context request/receipt/cache path itself preserves CRLF and passes. Source loading and publication are owned by the parent, not this change.

The existing `test_review_context_readers.py` resealed-container test also fails on both snapshots at its `publication.verified is True` assertion. The new policy does not clear that separate baseline blocker. Full-suite and live-inference acceptance remain with the parent; bounded source-bound receipts and logs are recorded separately.
