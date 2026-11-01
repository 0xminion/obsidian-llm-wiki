# Grounded synthesis: implementation and acceptance plan

Baseline: `7158bf2b7633f2270b6e8d6b2209486798d8aa18` (draft PR #23). The baseline source, previous generated responses, and previous failures remain immutable comparison evidence. This document defines planned acceptance, not a claim that implementation or benchmarks have passed.

## Root causes

1. Generation is asked to reproduce literal quotes. Markdown formatting, PDF line breaks, and repeated text make that brittle even when the paraphrase is otherwise correct.
2. Exact quote occurrence proves a location, not that the quote supports the whole claim. Summaries and other prose can also contain unsupported inferences even with no explicit claims.
3. A transcript URL/description is not enough to establish episode identity; missing discovery must not be repaired by weakening admission.
4. Legacy DOC conversion has a missing optional runtime dependency on this host.

## Implementation boundaries

- Build deterministic source-revision/identity-bound span catalogs. The model selects offered IDs; code materializes original quotes and Unicode offsets. Source bodies are never normalized, truncated, or rewritten to manufacture a match.
- Reject unknown, stale, wrong-source, out-of-window and malformed citations. Preserve explicit occurrence identity when text repeats.
- Add a separate support review before success state, synthesis caches and generated notes. Review coverage includes substantive summaries, highlights, concept bodies and claims, map explanations, and factual premises rather than only the optional claims array.
- Fail closed on incomplete/malformed review coverage, unsupported/contradicted/uncertain results and reviewer errors. A second model call is an evaluated gate, not ground truth.
- Bound context, batches, concurrency and retries. Never drop source chunks or review units silently. Version the relevant caches and persist review provenance/digests without trusting model-supplied approval flags.
- Preserve existing last-good notes, provenance, canonical identities, bilingual content and user edits. Unchanged same-version runs should make no inference calls or Markdown writes; metrics/locks remain observable writes.
- Install LibreOffice from a verified official distribution without changing unrelated configuration. System sudo is unavailable; prefer a side-by-side user-local install and test the actual legacy DOC path.
- Improve public podcast discovery only when independently checkable episode and artifact evidence exists. No cookies, paid ASR, account creation or access-control bypasses are authorized by this plan.

## Reference benchmark, frozen before new outputs

- Independently read the eight previously tested source captures and retain their exact hashes/offsets. Include both previously correct outputs and previous failures.
- Build at least 40 explicitly labelled claim/evidence cases, with correct examples and hard negatives involving actors, numbers, negation, causality, qualifiers, wrong passages and language/formatting differences.
- Separate calibration and held-out cases before evaluation. Expected labels, rationales and answer-bearing case names must not enter model requests.
- Review reference labels independently. Ambiguous labels are adjudicated or excluded explicitly, not changed to match model predictions.
- Record summary/highlight coverage obligations from the source so deleting claims, returning empty output or refusing everything cannot count as improvement.

## Release gates

1. Deterministic tests reject every invalid/stale/wrong-source citation control and preserve every original source character/offset in supported cases.
2. Frozen held-out semantic results report the full confusion matrix: supported acceptance, supported rejection, unsupported acceptance and unsupported rejection. The target is no known false acceptances and preservation of all agreed-correct benchmark cases; any miss is reported as a failed target, not silently relabelled.
3. Compare baseline and new results on the same cases and original source bytes. Report coverage, quote/anchor validity and entailment separately. Preserve every rejected model attempt and enforce a request budget before starting multi-chunk cases.
4. Exercise first ingestion, unchanged repeat, changed source/version, stale citation, failed review, resume, exact duplicates, user-edit preservation, EN/CN output and output syntax in disposable vaults.
5. Exercise actual LibreOffice DOC conversion and compare text with independent exports/known authored controls. Keep text fidelity separate from layout/OCR claims.
6. Recheck podcast positive cases A03/R01/S01/Y08 and negative identity/admission controls; test S02 discovery without converting an unverified episode into success.
7. Run the final Python matrix, package checks and independent source-to-artifact reviews. Keep all intermediate failures and bind final claims to exact source hashes/commits.

Passing a finite benchmark is not proof of universal factual correctness, publisher availability, unrestricted language coverage, or million-note capacity. No real vault migration or merge is implied by starting this implementation.
