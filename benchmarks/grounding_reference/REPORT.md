# grounding-reference-v1: report

Built 2026-10-27 at baseline commit `7158bf2`. No new implementation, model
output, provider call or paid request was made or used. This report covers
reference construction only; it evaluates no new pipeline.

## Outcome

- **152 labelled cases**: 78 positive, 70 hard negative, 4 uncertain (excluded from primary metrics).
- **Calibration 46** (25 positive / 21 negative), **held-out 106** (53 / 49 / 4 uncertain).
  The two splits share no pair group, and all uncertain cases are held-out.
- **Tasks**: claim support 106 (55/47/4), quote fidelity 36 (18/18), citation locator 10 (5/5).
- **Sources**: T07 25, T03 20, T14 21, T26 23, T28 17, A03 16, T08 15, T24 15.
  By source language: EN 88, zh-Hans 44, zh-Hant 20. One cross-lingual claim is an English claim about a zh-Hant source.
- **38 historical cases** reproduce recorded v3/v4 model text verbatim (audited by
  string containment against the recorded responses: 38/38). Of these, 18 are
  historically unsupported outputs and 20 are historically supported.
- **Coverage obligations** for all 8 sources: 46 must-cover items, 17 qualifiers,
  19 must-not-assert items (82 total), plus out-of-scope page-chrome regions for T03.
- **Verification**: `verify_reference.py` ran **3,504 checks with 0 errors** against the
  archive and 3,502 against the durable inputs. It re-verified 9 bodies (T29 via
  T28's identical body), **378 span offsets/hashes**, including **all 94 evidence spans
  of the 78 positive cases**, and recomputed every candidate-quote occurrence list.
  Verifier mutation tests (offset shift, label flip, split straddle, snippet edit,
  manifest drift) each made it exit non-zero.

## Methodology

1. **Inputs.** Eight captured SourceDocs were used exactly as stored:
   `wiki-live50-live-synthesis/inputs/<ID>/source.json` in the private archive.
   T29 was added only as a duplicate-ID control. The original `/scratch` copies had
   been pruned; the archive was read and later cross-checked against the parent's
   durable restore (byte-identical for all 8).
2. **Reading.** One agent reader read each body directly. Prior semantic reviews
   (`semantic-review.json`) were used to *locate* known failures. Every label was
   re-derived from the source text, not inherited. They are one agent's source
   interpretations, pending independent adjudication; generator/reviewer
   agreement was not accepted automatically as reference truth.
3. **Spans.** Each span is resolved by a literal anchor in the captured body. The
   authoring script aborts if an anchor is missing or has an unexpected occurrence
   count. Offsets, UTF-8 byte offsets and SHA-256 are computed from the body.
4. **Cases.** There are 57 multi-case minimal-pair groups (70 pair groups including 13 singletons). Most hard negatives have a matched
   positive on the same evidence, so the benchmark tests whether a system notices
   the mutation, not just topical relatedness.
   - **Deliberate mutations:** negation 9, actor 10, number 10, qualifier 9,
     causality 5, wrong-span 8, case alteration 2, language/script alteration 3,
     markup omission 7, whitespace normalization 5, punctuation normalization 1,
     cross-source wrong ID 5.
   - **Historical failures included:** the T07 quorum-70 vs actual 54–42 conflation,
     holder-vs-seat actor errors (v3 summary and v4 body), the insufficient
     10%-only and 980-only spans, T14 PDF line-break joins, the T28 Pebble quote
     with its link stripped, T26 stripped bold and wrong-span claims, and T24
     case/LaTeX alterations.
   - **Multiple valid locations:** 4 repeated-text cases list every valid occurrence
     (2–4 offsets each). The T29→T28 case is valid under either source ID. Five
     cross-source citations are wrong IDs.
5. **Split.** Whole pair groups are ordered by `sha256("grounding-reference-v1/split"+group)`
   and assigned to calibration until about 30% of cases are taken. Groups containing
   an uncertain case stay held-out. The split was fixed before any new output existed.
6. **Copyright bounding.** No full source text is stored. Only short spans are stored
   (≤ 320 chars), plus offsets and hashes. Unique characters disclosed per source:
   T07 9.0%, T08 26.2% (a 2,419-char post), T26 4.6%, T14 5.0%, T24 2.7%,
   T28 4.2%, A03 8.9%, T03 14.6% (body includes Medium page chrome).
   Caps are enforced by the verifier. Obligation spans are locator-only.

## Label semantics

- `supported`: the cited span(s), read in source context, entail the claim. Cross-lingual
  paraphrase is allowed for `claim_support`. Quote tasks require byte-exact match.
- `unsupported`: the cited span(s) do not entail the claim. This covers contradiction,
  mutation, and insufficient attached evidence (`notes: supported_elsewhere=true`).
  `gold_support_spans` gives the correct or contradicting location.
- `uncertain`: a reader cannot fix the label from the source alone:
  - GR-HO-033: who executed FREE1376
  - GR-HO-073: a partial conjunct about Chinese segmentation
  - GR-HO-078: T24 states both 41.0 and 41.8 for EN-FR
  - GR-HO-095: whether Eric personally assembled the Pebbles

  These are not forced to a label.

## Model-free sanity baselines (no provider calls)

| Baseline | Split | All acc | Claim-support acc | Unsupported recall (claims) | Quote acc | Pairs fully right |
|---|---|---:|---:|---:|---:|---:|
| all-supported | held-out | 0.520 | 0.534 | 0.00 | 0.50 | 0/38 |
| exact substring (quotes) | held-out | 0.667 | 0.534 | 0.00 | 1.00 | 9/38 |
| char-bigram overlap, threshold 0.56 tuned on calibration | held-out | 0.745 | 0.644 | 0.59 | 1.00 | 19/38 |

Exact substring solves the quote/locator tasks by construction, which confirms
they are internally consistent. It cannot detect unsupported claims, which is
exactly the historical failure mode. With lexical overlap, held-out
unsupported-claim recall is 0.40 on actor, 0.50 on causality/qualifier, 0.60 on
number and 0.625 on negation. The claim-support negatives are therefore not
solvable lexically.

## Limitations

- **One reader.** All labels come from one agent reader; there is no independent
  second adjudicator yet. The parent's checker should audit them, starting with
  the `uncertain` cases and the `supported_elsewhere` negatives.
- **Not world truth.** Labels are source-grounded. T24's internal 41.0/41.8
  inconsistency is preserved, not resolved.
- **Small and source-specific.** 8 sources, 152 cases. Held-out shares sources (not
  pair groups) with calibration, so it measures generalization across claims, not
  across unseen documents.
- **Constructed mutations dominate** (114 constructed vs 38 historical). Some
  rationales depend on discourse context (pronoun referents, sponsored-segment
  scope).
- **Coverage obligations need a reader.** They are reference lists for human/agent
  judgment, not an automatic metric. The ≥75% must-cover pass rule is a convention.
- **T03 and A03 are parser replays.** Their bodies were produced by the current parser
  from saved responses (provenance in `sources.json`). Captures are faithful as
  stored, not re-fetched.

## Stable hashes

| File | SHA-256 |
|---|---|
| sources.json | `304feaba3059a06fe6dbc934a1b711464f54c29d7677f89e0ae6102c9f66f5c6` |
| cases.calibration.jsonl | `4d3db3aaafdcc7642ee1a2b7ffc964a4e2a3bc4024e8a76214fab36ab5c88d48` |
| cases.heldout.jsonl | `604745ac5d73718c6dc6b5bc41b3181997e30ae934bd82239bcf710f856b220b` |
| coverage_obligations.json | `769565d1a95e4f6d36ade1a9317f213333093bfcfb37c916b7060599d5ee7887` |
| MANIFEST.sha256 | `b63c0237051cd273eff321708d701b8cfdff3fad68f3e5e6174b33007db76726` |
| private archive | `25703b5e1f452114acf66f3f5b65a160cdc7023d1e95737131e481ba762532e0` |

| Source | Lang | Chars | content_sha256 |
|---|---|---:|---|
| T07 | en | 22315 | `35bebf8564ea5c1f24d094edf7c56a156b92e2ecea9ac239ebe1d10031c89efc` |
| T08 | en | 2419 | `1987b1ef294aef7da80c7c2050f64ad721f9006aac8506e1c475c6bb97257b6a` |
| T26 | zh-Hans | 11258 | `45508a70d3288f98d49a8813bd8e8abeab6f3efbcd139b8b1b8d87c959ea231f` |
| T14 | zh-Hans | 11817 | `4343d59c835d35f5be27f6aee3b84438434436030041926410d189a4c6f766ba` |
| T24 | en | 36609 | `c3ae8400f1c03fe97bb6356ce799bcd68c36a6423f0cbe059603c0345673e634` |
| T28 / T29 | en | 25678 | `719848954a46965c9aba677c49df16b7f169ca7d27a23a995c99a63fa1f1d779` |
| A03 | en | 6889 | `bfd53a674e7f86a2c08dab5a66176109b513fce64a4b6eed478618c397a0ddfc` |
| T03 | zh-Hant | 2509 | `2225257acfbc16d241e6c2ac73238b75cf3d773217cf50c65786610e2bd5baa1` |

Private authoring evidence (scripts, history inventory, baseline predictions and
verifier mutation tests) is in `/home/linuxuser/.hermes/cache/scratch/wiki-grounded-reference/`.
It is not committed.
