# Grounding reference benchmark (`grounding-reference-v1`)

A frozen, **source-grounded** reference for evaluating source-span / claim-support
behaviour of the wiki synthesis pipeline. It was built independently of the
new implementation and before its evaluation outputs, from byte-exact captured
SourceDocs and recorded historical model outputs. Labels state whether the *captured source
text* supports a claim, not whether the claim is true in the world.

Nothing here is production code; no production module is imported.

## Files

The quoted case data and detailed coverage records are **private evaluation
inputs**, retained byte-for-byte under
`/home/linuxuser/.hermes/reports/obsidian-grounded/reference-v1/`, not committed
to GitHub. The repository contains the verifier/scorer, source metadata,
hashes and methodology. The original data manifest remains unchanged. The
character-fraction budget below is a storage check, not copyright clearance.

| File | Content |
|---|---|
| `sources.json` | 9 source records (8 benchmark sources + T29, a byte-identical duplicate of T28): URL, language, format, chars, UTF-8 bytes, `content_sha256`, SourceDoc file SHA-256, archive member, durable path. **No source bodies.** |
| `cases.calibration.jsonl` | 46 cases for tuning thresholds/prompts. |
| `cases.heldout.jsonl` | 106 cases for one-shot reporting after tuning is frozen. |
| `coverage_obligations.json` | Per-source summary/highlight obligations for 8 sources (must-cover, must-preserve qualifiers, must-not-assert, out-of-scope regions). Locator-only spans. |
| `MANIFEST.sha256` | SHA-256 of the four data files. |
| `verify_reference.py` | Stdlib verifier (public and private modes). |
| `score_predictions.py` | Stdlib scorer for `{id,label}` predictions. |
| `REPORT.md` | Methodology, label inventory, baselines, limitations, hashes. |

## Case schema (one JSON object per line)

- `id` (`GR-CAL-###` / `GR-HO-###`), `key` (stable readable key), `split`, `pair_group`
- `task`:
  - `claim_support`: do the cited `evidence` spans (jointly) entail `claim`?
  - `quote_fidelity`: is `candidate_quote` a byte-exact substring of the cited source?
  - `citation_locator`: does `candidate_quote` occur in `cited_source_id` (wrong source IDs, repeated text, duplicate sources)?
- `source_id`: the source the content actually comes from. `cited_source_id`: the ID being claimed.
- `label` ∈ `supported` | `unsupported` | `uncertain`; `polarity` ∈ `positive` | `hard_negative` | `uncertain`
- `mutation`: `negation`, `causality`, `actor`, `number`, `qualifier`, `wrong_span`,
  `case_alteration`, `language_alteration`, `markup_omission`,
  `whitespace_normalization`, `punctuation_normalization`, `cross_source_id`, or `null`
- `origin`: `constructed` or `historical:<run>/<case>/response-NN/<path>` (verbatim recorded model text)
- `evidence`: cited spans. `gold_support_spans`: where the source actually supports the claim, or contradicts it (for negatives).
  Each span has `{source_id,start,end,byte_start,byte_end,sha256,chars[,text]}`.
- `candidate_quote`, `candidate_quote_sha256`, `candidate_quote_occurrences` (all exact occurrences in the cited source)
- `rationale`, `notes`, `adjudication`

Offsets are Python code-point offsets into SourceDoc `content`. `byte_*` are UTF-8
offsets. `sha256` hashes the UTF-8 bytes of `content[start:end]`. Raw formatting
is part of the text and is never normalized: Markdown links/bold, LaTeX, PDF hard
line breaks, trailing double-space hard breaks, U+2019 and U+00A0.

`text` is stored only for spans ≤ 320 chars. The verifier caps the
unique source characters disclosed per source at 10% (30% for sources under
5,000 chars). Obligation spans store no text.

## Reproduce / verify

```bash
# Point to the frozen private labels; do not copy quoted case data into the repo.
REFERENCE_DIR=/home/linuxuser/.hermes/reports/obsidian-grounded/reference-v1

# Structural checks: schema, splits, counts, snippet hashes and manifest.
python3 -I benchmarks/grounding_reference/verify_reference.py --root "$REFERENCE_DIR"

# Full re-verification against the private captures (read-only).
python3 -I benchmarks/grounding_reference/verify_reference.py --root "$REFERENCE_DIR" \
  --durable-dir /home/linuxuser/.hermes/reports/obsidian-grounded/reference-inputs
python3 -I benchmarks/grounding_reference/verify_reference.py --root "$REFERENCE_DIR" \
  --archive /home/linuxuser/.hermes/reports/obsidian-live50/private-evidence.tar.gz

# Tune on calibration; report held-out once. Missing/abstained predictions
# never receive correctness credit. Duplicate/unknown IDs are rejected.
python3 -I benchmarks/grounding_reference/score_predictions.py preds.jsonl --root "$REFERENCE_DIR" --split calibration
python3 -I benchmarks/grounding_reference/score_predictions.py preds.jsonl --root "$REFERENCE_DIR" --split heldout
```

The private inputs are the SourceDoc JSONs under archive members
`wiki-live50-live-synthesis/inputs/<ID>/source.json` (archive SHA-256 in
`sources.json`). The parent restored byte-identical copies to the durable
directory above. T29 is absent from the durable set; it is verified through
T28's identical body.

## Reference v2 (`grounding-reference-v2`): verifier and scorers

v2 is derived from the frozen v1 and is **private** like v1: no v2 case data,
obligation text, label or ledger content is committed here. The repository
holds only the tooling below plus invented test fixtures
(`tests/new/test_grounding_reference_*.py`).

| File | Purpose |
|---|---|
| `verify_reference_v2.py` | Portable stdlib verifier. All locations are explicit, required arguments. |
| `score_predictions.py` | Label scorer for v1 **and** v2 gold (validates gold before scoring). |
| `score_coverage.py` | Aggregates independent coverage judgments under two rubrics. |

```bash
REF=/home/linuxuser/.hermes/reports/obsidian-grounded
python3 -I benchmarks/grounding_reference/verify_reference_v2.py \
  --root "$REF/reference-v2" --baseline "$REF/reference-v1" --sources "$REF/reference-inputs" \
  --baseline-pin benchmarks/grounding_reference/MANIFEST.sha256 \
  --expect-manifest-sha256 eef51dde53511c2dfdb08ffce4698c6c1ab0900c3e68b008984c79cbdb6ba69d
```

The verifier checks the following:

- **Hashes.** Both MANIFEST files must list exactly the expected files, the
  files must match on disk, and there must be no unlisted files. The optional
  pins are checked. v2 `sources.json` must be byte-identical to v1. Every
  SourceDoc file and content hash is checked; a byte-identical alias is
  verified through its target.
- **Unchanged v1.** It checks every v1 record-line hash, every v1 field value,
  the case order, the pair groups and the splits. A v1 field may differ only
  when all of these hold:
  - the field is changeable (`label`, `polarity`, `rationale`, `gold_support_spans`);
  - a `CHANGES.jsonl` row records exactly the original and new values (span
    lists may be recorded as exact ordered `{source_id,start,end}` locators);
  - the case cites that row in `change_refs`;
  - the case carries `adjudication_v2`.

  Labels may change only away from `uncertain`.
- **Typed fields.** `flags` must have the exact key set and types. Occurrence
  counts and `valid_source_ids` are recomputed from the bodies. It also checks
  the `reference_spans` roles and relations (`credits_citation` must be
  `false`), the E3 `context_spans`, and that every `guideline_refs` entry is
  defined in the dataset's GUIDELINES.
- **Coverage structure.** Item ids, points, qualifiers and must-not items must
  be unchanged. It checks atomic requirements (unique, prefixed, evidence ⊆ item
  evidence), `all_essential` and `at_least_k_groups` rules (each group needs an
  essential requirement), conditional blocks, and that every evidence change is
  logged and audit-flagged.
- **Spans.** Every span is re-hashed against the SourceDocs, using code-point
  offsets, UTF-8 byte offsets and SHA-256.

The JSON output contains only counts, never case text. Exit 1 on any error.

### Label scoring with v2 gold

`score_predictions.py` rejects gold that could change a denominator:

- duplicate, empty or non-string ids;
- an id prefix or `split` that doesn't match the file;
- an id shared with the other split file;
- an unknown label or task;
- a polarity or mutation inconsistent with the label;
- records that mix v1 and v2;
- v2 `flags` inconsistent with the label or task, e.g. a `primary_metric` that
  disagrees with `label != "uncertain"`.

`--verify-manifest` additionally requires `ROOT/MANIFEST.sha256` to match the
case files.

For v2 gold it adds the separate slices `flag:supported_elsewhere`,
`flag:multi_occurrence` and `flag:context_spans`. A missing prediction or an
abstention still never earns credit.

### Coverage scoring (`score_coverage.py`)

**The scorer aggregates judgments; it does not establish truth.** An
independent adjudicator makes every requirement judgment by reading the private
capture. The adjudicator is a human or an agent, but never the producer. The
scorer only checks how those judgments are bound and combined.

```bash
python3 -I benchmarks/grounding_reference/score_coverage.py --root "$REF/reference-v2" \
  --outputs outputs.json --judgments judgments.json [--require-complete]
```

**Inputs**

- `outputs.json` has the form `{"producer": ID, "outputs": [{"source_id", "path"}]}`.
  The scorer hashes each output file's bytes.
- `judgments.json` has the form
  `{"reference_version", "coverage_obligations_sha256", "judgments": [...]}`.
  Each judgment has these fields:
  - `source_id`;
  - `output_sha256`, which must equal the supplied bytes;
  - `adjudication`, with `judge_id` (not equal to the producer) and `independent: true`;
  - `language`, as `{observed}`;
  - `applicability`, for conditional items only: `applicable`, `not_applicable` or `unknown`;
  - `requirements`: `hit`, `miss`, `unknown`, or `not_applicable` (only under a
    `not_applicable` item);
  - `qualifiers`: `respected`, `omitted`, `violated` or `unknown`;
  - `must_not`: `absent`, `violated` or `unknown`.

The scorer rejects unknown ids, unknown values, duplicate judgments, hash
mismatches, non-independent judges, and obligations that don't match the
MANIFEST.

**Two rubrics, reported separately**

- **`frozen_v2`** is the frozen v2 rule, unchanged. An `all_essential` item is
  hit when all of its essential requirements are hit. A bundle is hit when
  ≥ k groups have all of their essential requirements hit.
- **`strict`** requires every essential requirement of an item across **every**
  bundle group. It ignores k. `strict.all_items_hit` reports whether every
  applicable item was strictly hit.

Both rubrics use the v1 source threshold, `hits ≥ ceil(0.75 × n_applicable)`.
Both also apply the same qualifier, must-not and language gates. The report
also includes atomic recall and essential recall.

**Completeness**

- A missing or `unknown` judgment never earns credit. It stays in every
  denominator and marks the source `incomplete`.
- An incomplete source's pass is `null`. It is `false` only if the source would
  fail even with every unknown resolved favourably.
- A conditional item leaves n **only** when it is judged `not_applicable`
  explicitly. A missing applicability judgment keeps the item applicable and
  unhit.
- Every expected source stays in the aggregate denominators. A source with no
  output fails, and a source whose output has no judgment is incomplete.

`--require-complete` exits 3 when the report is incomplete.

## Rules for use

- Do not edit labels after any new model output exists. Any change requires a
  new version string, a new manifest and a written reason.
- Calibration and held-out never share a `pair_group`.
- `uncertain` cases are excluded from primary metrics.
- Exact-quote presence is not entailment. Entailment is not world truth.
