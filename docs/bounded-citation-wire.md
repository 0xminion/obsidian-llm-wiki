# Bounded citation and per-unit context wire contracts

## What changes

The grounded producer emits only:

```json
{"evidence_span_range": {"start_id": "<offered ID>", "count": 3}}
```

`count` is an integer enum `[1,2,3,4,5,6]`. It includes the start span. Exactly that many consecutive catalog spans are selected. This is an explicit selection, **not** permission to extend or repair an incomplete selection. The parser rejects booleans, fractional values, integral floats, strings, nonpositive or over-six counts, mixed endpoint/count objects, catalog overflow, foreign/revised IDs, hidden intervening spans and excessive character length. No clamping, clipping, gap filling of free lists, quote rewriting or silent claim removal is introduced.

Bounded legacy `{start_id,end_id}` responses remain parseable with all original checks, including the six-span and 4,000-character limits. They are **not** an alternate generation branch in the response schema. Free ID-list producer responses remain rejected. Persisted evidence remains the canonical application-built `span_ids`, exact raw-source quote, source hash, catalog version and offsets; neither wire form is persisted as evidence. Existing outer-whitespace handling in `SpanCatalog.cite` is unchanged. Interior CRLF, tabs, Markdown, Unicode and UTF-8 bytes are preserved exactly.

Review schemas now receive the exact `bindings` map also used to construct that request's prompt and validate its response. Each claim's `context_id` enum contains only that unit's `offered_context` IDs. Non-claims and claims with no offered context have `maxItems: 0`; `[]` remains valid. No empty enum or sentinel context ID is used. Nonempty arrays retain closed `{context_id,purpose}` objects, required keys, existing purpose vocabulary, and the runtime cardinality bound. Duplicate (ID,purpose) pairs and all existing response, support, receipt and semantic rejection checks remain runtime checks.

## Production paths and component adapters

- `core.pipeline._call_structured_synthesis` imports the shared `synth.response_schema.SYNTHESIS_RESPONSE_SCHEMA`. It supplies the same updated contract for single-pass generation, each chunk, diagnostic retry, grounded two-pass draft and refinement; no per-call schema fork is introduced.
- `synth.review._review_units` computes bindings once, uses `_prompt(window,batch,bindings)`, sends `_schema(batch,bindings)`, and validates with `_validate_batch(...,bindings)`. Both normal window and residual/final-admission review use that path.
- **Component harness signature change:** `_schema(batch)` becomes `_schema(batch, bindings)`. Map keys are `ReviewUnit.key`, not the transport `u1`/`u2` names. An offset-catalog harness must pass its already-constructed exact bindings to both `_prompt` and `_schema`; the response must use only IDs from those same offered records. For non-claims map to `None`; for a real claim with zero offered contexts supply its normal binding with `offered_context: []`. Do not fabricate empty bindings for a claim merely to satisfy the signature. No private/frozen harness data was edited.

## Compatibility and history

- Citation interface: `evidence-span-range-v1` → `evidence-span-count-v1`.
- Compiler prompt epoch: `synthesis-v7-span-range-feedback-refine` → `synthesis-v8-span-count-feedback-refine`.
- Reviewer prompt/schema epoch: `grounding-review-prompt-v4` → `grounding-review-prompt-v5`; reviewer fingerprint consequently changes.
- Persisted synthesis schema remains `synthesis-schema-v4-grounded`; receipt remains `grounded-review-v3`, unit identity and context policy are unchanged. This is a generation-contract change, not a stored-evidence migration.

Old admitted generations remain historical evidence, not newly current-policy output. A current compile regenerates/reviews; subsequent unchanged runs reuse the new generation without inference. Separate-process EN/CN probes import the actual frozen old code, then this candidate, then this candidate again; they verify canonical evidence equality, historical generation bytes/metadata, human-source bytes/metadata and stable no-op receipt. These are scripted offline transport tests, not evidence of semantic model quality.

## Provider schema constraints and remaining runtime obligations

Sources: [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs), [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs).

- JSON Schema has no portable cross-field arithmetic for `end.index - start.index <= 5`. The documented OpenAI subset excludes `if`/`then`/`else`, `allOf`, `not` and dependent schema keywords. Enumerating all valid start/end pairs grows with the source window and competes with schema/enum limits. A small integer enum bounds range length without that expansion.
- Objects are closed and all their properties required, including nested objects. No `$data`, regex arithmetic, `uniqueItems`, custom validator keyword or unsupported conditional was added. `count` uses `integer` + `enum`, not an advisory description alone. JSON Schema treats `1.0` as an integer-valued number; the application intentionally remains stricter about decoded Python integer type.
- Empty context sets use the documented base-model array `maxItems` constraint. Fine-tuned OpenAI models additionally exclude `minItems`/`maxItems` (and the existing producer's numeric bounds); they are **not** certified by this contract. Do not drop constraints to make an unsupported endpoint appear compatible.
- OpenAI documents 5,000 total object properties, 10 nesting levels, 120,000 characters across property/definition names and enum/const strings, and 1,000 enum values overall. Default review batches of 16 fit these limits. With up to two offered context records per claim, this schema uses at most nine enum values per claim (digest, verdicts, purposes, context IDs). Configurations above 111 all-claim units can exceed the documented enum budget; the application currently allows configuration up to 200. This pre-existing large-batch compatibility limit is documented, not silently changed in this patch. Use at most 111 for that provider; the default is 16. Provider acceptance at custom sizes must be checked separately.
- Native Ollama receives the same schema under `format`; OpenAI-compatible requests use `response_format.json_schema.schema`. The existing adapter does not set `strict: true`; this patch does not change transport policy. Exact serialized schemas are validated offline, but no provider enforcement, HTTP acceptance or inference-quality claim is made. Providers that ignore constraints still face unchanged application rejection gates. Ollama's documentation says its cloud does not currently support structured outputs.
- A start ID must still resolve in the offered source revision; the fixed producer schema does not enumerate every source ID. Counts crossing a window/catalog edge, too-long character ranges, duplicate context uses, unsupported support IDs, bad digests, incomplete coverage and unsupported assertions remain application rejections. Never silently repair selected evidence to pass them.

## Test maintenance

Only positive schema/version expectations are superseded: closed ranges now require `start_id/count`; prompt assertions describe count-inclusive ranges; version assertions advance to the new epochs; the prompt/schema type comparison recognizes integer count and checks its six-value enum. All existing negative test functions and runtime semantic policy are preserved. Self-authored EN/CN controls cover exact serialized schema validation, canonical legacy/count evidence equivalence, overlong legacy rejection, runtime guard independence, single/two-pass and multiwindow requests, and zero-call current no-ops. The test-only `jsonschema` dependency is declared and locked. Offline upgrade scripts and receipts live with delivery evidence, not private harnesses.
