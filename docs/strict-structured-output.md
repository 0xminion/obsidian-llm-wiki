# Strict structured output

Application-owned producer, grounding-review and identity-assessment requests pass
`format=<JSON Schema>, strict_schema=True` through `acall_llm` / `call_llm`.
The OpenAI-compatible adapter consumes this local option and emits:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "response",
      "schema": {"...": "the unchanged application schema"},
      "strict": true
    }
  }
}
```

The schema placeholder above is explanatory, not an executable schema.

## Compatibility contract

- `strict_schema` is opt-in, not an inference from `format` or schema contents.
  Generic `format=dict` calls keep their prior envelope with **no** `strict` key.
  This matters for open objects and optional properties, valid JSON Schema forms
  that are not accepted by strict-output providers requiring closed objects and
  every property in `required`.
- For a new caller, enable the flag only after checking its entire schema against
  the chosen provider's supported strict subset, including nested objects and
  property/enum/depth budgets. The adapter does not rewrite or validate schemas.
- An explicitly supplied `response_format` always wins **as supplied**, including
  an explicit `strict: false`, a different name, JSON/text mode, or even `None`
  or `{}`. Falsey overrides are not treated as absent; invalid explicit requests
  remain the caller's responsibility and may be rejected by its server.
- `strict_schema=False` or omission leaves a generated envelope unchanged. The
  flag has no effect without a dict `format`, and does not alter `format="json"`.
- Native Ollama consumes the flag without sending it; `format`, native `options`,
  token limits and schema contents remain unchanged. Nothing is moved to OpenAI's
  `response_format` on the native Ollama route.
- A server that does not support strict structured output may reject these
  application requests. There is no automatic downgrade to JSON mode or non-strict
  output; existing retry/error and identity-unassessed behavior remains intact.

## Runtime validation is still required

Requesting strict output is not evidence that a particular server, model or
subscription bridge enforces it. Offline MockTransport tests prove the serialized
client request only; a real production-schema request is required to establish
server acceptance/enforcement. In particular, compatible providers may support
only part of JSON Schema (the producer uses numeric bounds, and the reviewer
uses `maxItems`).

The application retains all parsing, exact coverage, digest, evidence-anchor,
span-offer, bilateral identity and semantic review checks. A schema-conformant
response can still cite invalid support or make unsupported claims. Tests send
self-authored invalid responses even after a strict request to verify rejection;
those fixtures are not model responses or live enforcement evidence.

The generation schemas themselves are unchanged by this opt-in. Existing
schema/prompt version changes remain owned by their respective schema changes;
this request flag does not invalidate already runtime-validated artifacts.
