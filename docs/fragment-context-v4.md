# Bounded linked-attribution context v4

`review-context-v4` is the current fragment-context policy. It adds an optional
literal earlier introduction to the existing v3 local offers; it does not widen
citations, certify a common speaker, or authorize source-wide claim context.
The [v3 document](fragment-context-v3.md) remains the historical specification.

## Selection and limits

The existing nearest-frame/before/after rules remain. V4 may additionally offer
one earlier complete plain introduction when the raw structure is exactly:

```text
A complete introduction ending in : or ：

> An earlier top-level quotation, possibly multiple nonempty quoted lines.

One complete, plain, non-colon-ended commentary line.

> The first line of the current quotation containing the citation.
```

The additional record has `relation: linked_quote_introduction` and `side: before`.
It retains literal text, source/fragment identity, character and UTF-8 byte
coordinates, hashes, and a bound context ID. Only the introduction line is the
new offer; intervening gaps do not become evidence. The quotation and its cited
IDs/offsets remain unchanged. The model must still establish the interpretation
from the explicit offers and may reject ambiguous or changed-speaker cases.

The search is at most 1,536 Unicode characters (at most 6,144 UTF-8 bytes), within
one visible island and outside every other citation/context interval. Introductions
remain at most 384 characters / 1,536 bytes. Existing global limits remain six
records, 2,304 characters, 9,216 bytes, and three cited fragments / six cited span
IDs / 4,000 cited characters. Existing sides reserve their record slots before
an optional earlier offer; insufficient capacity causes a conservative omission.

Headings, lists, tables, nested/container quotes, quoted code fences, extra
intervening paragraphs, truncated introductions, and new explicit colon-ended
introductions do not qualify. This is not a Markdown or natural-language speaker
parser. In particular, an ordinary commentary line can contain an actor change;
a structurally offered frame is not a proof of unchanged identity.

**Conservative limitations:** an additional introduction is omitted unless the
bounded prefix begins at raw source offset zero, so longer/partial windows and
later fragments may receive only their ordinary local context. A preceding fence
marker also suppresses the additional offer, even if a human can see that the
fence was later closed. These limits are deliberate; no hidden context is guessed.

## Evidence, declarations, and history

Only cited spans support facts. Context may resolve a referent, attribution or
local qualifier; it cannot supply independent premises, roles, quantities, list
members or causal relationships. Purpose tokens remain `referent`, `attribution`,
`local_qualifier`. Context IDs are not valid support IDs. A declaration or digest
proves neither semantic use nor truth.

The receipt envelope remains `grounded-review-v4`; no prompt or citation/unit
identity epoch is changed for this policy-only offer extension. Explicit v3
construction/validation retains its original serialized offers and IDs. V1/v2
single-fragment APIs remain unchanged. Unknown future policies fail closed.

In required-review mode the active policy enters the reviewer fingerprint and
inner/outer publication versions. An old receipt may remain historically valid
but is not current-policy reuse or query eligibility. A supported rebuild writes
a new immutable generation; it must not rewrite old generations or human files.
Before upgrading, follow [backup and migration rehearsal](migration.md).

`GROUNDING_REVIEW=off` remains explicitly unchecked. A context-only policy bump
does not force its outer publication no-op to rebuild; this option does not earn
semantic certification. Its behavior was not broadened by the v4 change.

## Verification scope

Generic controls exercise literal EN/CN/CRLF offers, structural barriers, bounds,
visibility holes, fragment clipping, historical reconstruction and corruption
refusal. Test success is not model accuracy or source-coverage acceptance.
A new frozen request manifest and explicit finite provider allowance are required
for live evaluation; neither a valid schema nor HTTP 200 establishes grounding.
