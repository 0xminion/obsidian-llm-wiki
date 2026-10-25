# Podcast transcript admission

Podcast acquisition and publication share an admission boundary. A transcript
heading, HTTP 200, or cache hit does not independently prove episode identity.

## Identity and route contract

- A uniquely matched RSS item's title is authoritative for the source title and
  filename. Contradictory storefront metadata cannot rename that episode.
- Apple requests require one explicit episode selector and a resolved canonical
  RSS episode. Apple show URLs, unresolved episodes, and ambiguous identities
  cannot publish inline content or reuse a cache as an identity substitute.
- Spotify show URLs are not episode URLs. The same route checks apply when the
  catch-all extractor retries a platform-specific URL.
- Ambiguous RSS matching remains a rejection through Podcast Index and iTunes
  discovery; a weaker fallback cannot erase it.
- Genuine inline transcripts without RSS remain supported on explicit episode
  routes (`/episode/<id>`, `/episodes/<id>`, `/e/<id>`, or `/episode-<slug>`),
  with a nonempty document title distinct from the show author. Generic home and
  show pages are not episode selections. Unknown no-feed layouts fail closed;
  import a transcript file explicitly rather than treating an arbitrary page as
  an episode.
- An inline document must match the canonical episode title, with the existing
  limited platform/show suffix normalization. Cached inline content is checked
  using its own stored document title, not relabelled with current metadata.
- A transcript-labelled link is not inline speech. Padding it with chapter or
  sponsor prose cannot bypass blocked destinations or ambiguous-link admission.
  Valid labelled links still use the existing bounded, public-URL-safe resolver.
- A rejected inline transcript remains a claimed-podcast failure even without a
  `description` field. It cannot escape through generic web fallback.

## Artifact identity and fidelity

`TranscriptResult.document_title` is optional structured document evidence.
Public acquisition captures actual HTML title/H1 elements before normalization;
inline acquisition captures the metadata title. It does not scan spoken prose
for episode mentions, and it does not rewrite the transcript text.

Explicit conflicting numbered document identity is rejected for RSS-tagged,
labelled, and cached artifacts. The RSS tag is not permission to publish a
transcript explicitly identifying another episode. Labelled links retain their
stronger positive title and substantive speaker-turn checks.

An unnumbered RSS display title can legitimately have an artifact headed
`Transcript for Show #181`. Such naming differences alone are not proof of a
conflicting episode. The authentic Changelog capture exercises this distinction.
Repeated speech, literal comparisons, source-language text, nested inline
subsections, and cached language/provenance remain intact.

## Existing caches and persisted sources

Text normalization remains **version 2**. A separate
`podcast_admission_version: 1` marks the current cache representation.

The following public providers require that admission version on load:

- `rss_podcast_transcript`
- `publisher_episode_page`
- `publisher_transcript_link`

Older public caches can lack the original document identity or contain an
incorrect episode/link-padding result. They now **miss without deletion or
mutation**, even if their normalization version is already 2. Successful
reacquisition replaces the cache atomically; failed reacquisition preserves the
old bytes as evidence. Current caches still pass the shared episode admission
check before publication.

Legacy AssemblyAI, Supadata, and local Whisper caches are unaffected and remain
reusable. This is not blanket ASR invalidation. If free reacquisition fails, the
normal configured ASR fallbacks still apply; operators wanting a strictly free
refresh must disable paid providers in their effective configuration first.

Already persisted source notes are not silently deleted or repaired by a cache
miss. To reacquire a particular source, use the existing explicit refresh flow:

```sh
olw ingest /path/to/vault --url 'EPISODE_URL' --refresh --parallel 0 --json --skip-synthesis
```

A failed refresh preserves the last-good durable source and cache. Ordinary retry
backoff remains in effect. Regenerate dependent notes after successful refresh
using the normal compilation and user-edit protections.

## Regression coverage

`tests/new/test_live50_review_b_closure.py` exercises public dispatch, CLI
persistence, inline and artifact identity, route and fallback guards, exact
language/body preservation, and both legacy/current cache admission. Existing
transcript-fidelity and cache-version tests remain unchanged.

The original Review B cases are replayed without modifying response bodies.
Its old assertion that an admission-unversioned public normalization-v2 cache
must hit is intentionally obsolete; the revised replay changes only that
expected hit to a miss and retains the reacquisition/repeat-access assertions.
No live or paid acquisition is needed for these captured/offline regressions.
