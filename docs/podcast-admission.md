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

> **Cache keys (epoch 3).** Podcast transcript caches are keyed by the
> episode identity *and* the selected artifact. Each entry carries explicit
> `podcast_binding` provenance. Version-1 (`guid|enclosure`, raw-URL) and
> version-2 entries are **never read automatically**. They miss once and stay
> byte-identical; reuse needs an explicit operator-attested import. See
> [podcast authority binding](podcast-authority-binding.md#legacy-entries-v1-and-v2-never-read-automatically).
> The per-provider admission rules below still apply to every entry.

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

At the resolver level (`load_transcript_cache`), AssemblyAI, Supadata and
local Whisper payloads are still loadable. The podcast extractor itself,
however, now reads only epoch-3 bound entries (see the note above). Legacy
podcast ASR entries are therefore **not** reused automatically, and the first
request after upgrade runs the configured ASR chain again. This supersedes the
earlier "legacy ASR caches remain reusable" statement for podcasts. Operators
wanting a strictly free refresh must disable paid providers in their effective
configuration first, or use the operator-attested import for entries they
have verified.

Already persisted source notes are not silently deleted or repaired by a cache
miss. Mutating `--refresh` is refused by current immutable callers. Inspect a
fresh extraction without replacing existing sources:

```sh
olw ingest /path/to/vault --url 'EPISODE_URL' --refresh --preview --json
```

Preview may invoke extraction/transcription services and incur cost; it is not
a no-network mode. Review the extraction and provenance, then save accepted
text under a new input filename, preserving existing sources. Use normal
`olw build` to publish a new immutable generation through review; changed inputs
may call synthesis/review models. See [application callers](immutable-application-callers.md).

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
