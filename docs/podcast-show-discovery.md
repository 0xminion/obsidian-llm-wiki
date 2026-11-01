# Spotify show-identity feed discovery

Scoped to podcast discovery (`extractors/podcast.py`, `publisher_transcripts.py`).
Admission, provenance and transcript-body rules in
[podcast admission](podcast-admission.md) are unchanged. Feed authority for
every route (Spotify, Apple and generic pages) and the epoch-3 episode- and
artifact-bound cache are defined in
[podcast authority binding](podcast-authority-binding.md), which supersedes
the name/date-only parts of this page.

## Gap

Real Spotify defuddle metadata has **no `author` field**. Its only show identity
is Spotify's generated description prefix, `Listen to this episode from SHOW on
Spotify.` The iTunes fallback therefore searched by episode title alone. For S02
(`#442 – Donald Trump Interview`) that search returned five unrelated shows, so
the canonical Lex Fridman feed and its explicit transcript link were never
reached. Earlier synthetic fixtures supplied an `author`, so this did not show up.

An independent review then showed that the older routes still bypassed the
identity rule when the show was known. The iTunes title search and Podcast Index
accepted an exact-title item from a feed whose channel was named like the show,
with no date or link corroboration and no cross-feed ambiguity check, and the
first feed won. Public `extract()` and `olw ingest --preview` returned a
fabricated mirror transcript, or sent the mirror's enclosure to ASR. The real
iTunes directory lists three entries named exactly "Lex Fridman Podcast": the
canonical feed, an acast mirror and a spreaker mirror.

## Rule: one identity policy for every Spotify route

Every Spotify discovery route goes through `_find_spotify_episode_asset`.

1. **Show identity is required.** It is the show named in Spotify's template.
   The template is matched even when no space follows the period
   (`… on Spotify.Guest`), and HTML entities are decoded. If there is no
   template, an explicit metadata `author` is used. With neither, no directory
   or feed request is made, and extraction fails with `Spotify show identity
   unavailable; canonical RSS discovery not attempted`.
2. **Publisher anchor (final review FB2).** Spotify's own description must carry
   exactly one explicitly labelled transcript link. Only listed feeds hosted by
   that link's publisher (`_feed_anchored_to`) are candidates; others, including
   exact-name mirrors, are never fetched. Without such a link no directory
   request is made: a show name and date cannot authenticate a feed.
3. **Candidates are pooled across routes:** Podcast Index (by show name, only
   with credentials) and one iTunes search each by show name and by episode
   title. An entry is a candidate only if its own directory name exactly equals
   the show after normalization. Unnamed entries (`collectionName: ""`) and
   near names (`… Clips`, `… | summaries`) are never fetched. URLs are
   deduplicated across routes.
4. **Bound.** The policy fetches at most **3** distinct anchored feeds in total
   (`_MAX_SHOW_FEEDS`), and fetches each feed at most once. If the routes
   together list more than 3 exact-name feeds, the episode cannot be resolved
   within the bound. Extraction fails as ambiguous **before any feed is
   fetched**.
5. **No first-feed wins.** Every candidate feed's channel title must also equal
   the show, and exactly one distinct exact-title item must exist across all of
   them. Listings with the same GUID and enclosure count as one item. A bare
   relisting with no transcript tag or labelled link adds nothing. Relistings
   with *different* transcript evidence, or different items, are ambiguous.
6. **Corroboration.** The Spotify publication day must equal the RSS `pubDate`
   within one day (time zones). Alternatively, the single explicitly labelled
   transcript URL must be identical in both descriptions. A contradicting date
   or a disjoint labelled link rejects the item.

The policy runs before the cache lookup, transcript fetch and ASR. An ambiguous,
contradicted or uncorroborated item therefore produces no request to its
transcript tag or links, no ASR call on its enclosure, and no cache entry.

Only an item that passed this policy (`EpisodeAsset.corroborated`) may borrow
Spotify's description links. An unverified enclosure can never acquire
Spotify's transcript link (defence in depth at extraction).

Title substrings, guest names and long shared prose are not identity evidence.
The resolved item then uses the existing labelled-link path unchanged:

- public-URL and redirect checks
- transcript heading and number check
- at least three substantive timed turns
- a cache keyed by episode identity *and* the selected artifact (epoch 3;
  see [podcast authority binding](podcast-authority-binding.md))

If no canonical episode is established but the metadata advertises a transcript
link, the error says so: `publisher transcript link advertised, but canonical
RSS episode identity was not established; link not followed`. The link is still
not fetched. The error also gives the identity reason.

### Request bound for one Spotify URL

| Step | Requests |
|---|---|
| Metadata (defuddle) | 1 |
| Podcast Index | ≤1, only with credentials |
| iTunes directory | ≤2 |
| Feeds | ≤3 distinct, each ≤6 hops (the shared five-redirect cap), RSS body ≤50 MB |
| Transcript tag or one labelled link | ≤6 hops |

For the authentic S02 replay this is 5 requests: metadata, 2 directory searches,
the publisher feed and the transcript (the acast and spreaker mirrors are no
longer fetched). The catch-all extractor no longer re-runs discovery
for Spotify, Apple or known podcast-host URLs. Their specialist already failed
closed, and that failure already blocks the generic-web fallback.

Non-Spotify routes (generic pages, Apple) share one per-extraction budget of
**8** distinct feeds (`_MAX_FEEDS_PER_EXTRACTION`), each fetched at most once.
Apple uses only its `lookup?id=` feed; generic pages fetch only feeds hosted by
the page's publisher.

## Backward-compatibility changes

- Spotify without template or `author` no longer falls back to a title-only
  search. It fails closed before any network discovery.
- Directory entries must carry the show's exact name. Synthetic fixtures with a
  bare `{"feedUrl": …}` entry no longer resolve.
- Spotify RSS matches now always need date or identical-link corroboration, on
  every route, not only on the show route.
- Podcast Index no longer short-circuits Spotify. Its exact-show feeds join the
  shared candidate pool.
- More than 3 exact-name feeds is ambiguous, with no feed fetched. Previously
  the first 3 were searched.
- `EpisodeAsset` gained `corroborated` and `unresolved_reason` (both
  `compare=False`, so equality is unchanged).

- Spotify episodes whose description has no single labelled transcript link,
  or whose feed is not hosted by that link's publisher, now fail closed
  (previously name plus date could resolve them). Same-name mirrors are no
  longer candidates, so they cannot make a valid episode ambiguous.

## Not covered

- Directories must list the publisher-hosted feed (5 iTunes results per query,
  plus Podcast Index when configured); otherwise extraction fails closed.
- An unavailable or truncated publisher feed fails closed: a same-name feed on
  another host is never promoted (final review SP01/SP14).
- More than 3 publisher-hosted exact-name feeds fails closed.
- Apple and generic routes: see [podcast authority binding](podcast-authority-binding.md).
- A publisher-labelled `RSS:` link inside the Spotify description is not used,
  because Spotify flattens descriptions and can corrupt such URLs
  (`…/feed/podcast/-`).
- Exclusive shows without public RSS, renamed feeds whose channel title differs
  from Spotify's show name, and unavailable feeds still fail.
- When the publisher transcript is blocked on a *corroborated* identity, ASR
  runs on the canonical enclosure, if keys are configured.

Regression coverage: `tests/new/test_podcast_show_feed_discovery.py`.
