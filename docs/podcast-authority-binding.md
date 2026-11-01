# Podcast identity authority and cache binding

Scope: podcast discovery and the transcript cache (`extractors/podcast.py`).
This doc answers two questions. Which feed may speak for a requested episode
page? Which cached artifact may be served for it?

Transcript-body admission, labelled-link rules, URL safety and fidelity are
described in [podcast admission](podcast-admission.md) and
[show discovery](podcast-show-discovery.md). They are unchanged.

## Problem (final review FB1, FB2, AG01–AG07, CB, SC)

A directory search (iTunes or Podcast Index) only **lists** feeds. Earlier code
accepted a listed feed because its name, channel title, item title, date, or a
link copied from the platform matched. Those are all public, copyable strings.
Reproduced through public `extract()` and the CLI:

- **FB2.** With the canonical feed down (502) or truncated, a same-name mirror
  feed won. The mirror then supplied its own VTT, its own `Audio:` line and its
  own cache key. Copying Spotify's transcript link made that worse, not better.
- **AG01, AG02, AG04, AG05.** On generic pages and Apple, the first title hit
  in any listed feed won. Another show's VTT was returned, or another show's
  audio was sent to ASR. On Apple, Podcast Index ran before the authoritative
  ID lookup.
- **FB1.** An unresolved Spotify identity fell back to the raw URL as its cache
  key. A legacy entry stored under that key returned a fabricated mirror VTT.
- **CB and SC.** The cache key was `guid|enclosure`, `guid` or `enclosure`.
  Two episodes with no GUID (or the same GUID) and one shared enclosure served
  episode A's transcript or ASR text for episode B.

## Authority rule (one model for every route)

A feed item can publish a transcript, an `Audio:` line, an ASR input or a cache
identity only if its feed is **authoritative** for the request. The feed is
authoritative only through evidence from the requested page or platform:

| Route | Authoritative feed | Never authority |
|---|---|---|
| Direct RSS URL | The requested feed | — |
| Apple `…/id<show>?i=<ep>` | The `feedUrl` from `itunes.apple.com/lookup?id=<show>` (a `collectionId`, if present, must equal the show ID) | Podcast Index, iTunes title search, anything else listed |
| Spotify episode | A listed exact-show feed **hosted by the publisher of the one explicitly labelled transcript link in Spotify's own episode description** | Same-name feeds on other hosts, unnamed entries, feeds on the link's host but for another tenant, and everything when there is no such link |
| Generic episode page | A listed feed **hosted by the requested page's publisher** | Every other listed feed, whatever its title, author or date |

"Hosted by the publisher of URL `U`" (`_feed_anchored_to(feed, U)`) means:

- **Single-tenant hosts.** The feed host equals `U`'s host (ignoring `www.`),
  is a subdomain of it, or is its parent domain. For example, `lexfridman.com`
  anchors `lexfridman.com/feed/podcast/` and `feeds.publisher.example`, but not
  `publisher.example.evil.test`.
- **Shared platforms.** These include Buzzsprout, Podbean, Anchor, Libsyn,
  Acast, Spreaker, Substack and WordPress (the list is in
  `_SHARED_PODCAST_PLATFORMS`). Both URLs must be on the same platform, and
  `U`'s tenant must equal the feed's primary tenant, not an arbitrary token
  later in its path or subdomain chain. The tenant is the subdomain label
  adjacent to the platform domain, or else the first non-generic path
  segment of at least 3 characters. Recognized feed suffixes (`.rss`, `.xml`,
  `.atom`) are removed from that feed segment. For example,
  `www.buzzsprout.com/111111/episodes/…` anchors
  `feeds.buzzsprout.com/111111.rss`, but not `feeds.buzzsprout.com/999999.rss`.
  Nor can `bob.wordpress.com/tag/alice/feed/` speak for `alice.wordpress.com`.

Unanchored candidates are **never fetched**. An unavailable, redirected,
private, oversized or truncated authoritative feed therefore cannot hand the
episode to a mirror. Extraction fails closed.

## Ambiguity and corroboration

The authoritative candidates still go through the existing checks:

- Exact normalized title, or an episode ID on Apple.
- At most one distinct item across all anchored feeds. Relistings with the same
  GUID and enclosure collapse into one; more than one distinct item is
  ambiguous.
- Candidate caps: more than 3 (Spotify) or 8 (other routes) anchored feeds is
  ambiguous, and nothing is fetched.
- **Spotify** also requires the channel title to equal the show, and the item
  to be corroborated by publication day or the identical labelled link. A
  contradiction rejects the item.
- **Generic pages** reject an item whose `pubDate` contradicts the page date by
  more than one day.
- **Apple title-only matches** get the same rule. If the item was selected by
  exact title rather than by the `?i=` episode ID, a `pubDate` more than one day
  from the Apple page date rejects it. An ID match, or an undated page, is not
  date-checked.

### Hard contradictions before title fallback (final check H2)

A unique exact title is not enough when the item **itself** names another
episode. On the generic and Apple routes, the selected item's own `<guid>` and
`<link>` are parsed for explicit episode IDs (`_explicit_episode_ids`). IDs are
compared **only within one namespace**:

| Namespace | Recognised in | Example |
|---|---|---|
| `apple` | the single `?i=` value of an `*.apple.com` URL | `…/id999?i=1001` |
| `buzzsprout` | `(www.)buzzsprout.com/<show>/episodes/<id>[-slug]` or the short `/<show>/<id>[-slug]` | `…/111111/episodes/444444-x`, `…/111111/444444` |
| `path:<site>` | a numeric `/episode(s)|e|ep/<id>[-slug]` segment on the same site | `example.com/episodes/444-x` |

If the requested URL and the item state different IDs in the same namespace,
the item is rejected with a visible reason. Examples: `publisher feed item
names another episode on buzzsprout (444444, requested 333333)` and `Apple
show id 999: feed item names another Apple episode (?i=1001, requested
?i=1002)`. The guard does not discard the item and fall back to a weaker match.
It applies in every anchored feed, so one contradicted listing stops
resolution.

Preserved aliases (no contradiction): HTTP vs HTTPS, `www.` vs bare host,
trailing slash, tracking query strings, Buzzsprout short ID paths, an external
publisher site (another site, another namespace), no link, opaque GUIDs
(`Buzzsprout-444444` and `opaque-1` are not parsed), and Apple URLs in a generic
item, or publisher links in an Apple item. Equivalence across unrelated
namespaces is never guessed. An opaque GUID is neither proof nor contradiction.

There is also a defence in depth. A Spotify asset that the policy did not
corroborate is reduced to "unresolved" at extraction. Its tag, enclosure,
`Audio:` line, ASR and cache are never used.

Without an authoritative feed, the only publishable artifact is the
**requested page's own inline transcript** (`publisher_episode_page`). That
artifact keeps the existing episode-route and title rules. The labelled link on
a Spotify page is **not** fetched without an authoritative feed. A re-upload or
summary show (for example the F02 control) may copy it, and only an
authoritative item proves it is this episode's text. Instead the error says
`publisher transcript link advertised … link not followed`.

## Cache binding (epoch 3)

Epoch 3 is a **clean cache epoch**. Version-1 and version-2 entries are never
read automatically (see [legacy entries](#legacy-entries-v1-and-v2-never-read-automatically)).

The key is `olw-podcast-transcript-binding-v3` plus a JSON object with:

- **Episode identity.** For a feed item: feed URL, feed authority
  (`requested_feed`, `apple_lookup`, `spotify_publisher_feed` or
  `page_publisher_feed`), GUID, enclosure, normalized title and `pubDate`. For an
  unresolved page: the requested URL, normalized page title and page date.
- **The selected artifact** `[kind, url]`. `kind` is `tag` (podcast:transcript),
  `page` (inline), `link` (labelled link) or `audio` (any ASR provider).

Every payload also stores explicit provenance in `podcast_binding`:

```json
{"epoch": 3, "origin": "acquired" | "operator_import",
 "episode": {…the episode identity above…}, "artifact": [kind, url],
 "import": {…only for operator_import…}}
```

On every lookup all of these must hold, or the entry is a miss (never mutated):

1. The payload's `identity` equals the requested key identity.
2. `podcast_binding.epoch` is 3, `origin` is `acquired` or `operator_import`,
   and `episode`/`artifact` equal the **current** binding and artifact.
   An `operator_import` also needs its import record (attestation digest).
3. The payload's provider matches `kind` and its `artifact_url` equals `url`
   (an ASR payload with an empty URL is accepted for `audio`).
4. The existing normalization/admission-version checks and episode admission pass.

Two episodes that share an enclosure, or both a GUID and an enclosure, have
disjoint keys. A changed transcript-tag URL is a miss that refetches the new
artifact. The same feed item reached from another Spotify URL still reuses its
entry (CS05). Fresh results are written only under epoch-3 keys with
`origin: acquired` and the real acquisition time; a second request reuses them
(cold acquisition once, warm hit after).

### Legacy entries (v1 and v2): never read automatically

The independent final check (H1) showed that no automatic rule is safe:

- A version-1 key (`guid|enclosure`, `guid`, `enclosure` or the raw URL) names
  **media, not an episode**. Real HEAD (7158bf2) entries for episode A were
  served for episode B in eight variants: shared or reused GUID/enclosure,
  transcript-tag GUID reuse, and feed rotation in which A left the feed after
  B's `pubDate`. In the rotation cases the entry was written *after* B was
  published, so `acquired_at` ordering proves nothing.
- The former guarded v1→v2 migration copied such text under B's **v2** key
  with a fresh `acquired_at` and no migration marker. A v2 entry therefore
  cannot be told apart from a poisoned one.

So the rule is: **unbound automatic migration misses.** No GUID, enclosure,
timestamp, or current-feed-uniqueness heuristic is accepted as identity proof,
and no v1 or possibly-contaminated v2 entry is promoted into epoch 3. All old
files are kept byte-identical; nothing deletes, rewrites or renames them. The
former `_migrate_legacy_cache` was removed.

### Cost and refresh impact (deliberate)

After upgrading, every podcast transcript previously cached under v1 or v2 is
a miss **once** per episode/artifact:

| Old entry | What the first request does | Cost |
|---|---|---|
| Public tag / inline page / labelled link | Refetches the publisher artifact | Free (network only) |
| AssemblyAI / Supadata / local Whisper ASR | Runs the configured ASR chain again: AssemblyAI, then Supadata, then local Whisper | **Paid** for remote ASR; local CPU for Whisper |

After that, the fresh epoch-3 entry is reused. Already persisted source notes
are not touched by a cache miss. Current immutable callers refuse mutating
`--refresh`; use `--refresh --preview` to inspect extraction, then save reviewed
text under a new input filename without replacing an existing source. Preview
and new ingestion can acquire again; see [current callers](immutable-application-callers.md). To avoid unattended paid re-transcription, before
the first post-upgrade ingest either:

- remove `ASSEMBLYAI_API_KEY` / `SUPADATA_API_KEY` from the effective
  configuration (and leave `faster-whisper` uninstalled). Episodes that need
  ASR then fail closed with `Podcast incomplete: no transcript …`, and nothing
  is spent; or
- use the operator-attested import below for entries you have verified.

### Operator-attested import (explicit recovery)

`obsidian_llm_wiki.ingest.podcast_cache_recovery` is the only way an old entry
can be served again. It never contacts a transcription provider and never
modifies or deletes the legacy file.

1. **Describe** (read-only: metadata and feed discovery only; no transcript
   fetch, ASR or cache write):

   ```sh
   VAULT_PATH=/path/to/vault python -m obsidian_llm_wiki.ingest.podcast_cache_recovery \
     describe 'EPISODE_URL' > binding-report.json
   ```

   For each publishable artifact the report lists the exact epoch-3
   `identity`, its SHA-256 `digest`, any existing epoch-3 entry, and legacy
   v1/v2 files naming **that artifact**. Each candidate shows its file,
   SHA-256, epoch, provider, `acquired_at`, whether current admission still
   accepts it, a warning, and the exact `acknowledgement` sentence. A
   `warnings` entry flags feeds where another item shares the GUID/enclosure.
   Candidates are found by the old key formulas. That is a lead to inspect,
   **not** evidence that the text is this episode.
2. **Verify** the candidate text really is this episode, for example by
   listening to a sample or comparing it with the publisher's page.
3. **Import** one file under one binding. Save that binding's `identity` string
   to a file first (a single trailing newline is tolerated):

   ```sh
   VAULT_PATH=/path/to/vault python -m obsidian_llm_wiki.ingest.podcast_cache_recovery import \
     --legacy-file FILE --legacy-sha256 SHA256 --binding-file binding.txt \
     --acknowledgement 'I attest that legacy podcast transcript SHA256 is the artifact of episode binding DIGEST; it was not freshly acquired' \
     --operator 'NAME'
   ```

The import is refused, and nothing is written, unless all of these hold:

- the file's SHA-256 is exactly `--legacy-sha256`;
- the binding is an exact, canonical epoch-3 identity;
- the acknowledgement is exactly the sentence for that file and binding digest;
- the file name matches its stored identity, and that identity is a v1 key of
  this binding or this binding's v2 key for the same artifact;
- the stored provider/artifact equals the binding's artifact;
- current cache admission still accepts the payload. Pre-admission public
  artifacts are refetched for free instead;
- no epoch-3 entry exists yet for the binding. An import never overwrites.

`FILE` may be the default-cache entry or an operator backup copy anywhere,
as long as it keeps its cache-key file name. The file is read **exactly once**
(a regular file, read through one descriptor). That one snapshot is hashed,
checked against `--legacy-sha256` and the acknowledgement, decoded by the same
cache decoder and admission rules that normal cache reads use
(`transcript_resolver.decode_transcript_cache_bytes`), and imported. Nothing
is re-read from `FILE`, and no other cache file with the same identity is
consulted, so the attested SHA-256 always names the bytes that were imported.
If the file changes while it is being read, the import is refused. A change
after the snapshot cannot affect what is imported.

Publication is atomic and exclusive. The new entry is written and fsynced to a
sibling temporary file, then hard-linked into place, or published with
`renameat2(RENAME_NOREPLACE)` where hard links are unsupported. Either way the
call fails if anything already holds the name: a fresh entry written
concurrently, an existing file, or a dangling symlink. A refused or failed
import leaves no entry and no temporary file, and can simply be retried. Normal
acquisition (`save_transcript_cache`) still refreshes by atomic replace.

The imported entry keeps the **original `acquired_at`**. Its
`podcast_binding.origin` is `operator_import`, and its `import` record holds
the legacy epoch, identity, file name and SHA-256, the original provider and
`acquired_at`, `imported_at`, the operator and a digest of the attestation.
Extraction then serves it without ASR, but labels it every time. The content
line reads `Transcript source: PROVIDER (operator-attested import of a legacy
epoch-N cache entry; originally acquired …; imported …; not freshly acquired;
legacy sha256 …)`, and the same note is in `provenance.diagnostics`. Imported
text is never presented as freshly acquired. To undo an import, delete only
the epoch-3 file it reported. The legacy file is unchanged.

## Visible failure reasons

When a reason exists, it leads the error, before the boilerplate. This keeps
it visible after the connector's 240-character truncation, in public
`extract()` and in the CLI's JSON `source` event `error`. The machine event
schema is unchanged. Examples:

- `Podcast incomplete: no transcript (Spotify show identity: publisher feed on lexfridman.com unavailable or lacks this episode). …`
- `… (Spotify show identity: no labelled transcript link names the publisher; show name/date cannot authenticate a feed)`
- `… (no discovered feed is hosted by the page publisher www.buzzsprout.com)`
- `… no transcript with resolved Apple episode identity (Apple show id 999: lookup unavailable (HTTP 503))`

## Bounds and guards (unchanged or tighter)

| Route | Requests |
|---|---|
| Spotify | Metadata 1, Podcast Index ≤1 (only with credentials), iTunes ≤2. Feeds: anchored only, ≤3, each ≤6 hops, ≤50 MB. Transcript tag or one labelled link: ≤6 hops. No anchor link: metadata only. |
| Apple | Metadata 1, lookup 1, lookup feed 1 (≤6 hops), then the artifact. No directory requests. |
| Generic | Metadata 1, Podcast Index ≤2 (only with credentials), iTunes 1. Feeds: anchored only, ≤8. |

Private and redirect validation (`url_safety`) is unchanged and runs before
transport. In the final-review harness, the authentic S02 replay drops from 7 to 5
requests, because the acast and spreaker mirrors are no longer fetched.

## Compatibility and trade-offs (real, deliberate)

- **Spotify without one labelled transcript link fails closed** before any
  directory request. This applies even when an exact-name feed has a same-day
  item. Previously name plus date was enough, and ASR could run. Use the
  episode's RSS feed URL, Apple URL or transcript URL instead.
- **Spotify whose publisher feed is on a different host** than its transcript
  link fails closed. This covers a site on its own domain with a feed on
  Libsyn or Megaphone.
- **Generic pages whose feed is not on the page's host**, or not on a tenant
  matching the platform, fail closed. For example, an `example.com` episode
  page with a `example.libsyn.com` feed, or an `anchor.fm/<show>` page with an
  `anchor.fm/s/<hash>` feed. Previously the first title match won, which is
  how AG01 and AG02 returned the wrong show.
- **Apple without a lookup feed** (outage, or no `feedUrl`) fails closed.
  Podcast Index is never consulted for Apple.
- Same-name mirrors no longer make a valid Spotify episode ambiguous. SP03 and
  SP13 now resolve the canonical feed, where they used to fail closed.
- Two episodes sharing one enclosure each pay for their own ASR (CB06 and
  CB07).
- **Epoch 3:** no v1 or v2 podcast transcript entry is reused automatically,
  including the formerly migrated "uniquely keyed" ones. Each pays one fresh
  acquisition (paid ASR where configured) unless an operator imports it with
  an attestation. See [cost and refresh impact](#cost-and-refresh-impact-deliberate).
- An item whose own GUID/link names another same-namespace episode ID, or an
  Apple title-only item contradicting the page date, now fails closed where it
  was previously admitted.
- `EpisodeAsset` gained `feed_url`, `authority`, `media_shared`, `link`,
  `matched_by` and `contradicted`, all with `compare=False`.
  `_anchored_matches` returns `(matches, ambiguous, contradiction)`.
  `_find_apple_episode_asset` takes keyword `published=`. The podcast cache
  seam is now `load_transcript_cache_record`, which returns the result and its
  payload. The first-match helpers `_find_asset_via_podcast_index` and
  `_find_asset_via_itunes` were removed.

## Not covered

- Host anchoring trusts DNS control of the anchor host. A compromised
  publisher host, or an exact-tenant collision on a shared platform, is out
  of scope.
- The shared-platform list is finite. An unlisted multi-tenant host is treated
  as single-tenant: same host or subdomain only. This is stricter than
  registrable-domain matching.
- A publisher-labelled link on a Spotify page is never fetched without an
  authoritative feed, even when it is genuinely the publisher's.
- Contradiction namespaces are a short, explicit list. An item that names
  another episode only through an unrecognised URL scheme or an opaque GUID is
  not caught. The exact-title, authority, ambiguity and date rules still apply.
  Spotify items are not ID-checked, because Spotify pages state no comparable ID.
- An operator import trusts the operator's attestation. The tool proves that
  the file, binding and artifact are exactly those named, but it cannot prove
  the spoken content.
- Import exclusivity only holds against other writers. A fresh acquisition
  that started before an import and finishes after it still refreshes the
  key by atomic replace (normal refresh semantics) and supersedes the import.
  Change detection during the read compares size, mtime and ctime. A
  same-size rewrite within one timestamp tick is not detected, but the
  imported bytes are still exactly the bytes hashed and attested. Atomic
  no-overwrite needs `link(2)` or Linux `renameat2`; on a filesystem with
  neither, for example some network mounts, the import is refused rather than
  published non-atomically. Durability of the directory entry is best-effort
  (directory fsync), and the cache directory itself is trusted, not
  symlink-checked.
- Feed rotation that keeps the same GUID and enclosure for a different
  episode *within epoch 3* (the item's title and `pubDate` also changing)
  produces a different binding and therefore a miss, not reuse. A publisher
  that silently replaces audio while keeping every item field is out of scope.

Regression coverage: `tests/new/test_podcast_authority_binding.py` and
`tests/new/test_podcast_binding_epoch.py` (epoch 3, legacy/poisoned-v2 misses,
operator import, H2 contradictions and aliases), and
`tests/new/test_podcast_import_integrity.py` (import byte binding, shared
decoder, atomic no-overwrite publication). The updated
contract assertions are in `test_podcast_show_feed_discovery.py`,
`test_podcast_index.py`, `test_live50_publisher_transcript_links.py`,
`test_live50_review_b_closure.py`, `test_live50_podcast_integrity.py` and
`test_transcript_resolver.py`.
