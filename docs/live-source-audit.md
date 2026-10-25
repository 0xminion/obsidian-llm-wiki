# Live 50-URL extraction and synthesis audit

## Scope and baseline

The baseline is `ca0f99b48e4c365800effc792507a1b065b0e92c` (PR #22). The [committed manifest](../tests/fixtures/live50/manifest.json) contains **50 distinct original URL strings**, including the supplied X post and ScienceDirect paper. One fragment variant intentionally repeats the same essay for duplicate testing; this is not a claim of 50 independent works.

Every URL was attempted through the actual public `extract()` entry point in a disposable environment. Source text was not substituted from search snippets. Test conditions prohibited credentials, browser cookies, proxies, paid transcription, local audio/model downloads, and real-vault writes. The initial harness used at most three processes and a 100-second per-case process-group deadline. No case hit that deadline. Hosted extraction rate limits and origin restrictions are recorded as failures, not proof that content does not exist.

| Category | URLs | Baseline SourceDoc returns | Baseline errors |
| --- | ---: | ---: | ---: |
| Medium | 6 | 1 | 5 |
| X/Twitter | 5 | 5 | 0 |
| PDF, including a Drive viewer | 6 | 4 | 2 |
| DOC/DOCX/Google Docs | 5 | 2 | 3 |
| Scholarly/general articles | 8 | 7 | 1 |
| YouTube | 8 | 1 | 7 |
| Apple Podcasts | 5 | 5 | 0 |
| Spotify | 5 | 1 | 4 |
| Podcast RSS | 2 | 0 | 2 |
| **Total** | **50** | **26** | **24** |

**A returned SourceDoc was not automatically a successful extraction.** Five media returns contained descriptions/show notes rather than transcripts; both genuine caption/transcript returns had reproduced text-loss bugs. The Medium return came from an archive but incorrectly claimed live-origin provenance.

The supplied X URL returned a substantive 22,315-character article. The supplied ScienceDirect URL returned HTTP 403. Separately, the user-attached ScienceDirect capture contained an abstract and references but no main body; its 37,786 characters exposed a false-completeness acceptance bug. Attachment replay is distinguished from live URL extraction.

## Confirmed defects and fixes

| Defect | Root cause / affected implementation | Regression coverage |
| --- | --- | --- |
| Incomplete scientific papers accepted | Generic heading/length checks counted title, navigation, references, and nested structured-abstract headings as body sections. The gate now checks actual body sections, including conventional English/Chinese plain-text headings, and respects abstract heading hierarchy. | `test_live50_scholarly_quality.py`, `test_live50_review_a.py` |
| Quality failure prevented useful fallback | `extract_web()` selected the first nonempty candidate before the outer gate rejected it. Each candidate now passes acceptance before selection; earlier bounded diagnostics survive. | `test_live50_web_fallback_quality.py` |
| Empty/incorrect hosted frontmatter | Generic parsing required a trailing newline; Twitter split at `---` inside quoted titles. Both paths use the canonical safe YAML parser. Empty bodies are rejected inside the extraction strategy. | `test_live50_web_provenance.py`, `test_live50_review_a.py` |
| X article-card previews accepted | Specialist length exceptions and first-error title diagnostics masked the preview check. Preview recognition is independent of title validation; legitimate short posts remain valid. | `test_live50_twitter_quality.py`, `test_live50_review_a.py` |
| Missing retrieval provenance | Archive and hosted extraction lost actual artifact URLs/winning stages. Archive snapshot identifiers, hosted URLs and selected stages are retained separately from the requested source identity. | `test_live50_web_provenance.py`, `test_live50_review_a.py` |
| Valid PDF rejected as corrupt | Request headers advertised Brotli without an installed decoder. Standard requests now negotiate supported gzip/deflate, without a new dependency. | `test_live50_document_downloads.py` |
| Public Drive file remained a viewer stub | No connector followed the explicitly advertised official download. Exact public file-viewer routes now use anonymous, identity-bound, redirect-validated acquisition with MIME/signature and size/time limits. No login, confirmation, folder/editor, or arbitrary Google-host expansion is allowed. | `test_live50_document_downloads.py` |
| Transcript HTML silently lost prose | Entity decoding preceded regex tag removal, turning literal `<` into markup. HTML parsing now preserves literal text and entities. | `test_live50_transcript_fidelity.py` |
| Genuine repeated captions removed | Global text deduplication collapsed speech at distinct cue times. Cue normalization only removes adjacent redisplays with positive timestamp overlap and matching voices; touching/gapped cues remain distinct. | `test_live50_transcript_fidelity.py` |
| Old damaged public transcripts stayed cached | Cache entries lacked normalization versioning. Old `rss_podcast_transcript` artifacts are cache misses; current normalized results round-trip. Unaffected legacy ASR caches remain reusable. Reading an old entry does not delete it. | `test_live50_transcript_cache_version.py` |
| Wrong/ambiguous podcast matching | Substring title/ID matches could win before an exact identity. Exact identity and normalized exact title precede explicit platform/show-suffix rules; ambiguous matches fail closed. | `test_live50_podcast_integrity.py` |
| Description/show-only ingestion appeared complete | Any SEO description counted as episode content; a long audio URL could conceal a tiny truncated description. Episode ingestion now requires a transcript. Intentional manual clipping remains separate. | `test_live50_podcast_integrity.py` |
| RSS ignored available transcript-bearing items | The documented first transcript-bearing policy actually selected the first item. Feed selection now honors transcript availability in publisher order; no-tag feeds retain their existing cache/ASR behavior. | `test_live50_podcast_integrity.py` |
| Publisher transcript link ignored | Only inline headings/RSS transcript tags were recognized. Explicit labelled links on a uniquely matched episode can now resolve one public artifact with identity, transcript-content and URL-safety checks before ASR. | `test_live50_publisher_transcript_links.py` |
| Podcast artifact provenance lost / duplicate requests | Source construction discarded the actual transcript artifact, and `metadata or fetch()` repeated requests after an empty result. Provenance is retained; supplied empty metadata is not fetched again. | `test_live50_podcast_integrity.py` |
| Full synthesis schema rejected by real provider | Generic nested object schemas lacked closed-object properties and complete required-field lists. A canonical fully specified wire schema now matches the prompt/parser contract. Empty lists/strings remain valid; semantic validation is unchanged. Cache schema version is `synthesis-schema-v3`. | `test_live50_provider_schema.py`, `test_structured_synthesis_requests.py` |

## Post-fix live rerun: text and documents

The exact original **30 text/document URLs** were rerun serially with fresh caches: **21 returned, 9 errors**, versus 19/11 initially. No timeout or harness error occurred. The two newly successful cases are the public Drive PDF and the compressed PDF. All 21 returned bodies were reviewed; the supplied X body was byte-identical to its first capture. Remaining failures were five Medium pages, two Google Docs editor pages, the `.doc` conversion dependency, and the ScienceDirect HTTP 403.

This live batch used an immutable private source snapshot (`a439abe098c2630c25053c8dd67ae68527ccfb53ff1b65853f7148cf9a8c123b`). It predates the final Twitter frontmatter/provenance and structured-abstract closure fixes. Those follow-ups were independently exercised against captured inputs and negative controls, not misrepresented as part of this earlier live batch. Media results are reported separately.

## Post-fix live rerun: media

All **20 original media URLs** were also rerun serially with fresh caches: **4 fidelity-verified transcript/caption returns and 16 explicit failures/rejections**, with no timeouts or HTTP 429s in that run. The returns cover three distinct payloads because the Apple episode and RSS feed select the same transcript. The remaining outcomes were nine access failures, five description/show-only rejections, one missing public artifact and one unresolved episode identity.

The live checks preserved the missing HTML span, all 61 YouTube cues/90 payload lines, the RSS transcript-bearing item at publisher index 17, and all 781 segments of the explicitly linked Spotify publisher transcript. Access to seven YouTube cases remained blocked; Dwarkesh feeds returned 502. S02 still did not establish canonical episode identity despite an advertised transcript link; it failed rather than publishing unverified show notes.

Across these two live retest snapshots, **25 of 50 URLs returned content**. The lower raw return count versus baseline is not a regression target: five old metadata-only successes are now rejected, while complete PDFs and real transcripts have been recovered. [Recorded outcomes](../tests/fixtures/live50/outcomes.json) retain every original URL and before/after result. Subsequent independent ReviewB found inline/RSS admission bypasses on additional adversarial cases; this live corpus run is not an approval of those paths or of later fixes.

## First representative live-synthesis experiment

Eight actual corpus captures reached the real subscription-backed model with the fixed production schema: **14 requests, all HTTP 200 and schema-valid; 3 compiled and 5 were rejected by exact-evidence validation**. This was real synthesis over captured inputs, not eight fresh source fetches or fifty completed builds.

| Source | Outcome |
| --- | --- |
| T07, supplied long X article | Rejected: changed Markdown/citation text and capitalization inside quotes. |
| T08, shorter X post | Compiled; 5 exact verified quotes. |
| T26, Chinese article | Rejected: removed emphasis or joined separated source passages. |
| T14, Chinese PDF | Rejected: changed source line breaks/spacing inside quotes. |
| T24, arXiv | Main chunk rejected; bibliography-only chunk did not cause false whole-source completion. |
| T28, essay | Rejected: omitted an inline Markdown link from the quoted span. |
| A03, repaired podcast | Compiled; 13 exact verified quotes. |
| T03, archived Chinese Medium article | Compiled; 10 exact verified quotes. |

Successful sources preserved English/Chinese prose and all 28 published quotes matched persisted source offsets/hashes. For these three successful sources, unchanged and human-annotation follow-ups made **zero attempted/transmitted model calls and zero Markdown writes**; annotations survived exactly. Metrics, SQLite metric records and transient locks still changed. An unchanged build is **not a read-only filesystem operation**; operational telemetry was not removed to satisfy an unnecessarily broad zero-write assertion.

A freshness view exposed a genuine bare-stem ambiguity between `entries/NAME.md` and `sources/NAME.md`. Both ordinary and sharded freshness rows now use `[[sources/NAME|title]]`, with collision regressions. Same-directory index links are a distinct case; interactive Obsidian UI resolution was not tested.

The rejected T07 response also contained an unsupported inference about which agent performed work. It was not published. Exact quote presence does not prove entailment; source-aware review remains necessary. Later prompt experiments are separate evidence and do not erase these initial failures.

## Bounded quotation-prompt follow-up

The shared prompt now explicitly requires raw Markdown/citation/case/whitespace preservation and a shorter unique contiguous span only when it supports the entire claim. Unsupported actor/contribution/causal inferences remain forbidden. Prompt compatibility is `synthesis-v4`; evidence resolution and semantic acceptance code were not relaxed.

A separately budgeted experiment used **7 new requests** on four of the five previously rejected sources. T26 compiled on its built-in retry and T28 compiled; T07 and T14 still failed. T24 was explicitly not rerun because its worst-case two-chunk retry reserve exceeded the remaining request budget. Original inputs and failed responses were retained unchanged.

The two compiled v4 cases passed same-version unchanged and human-edit checks with zero provider attempts/transmissions and zero Markdown writes. A deeper review nevertheless found that some T26 attached quotes did not support every detail of their claims, even though those details existed elsewhere in the source. **Only T28 passed both compilation and that deeper review.** Literal quote matching proves an occurrence, not entailment. The supplied X article and Chinese PDF still have unresolved model quotation failures; automatic publication is not a substitute for review.

The next quality step is a tested source-span citation and claim-support review design, not additional prompt wording or silently stripping source formatting. This is remaining work, not an implemented guarantee.

## Operational changes and recovery

- **Back up before upgrading** using the [migration guide](migration.md). No migration of a real vault was performed in this audit.
- Existing source files do not become fresh merely because extractor code changes. To reacquire a known URL, use `olw ingest VAULT --url URL --refresh --skip-synthesis`, inspect the source/provenance, then `olw build VAULT`. Normal source backup and user-edit protection still apply.
- The synthesis schema version change makes incompatible cached synthesis eligible for regeneration; a subsequent build may call the configured model. Do not interpret an unchanged source hash as a promise of zero calls across a schema upgrade.
- Description-only podcast results that used to succeed now fail with an actionable missing-transcript diagnostic. Use an episode URL with a public transcript, explicitly configure an authorized transcription provider, or deliberately import a labelled description as a manual clipping. Do not relabel show notes as a full transcript.
- Public Drive support is deliberately narrow: `https://drive.google.com/file/d/ID/view` with an optional `usp` query and an advertised matching download. It does not authorize access to private files, confirmations, arbitrary redirect destinations, folders, or Google Docs editor pages.
- A legacy `.doc` case reached LiteParse but needed **LibreOffice**, which was absent. This optional conversion dependency was not installed automatically. A missing executable on PATH is distinguished from a missing installation.
- Public publisher-link discovery is conservative. Unsupported layouts, ambiguous links, wrong-episode targets and monologues without adequate transcript markers may be rejected. The shared transcript resolver currently retains the advertised artifact URL, not a separate terminal-redirect URL.
- [Podcast admission](podcast-admission.md) now applies across inline, RSS-tagged, labelled and cached transcripts. A uniquely selected RSS title is authoritative; explicit conflicting artifact identity, ambiguous episodes and show-only routes cannot escape through weaker fallbacks. Public caches also need `podcast_admission_version: 1`; older entries miss without being deleted. Normalization remains version 2 and legacy ASR caches are not invalidated.

## Verification boundaries

The baseline suite returned **1,761 passed, 1 skipped**. The skip was a PATH issue: the installed LiteParse CLI integration passed when the project virtual environment was placed on PATH.

Final locked matrix, with the actual LiteParse integration included:

| Python | Passed | Failed / errors / skipped | Seconds |
| --- | ---: | --- | ---: |
| 3.11 | 2,068 | 0 / 0 / 0 | 156.627 |
| 3.12 | 2,068 | 0 / 0 / 0 | 167.863 |
| 3.13 | 2,068 | 0 / 0 / 0 | 155.365 |

All package/test Python file hashes stayed unchanged across the final matrix. Earlier runs with 2,067 passes and one obsolete `/episode` preview-fixture failure are retained; the positive fixture now uses an explicit `/episodes/42` route with its no-write assertions intact. Ruff, lock validation and `git diff --check` passed.

A final wheel was built and installed outside the checkout. All **90 installed Python modules match the checkout byte-for-byte**; CLI help, schema/version/module imports and incomplete-paper rejection were exercised. Wheel SHA-256: `759f4627d99d9d828f8c7752d132e2d9d0e083661d953f22246efff158aa86c6`. It retains package version 3.0.0 and is not a released package or a new version tag. Mermaid 11.12.2 parsed all six diagrams from the first three successful builds and all four from the prompt follow-up; malformed controls failed. This verifies syntax, not Obsidian GUI rendering.

Real capture replays recovered the missing HTML prose and all 61 VTT cues/90 text lines. Fresh public document checks retrieved the Drive PDF (24 pages) and the formerly Brotli-encoded PDF (174 pages), preserving every nonempty native-text page. The explicit publisher transcript check retained all 781 timestamps. These are text-preservation checks, not OCR, image, chart or audio-accuracy certification.

A tiny provider preflight passed while the original production schema failed HTTP 400. The fixed production schema passed a genuine provider-backed pipeline request. Another schema-valid response was correctly rejected for non-verbatim quotes. Schema acceptance is therefore reported separately from parsing, semantic acceptance and generated-note quality.

The initial independent review found five failing assertions after the first local fixes. Those exact failures were reproduced and fixed; their historical FAIL receipts remain intact. Review A independently reran its unchanged original probes: **8/8 minimal cases, 60/60 independent checks, and 216/216 relevant repository tests passed** with stable production hashes. That approval covers document/web/Twitter acceptance and provenance, not the entire repository. Independent Review B subsequently closed all six podcast defects: its unchanged original replay passed 27 cases with one deliberately obsolete public-cache-hit expectation; changing only that expectation to a miss passed 28/28, including fresh acquisition and cache reuse. Its eight actual CLI controls passed, with unsafe cases failing and writing no durable sources. Original capture hashes and canonical-success body/provenance were preserved. These scoped approvals do not certify the unresolved model entailment problem.

## Reproduction and evidence

```bash
uv sync --locked --extra dev --extra pdf --extra docx --extra liteparse --extra youtube
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check obsidian_llm_wiki tests scripts benchmarks
git diff --check
```

`tests/fixtures/live50/manifest.json` is data, not an instruction to fetch URLs during ordinary CI. Network audits need fresh disposable HOME/cache/vault directories, explicit credential isolation, process deadlines, bounded concurrency and separate result files. Reusing a saved response is a **capture replay**, not a live network success. Do not retry access restrictions aggressively or purchase transcription just to make a table green.

Detailed local receipts live under the audit host's scratch directories named `wiki-live50-text`, `wiki-live50-media`, `wiki-live50-parent`, `wiki-live50-document-fixes`, `wiki-live50-transcript-fixes`, `wiki-live50-podcast-fixes`, `wiki-live50-publisher-links`, `wiki-live50-schema-fix`, and the independent review directories. They include manifests, exact commands, hashes, per-case errors and captured-response replay harnesses. Raw source captures and credentials are not committed to this repository. Scratch evidence needs archiving for long-term retention.

This audit does not certify all future publisher layouts, provider behavior, native platforms, an Obsidian GUI session, or million-note capacity.
