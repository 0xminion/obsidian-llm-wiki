# Data model and identity contract

Active reference. Source definitions: [`core/models.py`](../obsidian_llm_wiki/core/models.py), [`core/registry.py`](../obsidian_llm_wiki/core/registry.py), [`core/cache.py`](../obsidian_llm_wiki/core/cache.py), and [`synth/dedupe.py`](../obsidian_llm_wiki/synth/dedupe.py).

## Layers and authorities

- **Source Markdown** contains input text and extraction metadata. `SourceDoc` is the decoded, frontmatter-stripped representation used for synthesis. The body is stripped at its outer whitespace boundaries; metadata changes and body changes are tracked differently.
- **Per-source synthesis** is structured intermediate knowledge (`SourceSynthesis`), persisted in SQLite and a portable JSON cache. It contains source summary/tags/key points/open questions, language, concepts, MoCs, and source filename.
- **Merged bundle** (`SynthesisBundle`) is the in-memory union of live source syntheses. It is not a separate authoritative source document. Rendering consumes the entire bundle on a changed build.
- **Registry** indexes runtime identities, ownership, version compatibility, search, queues, and render state. It does not make generated Markdown authoritative over human edits.
- **Generated notes** are a materialized view with human-edit protection. A protected note may diverge from the synthesis registry. Queries reading Markdown can therefore see a different view than synthesis candidate retrieval reading SQLite.

## Identities

| Entity | Current identity | Important boundary |
| --- | --- | --- |
| Source | SQLite integer `source.id`; unique source filename within the flat source directory | Renaming a source file is not automatically a stable-ID move: it can be interpreted as deletion plus addition. URL canonicalization/content hashes serve ingest dedupe, not universal document equivalence. |
| Source revision | Source ID + body content hash, with extractor/provenance metadata | A reverted body can have the same content hash as an earlier revision while metadata differs. Do not infer current metadata from the oldest matching hash. |
| Synthesis | One row per source ID with input hash and artifact versions | A portable cache file alone does not certify a completed render/checkpoint. |
| Concept | `canonical_id`, initially `concept-` plus the first 24 hex characters of SHA-256 of `"concept\0" + slug` | Existing locally known IDs are reused; arbitrary model-invented IDs are not trusted as established identity. IDs are not globally unique semantic definitions. |
| Concept path | Normalized human-readable `slug`, used by filenames and wikilinks | Can change during title/slug normalization. A filesystem-safe name does not prove two meanings are the same. |
| MoC | Registry ID derived from `"moc\0" + slug` by the same stable-ID scheme | The `MapOfContent` dataclass itself has no durable `canonical_id` field; arbitrary MoC rename continuity is not implemented. |
| Entry | Source-file-derived output stem, with normalization/hash fallback for unsafe or long names | Two sources with equal display titles need not overwrite each other's entries. |
| Graph node | Type-qualified/collision-safe export identity | Export IDs, canonical IDs, and human-facing link slugs are different namespaces. Do not assume a graph ID is a Markdown path. |

## Matching and merge rules

1. **Retrieval is not identity.** Bounded FTS/ANN candidates expose known IDs/slugs/titles to the model. An alias is a search hint, not an independently validated equivalence decision.
2. **Known canonical ID wins.** Registry resolution accepts a supplied ID only if it exists locally, then uses that row's canonical slug. A wrong but existing ID supplied by the model remains a semantic risk; there is no general entailment proof of identity.
3. **Otherwise compare title/definition and prior source ownership.** New implicit reuse requires the same normalized title and summary (definition), or an existing ownership relationship for this source. A different definition—even with the same title—gets a bounded deterministic fingerprint suffix. Pending in-memory reservations separate first-batch collisions before commit. A unique same-title match among this source's indexed owned concepts preserves a previously disambiguated ID across source revisions. These are conservative deterministic rules, not semantic truth: paraphrases may split, while an owned definition can change meaning without changing ID.
4. **Same-slug merge is mechanical.** `merge_concepts` uses a deep copy, first occurrence as base, unioned tags/aliases, appended sections/claims, deduplicated related targets, maximum confidence, and `is_new=False` if any contributor says false. MoC identity is resolved first: across sources, reuse requires the same normalized title, detected language, and overlapping canonicalized members. Disjoint same-title topics or different-language summaries get disambiguated slugs; in-batch reservations protect first-build collisions. A unique prior source-owned MoC with the same title retains its slug across revisions even as membership changes. Only then do same-slug MoCs union members. This is not semantic conflict resolution; repeated sections, inconsistent claims, or same-label different meanings require review.
5. **Cross-slug “semantic dedupe” needs additional authority.** `semantic_dedupe_concepts` requires a shared nonempty canonical ID or explicit `confirmed_merges` victim→survivor mapping, in addition to its same-language/similarity filters. Equal titles or cosine similarity alone do not permit destructive merge. The ordinary pipeline does not pass a human confirmation map; there is no CLI merge-approval command.
6. **References must follow an authorized merge.** The merge routine flattens transitive redirects and remaps source-local concepts, MoC membership, relationship targets, and claim concept slugs; self-edges and duplicates are removed. Persisted source-local ownership matters: changing only the visible merged bundle can resurrect old victims on the next build.

`concept_alias` is a normalized alias→concept lookup, not a many-to-many ambiguity model. A reused alias may be reassigned. Neither alias equality nor a stable hash is proof of semantic sameness.

### Typed links versus similarity

A `ConceptLink` contains target `slug`, `relation`, and display label. Known relation types are normalized. The renderer serializes links in two ways:

```yaml
relations:
  - "learning-rate|depends_on|Learning Rate"
```

```markdown
[[learning-rate|Learning Rate]] — `depends_on`
```

The flat frontmatter list avoids nested-object warnings in Obsidian Properties. In diagrams, fenced text is illustration; live wikilinks are rendered outside fences.

Backlink propagation creates a reverse edge when absent. It uses the implementation's inverse mapping (`depends_on`/`prerequisite_of`, `component_of`/`part_of`), keeps listed symmetric labels, and falls back to `related_to` where no inverse is represented. This mapping is limited by the schema; it is not a universal ontology. Do not describe every edge as having the same relation in both directions.

Cross-language cosine matches are separately versioned candidates. The pipeline resolves the human policy below to the renderer's `confirmed_cross_lingual_pairs`, independently of embeddings; scores alone do not authorize counterpart membership. Explicit source-generated MoC membership and typed crosslinks also work without embeddings. Embedding-based orphan-to-MoC assignment is a separate optional topic heuristic, not an identity merge.

## Human-confirmed equivalences

The optional policy is **`04-Wiki/.llmwiki/equivalences.json`**, not vault-root `equivalences.json`. It is a human assertion of equivalent meaning, not an embedding export, translation request, rename, or destructive merge approval. Review both definitions, source owners, and scope before asserting a pair; matching titles/similarity are insufficient, especially for homonyms.

A valid empty policy is:

```json
{"schema_version": 1, "equivalent_concepts": []}
```

Each nonempty list item must be a two-element array of **distinct existing** `concept-` + 24 lowercase hexadecimal IDs. The top-level keys must be exactly those shown; comments/extra metadata are rejected. Duplicate/reversed pairs normalize to one undirected pair. Limits are 1,000,000 bytes and 10,000 input pairs. This is pairwise assertion, not transitive equivalence-class merging.

### Discover IDs, do not derive them from titles

Current generated concept frontmatter includes `canonical_id`. Legacy/protected notes may lack it or retain stale metadata; do not edit their IDs to force a match. Prefer the current registry on the stopped-writer **rehearsal copy** created in [migration](migration.md). This snippet lists IDs, slugs, definitions and source ownership without invoking providers or the migrating `VaultRegistry` constructor:

```bash
TARGET="$REHEARSAL" python - <<'PY'
import json, os, sqlite3
from pathlib import Path
p = Path(os.environ['TARGET']).resolve() / '04-Wiki/.llmwiki/registry.db'
assert p.is_file(), 'Migrate the rehearsal copy first'
c = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True)
c.row_factory = sqlite3.Row
try:
    for row in c.execute('SELECT id, slug, title, summary FROM concept ORDER BY slug'):
        item = dict(row)
        item['sources'] = [r[0] for r in c.execute(
            'SELECT s.filename FROM concept_source cs JOIN source s ON s.id=cs.source_id '
            'WHERE cs.concept_id=? ORDER BY s.filename', (item['id'],))]
        print(json.dumps(item, ensure_ascii=False))
finally:
    c.close()
PY
```

`mode=ro` prevents DB content writes but SQLite WAL handling can create sidecars; use the disposable copy for inspection, not a claimed zero-write production probe. An orphaned/missing concept is not a valid active assertion target.

### Assert a reviewed pair on the copy

Set `LEFT_ID` and `RIGHT_ID` to two exact IDs from the listing (not slugs). Then run this snippet with writers stopped. It preserves existing valid pairs, verifies both registry IDs, validates the candidate policy, and replaces only that policy file atomically. Keep the full snapshot; record human rationale outside this strict-schema file.

```bash
# Set these yourself after reviewing both definitions and their sources:
# export LEFT_ID='concept-<24 hex characters copied from the listing>'
# export RIGHT_ID='concept-<24 hex characters copied from the listing>'
TARGET="$REHEARSAL" python - <<'PY'
import json, os, sqlite3, tempfile
from pathlib import Path
from obsidian_llm_wiki.core.identity_policy import load_equivalences
root = Path(os.environ['TARGET']).resolve() / '04-Wiki/.llmwiki'
policy = root / 'equivalences.json'
assert not root.is_symlink() and not policy.is_symlink(), 'Unexpected policy symlink'
left, right = os.environ['LEFT_ID'], os.environ['RIGHT_ID']
pairs = [list(pair) for pair in load_equivalences(policy)]
c = sqlite3.connect((root / 'registry.db').as_uri() + '?mode=ro', uri=True)
try:
    for identity in (left, right):
        assert c.execute('SELECT 1 FROM concept WHERE id=?', (identity,)).fetchone(), identity
finally:
    c.close()
pairs.append([left, right])
fd, name = tempfile.mkstemp(dir=root, prefix='equivalences-', suffix='.json')
temporary = Path(name)
try:
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump({'schema_version': 1, 'equivalent_concepts': pairs}, stream)
        stream.write('\n')
    validated = load_equivalences(temporary)
    os.replace(temporary, policy)
    print(f'Policy saved: {len(validated)} unique pair(s); not yet rendered')
finally:
    temporary.unlink(missing_ok=True)
PY
olw build "$REHEARSAL" --refresh-render
```

The final build is **not a no-provider guarantee**: current compatible sources reuse cached synthesis, but changed/incompatible ones may call the configured provider. Check its exit status and review concept/MoC files. A policy edit participates in the checkpoint, so a normal unchanged-source build also notices it. Syntax/format errors abort; unresolved IDs add errors and prevent a successful checkpoint, but other valid work may still render—policy application is not all-or-nothing publication.

Resolved pairs add symmetric `equivalent_to` links to the render bundle and authorize direct counterpart MoC membership **with embeddings disabled**. Titles/prose and canonical IDs remain separate; no concepts are merged. This is derived render policy, not an automatic rewrite of per-source synthesis ownership/FTS. Removing a pair retracts its policy-derived effect on the next reconciliation, unless another source/explicit link supplies it or human-edit protection preserves an older page. There is no dedicated CLI pair-approval or destructive-merge command.

## Provenance and evidence

`SourceProvenance` carries requested, resolved, and extracted URLs; extractor chain; MIME/format; retrieval timestamp; extracted-content SHA-256; and bounded diagnostics. Rendered source frontmatter stores readable flat strings; the loader also accepts legacy nested provenance. This is extraction history, not an assertion that the whole requested article was available or correct.

A `Claim` has text, `concept_slug`, `source_ref`, and optional `EvidenceSpan`. Evidence carries the quote, source filename, source-body hash, verification state, and character offsets. Offsets are **Unicode/Python string offsets**, not UTF-8 byte offsets. A unique exact match is verified; repeated occurrence is ambiguous; missing occurrence is unmatched; legacy evidence can remain unverified. Completed synthesis admission re-resolves supplied quotes rather than trusting model-provided offsets/status.

Only accepted per-source text should be treated as evidence. Mechanical merge preserves source-local material separately, but not every section has paragraph-level provenance. The contradiction store records evidence/revisions and a human-governed status lifecycle; it does not autonomously adjudicate truth.

## SQLite and portable files

All paths below are relative to `04-Wiki/.llmwiki/`.

| Store/table group | Purpose |
| --- | --- |
| `registry.db`: `meta` | Schema version, migration markers, checkpoint, maintenance/semantic namespace metadata. |
| `source`, `source_version`, `source_file_state`, `synthesis` | Source/current revision metadata, body/version validity, committed signatures, structured synthesis. |
| `concept`, `concept_alias`, `concept_fts`, `concept_fts_identity` | Canonical concepts, aliases, FTS5 documents; indexed FTS rowid mapping avoids deleting by a full scan of an unindexed concept ID. |
| `concept_source`, `moc_source`, `membership_source`, `membership` | Source-level ownership and derived MoC membership; deleting one owner must not delete another owner's contribution. |
| `embedding`, `ann_bucket`, `cross_lingual_link` | Versioned vectors, LSH candidate index, and cross-language candidates. ANN has entity-leading deletion indexes as well as bucket lookup indexes. |
| `dirty_queue`, `operation`, `operation_archive` | Leased retry/invalidation work and ingest history. Bounded live history is not bounded total archive size. |
| `render_state`, `health_finding`, metric tables | Render manifests, persistent maintenance findings, live/archive metrics. |
| `state.json` | Compatibility/source compilation hashes and owned concept slugs. Still read/written on non-fast-path builds; not replaced by a fully local dirty graph. |
| `cache/<source filename>.json` | Portable synthesis envelope. `_resynthesis_overlay.json` holds coherent concept resynthesis separately. |
| `equivalences.json` | Optional schema-1 human canonical-ID pairs; checkpointed render/link policy, not inferred similarity or destructive-merge approval. |
| `render-page-hashes.json` | Last recognized generated content for edit protection. Missing/changed hashes make existing notes conservative, not disposable. |
| `backups/`, graph/report/transcript files | Recovery copies and derived/cached artifacts with different retention/persistence rules. |

SQLite is opened in WAL mode with foreign keys enabled and `synchronous=NORMAL`. Markdown and DB commits are not one atomic unit; back up the **whole vault with writers stopped**, including DB/WAL/SHM, rather than copying only `registry.db` during activity.

## Version namespaces

These are independent numbers/labels, not a single “version 3” contract:

| Contract | Current value / meaning |
| --- | --- |
| Package | `3.0.0` in `pyproject.toml`. |
| SQLite physical schema | **6**, in `meta.schema_version`; opening a newer unknown schema fails. Opening an older supported schema can write migrations immediately. |
| Legacy ownership migration | Pre-v3 registry data backfills source/concept/MoC/membership ownership. |
| v5 search migration | Adds/backfills `concept_fts_identity`, cleans superseded FTS rows, clears the pipeline checkpoint. |
| v6 queue migration | Adds lease-token fencing and durable successor payload/due-time fields; preserves a running payload while newer invalidations wait. |
| Portable synthesis cache envelope | `_CACHE_ARTIFACT_SCHEMA_VERSION = 2`. |
| Extraction artifact label | `extract-v2`. This invalidates compatibility; `build` does not itself re-download a URL. |
| Synthesis schema label | `synthesis-schema-v2`. |
| Prompt compatibility | `synthesis-v3:<effective-settings hash>`; includes model routing, language, synthesis mode, candidate/evidence/chunk settings, and schema policy. |
| Rendering fingerprint | `render-v4` inside renderer fingerprint calculation. |
| Pipeline checkpoint | `integrity-v2` plus artifact versions, selected non-secret build settings, and normalized human equivalence pairs. |

Embedding identity additionally includes model, dimensions/input/version information. Do not reuse vectors across an unverified model namespace change. Metadata-only identity repair must not relabel stale synthesis as current. Legacy import deliberately uses `legacy` version labels so the normal build can regenerate incompatible output.

See [migration](migration.md) for supported operational steps, and [scaling](scaling.md) for what these indexes do **not** yet make incremental.
