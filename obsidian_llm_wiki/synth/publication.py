"""Final publication plan: bind reviewed facts to the entity a page publishes.

Review receipts bind each unit to the subject/label it was reviewed under, in
ONE source. After that, cross-source canonical merges, relation targets
(slugs proposed by a model), MoC membership and app-derived structure decide
what a page finally shows. This module is the last deterministic gate before
rendering. It never calls a model and never pretends a review happened:

* **Subject binding** — every contribution merged into a concept page must
  be bound to that page's entity AND be publishable under its title:

  - *identity*: when the pipeline supplies ``identity_evidence`` (the basis on
    which the registry bound each ``(source, entity)`` pair: ownership,
    identical definition or an explicit identity assessment), a page with
    several contributing sources refuses any contribution without a basis —
    equal labels alone never merge two subjects;
  - *naming*: the page title is one of the contribution's reviewed labels
    (title/aliases), another contributor's reviewed labels attest both names,
    or the contribution carries a ``same`` identity assessment bound to this
    entity (e.g. a reviewed Chinese subject assessed as the English page's
    entity; its claims show ``reviewed under "…"``).

  An unbound *admitted* contribution is a visible error (non-green build)
  and its content is labelled ``unbound``.
* **Relation resolution** — a relation is a live link only when its reviewed
  display label names the actual target page (title, reviewed alias, or the
  literal slug). Otherwise the assertion is preserved as an *unresolved*
  relation without a wikilink or backlink — whether the target exists now or
  is created later.
* **Derived backlinks** — typed app structure derived only from a live
  forward edge; they are never inserted into ``related`` (so they are never
  presented as reviewed relations) and carry the forward edge's status
  (a stale/unchecked forward edge never yields an admitted-looking backlink).
* **MoC membership** — a member is ``reviewed`` only if a contributing
  source reviewed a ``map_member`` unit whose label binds to the member page;
  members added by the app (similarity assignment, confirmed equivalences)
  are typed ``derived``; reviewed-label mismatches are not published.
* **Consistency check** — :func:`verify_publication` re-derives the plan's
  fingerprint; any later transform that retitles, re-aliases or retargets a
  planned page is reported instead of being published silently.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from obsidian_llm_wiki.core.entity_identity import (
    definition_snapshot,
    find_assessment,
    label_key,
    labels_of,
)
from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
    SynthesisBundle,
)

__all__ = [
    "DerivedBacklink",
    "claim_key",
    "PublicationPlan",
    "UnresolvedRelation",
    "plan_publication",
    "publication_statuses",
    "verify_publication",
]

logger = logging.getLogger("obswiki.synth.publication")

_INVERSE = {
    "depends_on": "prerequisite_of",
    "prerequisite_of": "depends_on",
    "contrasts_with": "contrasts_with",
    "complements": "complements",
    "competes_with": "competes_with",
    "equivalent_to": "equivalent_to",
    "related_to": "related_to",
}


@dataclass(frozen=True)
class UnresolvedRelation:
    """A reviewed relation assertion with no bound target page (rendered unlinked)."""

    relation: str
    display: str
    proposed_slug: str
    status: str = ""
    reason: str = ""
    source_file: str = ""
    source_revision: str = ""
    occurrence: str = ""


@dataclass(frozen=True)
class DerivedBacklink:
    """App-derived inverse of a live forward edge (not a reviewed unit)."""

    slug: str
    relation: str
    display: str
    forward_text: str
    status: str = ""


@dataclass
class PublicationPlan:
    """What each page may publish, with provenance/status for every derived item."""

    source_status: dict[str, str] = field(default_factory=dict)
    concept_status: dict[str, dict[str, str]] = field(default_factory=dict)
    moc_status: dict[str, dict[str, str]] = field(default_factory=dict)
    unresolved: dict[str, list[UnresolvedRelation]] = field(default_factory=dict)
    derived: dict[str, list[DerivedBacklink]] = field(default_factory=dict)
    link_status: dict[str, dict[tuple[str, str], str]] = field(default_factory=dict)
    moc_members: dict[str, dict[str, str]] = field(default_factory=dict)
    unresolved_members: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    claim_subjects: dict[str, dict[tuple, str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    # (source_file, canonical_id) -> registry binding basis; None = not supplied.
    identity_evidence: dict[tuple[str, str], str] | None = None

    def concept_render_payload(self, slug: str) -> dict[str, Any]:
        """Stable, JSON-friendly plan facts that determine one concept page."""
        return {
            "status": dict(sorted(self.concept_status.get(slug, {}).items())),
            "unresolved": [vars(item) for item in self.unresolved.get(slug, [])],
            "derived": [vars(item) for item in self.derived.get(slug, [])],
            "link_status": sorted(
                [list(key), value] for key, value in self.link_status.get(slug, {}).items()
            ),
            "claim_subjects": sorted(
                [list(map(str, key)), value]
                for key, value in self.claim_subjects.get(slug, {}).items()
            ),
        }

    def moc_render_payload(self, slug: str) -> dict[str, Any]:
        return {
            "status": dict(sorted(self.moc_status.get(slug, {}).items())),
            "members": dict(sorted(self.moc_members.get(slug, {}).items())),
            "unresolved": self.unresolved_members.get(slug, []),
        }


def publication_statuses(
    syntheses_by_file: Mapping[str, SourceSynthesis],
    retained: Mapping[str, str],
) -> dict[str, str]:
    """Retained statuses plus explicit ``unchecked`` for review-off receipts.

    Content produced with ``GROUNDING_REVIEW=off`` is published, but never as
    if it had been reviewed: its receipt status ``unchecked`` becomes a
    visible page status even when it is the current revision.
    """
    statuses = dict(retained)
    for filename, synthesis in syntheses_by_file.items():
        receipt = synthesis.review_receipt if isinstance(synthesis.review_receipt, dict) else {}
        if filename not in statuses and receipt.get("status") == "unchecked":
            statuses[filename] = "unchecked"
    return statuses


def claim_key(claim: Any) -> tuple:
    """Identity of one published claim (text + its own evidence anchor)."""
    evidence = claim.evidence
    from obsidian_llm_wiki.core.models import EvidenceFragments, evidence_identity

    if isinstance(evidence, EvidenceFragments):
        return (claim.text, claim.source_ref, "fragments", evidence_identity(evidence))
    return (
        claim.text, getattr(evidence, "source_file", ""),
        getattr(evidence, "start_offset", None), getattr(evidence, "end_offset", None),
    )


def _contributions(bundle: SynthesisBundle) -> tuple[
    dict[str, list[tuple[str, ConceptNote]]],
    dict[str, list[tuple[str, MapOfContent, dict[str, str]]]],
]:
    concepts: dict[str, list[tuple[str, ConceptNote]]] = {}
    maps: dict[str, list[tuple[str, MapOfContent, dict[str, str]]]] = {}
    for synthesis in bundle.sources:
        filename = synthesis.source_file or synthesis.source_title
        titles = {concept.slug: concept.title for concept in synthesis.concepts}
        for concept in synthesis.concepts:
            concepts.setdefault(concept.slug, []).append((filename, concept))
        for moc in synthesis.maps:
            maps.setdefault(moc.slug, []).append((filename, moc, titles))
    return concepts, maps


def _assessed_into(bundle: SynthesisBundle) -> dict[tuple[str, str], str]:
    """``(filename, slug) -> entity ID`` for contributions with a ``same`` assessment."""
    assessed: dict[tuple[str, str], str] = {}
    for synthesis in bundle.sources:
        filename = synthesis.source_file or synthesis.source_title
        for concept in synthesis.concepts:
            if concept.canonical_id and find_assessment(
                synthesis.review_receipt,
                subject_sha=definition_snapshot(concept.title, concept.summary),
                candidate_id=concept.canonical_id,
            ):
                assessed[(filename, concept.slug)] = concept.canonical_id
    return assessed


def _bound(title: str, contribution: set[str], all_labels: list[set[str]]) -> bool:
    """Is a contribution with reviewed labels ``contribution`` bound to ``title``?"""
    page = labels_of(title)
    if page & contribution:
        return True
    # Attestation: another contributor's own reviewed labels name both.
    return any(page & labels and contribution & labels for labels in all_labels)


def _choose_title(concept: ConceptNote, contributions: list[tuple[str, ConceptNote]]) -> None:
    """Prefer a contributor title every contribution reviewed itself.

    First choice: a title that is one of EVERY contributor's own reviewed
    labels (current title first, then contributors in filename order). Next:
    a title bound to every contributor through another contributor's
    attestation. Otherwise the title is kept and the binding check reports it.
    """
    labelled = [labels_of(c.title, c.aliases) for _f, c in contributions]
    candidates = [concept.title, *(c.title for _f, c in contributions)]

    def direct(title: str) -> bool:
        return all(labels_of(title) & labels for labels in labelled)

    def attested(title: str) -> bool:
        return all(_bound(title, labels, labelled) for labels in labelled)

    for test in (direct, attested):
        chosen = next((title for title in candidates if test(title)), None)
        if chosen is None:
            continue
        if chosen != concept.title:
            old = concept.title
            concept.title = chosen
            concept.aliases = [
                alias for alias in dict.fromkeys([old, *concept.aliases])
                if label_key(alias) and label_key(alias) != label_key(chosen)
            ]
        return


def _binding_errors(
    bundle: SynthesisBundle,
    contributions: dict[str, list[tuple[str, ConceptNote]]],
    statuses: Mapping[str, str],
    identity_evidence: Mapping[tuple[str, str], str] | None = None,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Unbound contributions per concept slug, and the visible errors they raise."""
    unbound: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    assessed = _assessed_into(bundle)
    for concept in bundle.concepts:
        parts = contributions.get(concept.slug, [])
        if not parts:
            continue
        _choose_title(concept, parts)
        labelled = [labels_of(c.title, c.aliases) for _f, c in parts]
        several = len({filename for filename, _c in parts}) > 1
        for (filename, contribution), labels in zip(parts, labelled, strict=True):
            entity = contribution.canonical_id or concept.canonical_id
            has_evidence = (
                identity_evidence is None or not several
                or bool(identity_evidence.get((filename, entity)))
            )
            named = _bound(concept.title, labels, labelled) or (
                assessed.get((filename, contribution.slug)) == entity
            )
            if has_evidence and named:
                continue
            if filename in statuses:
                continue  # already published with its retained/unchecked label
            unbound.setdefault(concept.slug, {})[filename] = "unbound"
            if not has_evidence:
                errors.append(
                    f"publication:{filename}: concept {concept.slug!r} merges this source's "
                    f"subject {contribution.title!r} without identity evidence; equal "
                    "labels are not proof that two sources describe the same entity"
                )
                continue
            errors.append(
                f"publication:{filename}: concept {concept.slug!r} is titled "
                f"{concept.title!r}, but this source's content was reviewed under "
                f"{contribution.title!r}; the subject is not bound to the published page"
            )
    return unbound, errors


def _target_labels(target: ConceptNote) -> set[str]:
    return labels_of(target.title, target.aliases)


def _link_status(
    concept: ConceptNote,
    link: Any,
    parts: list[tuple[str, ConceptNote]],
    statuses: Mapping[str, str],
) -> str | None:
    """``""`` if an admitted source asserts the edge, else the best status.

    ``None`` means no source asserts it (app/human structure such as a
    confirmed equivalence).
    """
    asserted = [
        filename for filename, contribution in parts
        if any(
            (other.slug, other.relation, (other.display or "").strip())
            == (link.slug, link.relation, (link.display or "").strip())
            for other in contribution.related
        )
    ]
    if not asserted:
        return None
    if any(filename not in statuses for filename in asserted):
        return ""
    return statuses[asserted[0]]


def _fingerprint(bundle: SynthesisBundle) -> dict[str, Any]:
    return {
        "concepts": {
            concept.slug: [
                concept.title, sorted(concept.aliases),
                sorted([link.slug, link.relation, link.display] for link in concept.related),
            ]
            for concept in bundle.concepts
        },
        "maps": {
            moc.slug: [moc.title, list(moc.concept_slugs), sorted(moc.tags)]
            for moc in bundle.maps
        },
    }


def plan_publication(
    bundle: SynthesisBundle,
    statuses: Mapping[str, str] | None = None,
    *,
    identity_evidence: Mapping[tuple[str, str], str] | None = None,
) -> PublicationPlan:
    """Compute (and apply to ``bundle.concepts``/``bundle.maps``) the publication plan.

    ``statuses`` maps source files whose content is published but not currently
    admitted (``stale``/``unadmitted``/``unchecked``) to that status. Only the
    merged bundle concepts/maps are changed; per-source syntheses (and their
    review receipts) are never mutated. ``identity_evidence`` maps
    ``(source_file, canonical_id)`` to the registry's binding basis; when it is
    given, a multi-source page refuses contributions without one.
    """
    statuses = dict(statuses or {})
    plan = PublicationPlan(
        source_status=dict(statuses),
        identity_evidence=dict(identity_evidence) if identity_evidence is not None else None,
    )
    contributions, map_contributions = _contributions(bundle)
    unbound, errors = _binding_errors(bundle, contributions, statuses, identity_evidence)
    plan.errors.extend(errors)

    def effective(slug: str) -> dict[str, str]:
        """Per-page status of each contributing source (retained or unbound)."""
        return {**statuses, **unbound.get(slug, {})}

    for slug, parts in contributions.items():
        retained = {f: statuses[f] for f, _c in parts if f in statuses}
        retained.update(unbound.get(slug, {}))
        if retained:
            plan.concept_status[slug] = retained

    # Provenance: a claim reviewed under another (bound) subject label shows it.
    for concept in bundle.concepts:
        for _filename, contribution in contributions.get(concept.slug, []):
            if label_key(contribution.title) == label_key(concept.title):
                continue
            for claim in contribution.claims:
                plan.claim_subjects.setdefault(concept.slug, {})[
                    claim_key(claim)
                ] = contribution.title

    by_slug = {concept.slug: concept for concept in bundle.concepts}
    # A page answers only to labels its BOUND contributors reviewed. An unbound
    # contributor's title must not become a live alias of someone else's page.
    page_labels: dict[str, set[str]] = {}
    for concept in bundle.concepts:
        parts = contributions.get(concept.slug, [])
        rejected = unbound.get(concept.slug, {})
        if rejected:
            keep = labels_of(concept.title)
            for filename, contribution in parts:
                if filename not in rejected:
                    keep |= labels_of(contribution.title, contribution.aliases)
            concept.aliases = [alias for alias in concept.aliases if labels_of(alias) & keep]
        page_labels[concept.slug] = labels_of(concept.title, concept.aliases)

    def binds(display: str, target: ConceptNote) -> bool:
        shown = (display or "").strip()
        if not shown or shown == target.slug:
            return True
        return bool(labels_of(shown) & page_labels.get(target.slug, _target_labels(target)))

    relation_occurrences = {}
    member_occurrences = {}
    for synthesis in bundle.sources:
        references = (synthesis.review_receipt or {}).get("references", {})
        for i, contribution in enumerate(synthesis.concepts):
            for j, link in enumerate(contribution.related):
                record = references.get(f"concepts[{i}].related[{j}]")
                if record is not None:
                    relation_occurrences.setdefault(
                        (contribution.slug, link.slug, link.relation, link.display), []
                    ).append(record)
        for i, moc in enumerate(synthesis.maps):
            for j, member in enumerate(moc.concept_slugs):
                record = references.get(f"maps[{i}].members[{j}]")
                if record is not None:
                    member_occurrences.setdefault((moc.slug, member), []).append(record)

    live_edges: list[tuple[ConceptNote, Any, str]] = []
    for concept in bundle.concepts:
        parts = contributions.get(concept.slug, [])
        kept = []
        for link in concept.related:
            if link.slug == concept.slug:
                continue
            target = by_slug.get(link.slug)
            status = _link_status(concept, link, parts, effective(concept.slug))
            occurrences = relation_occurrences.get(
                (concept.slug, link.slug, link.relation, link.display)
            )
            if occurrences is not None:
                for record in occurrences:
                    if target is not None and record.get("target_id") == target.canonical_id:
                        continue
                    plan.unresolved.setdefault(concept.slug, []).append(UnresolvedRelation(
                        relation=link.relation, display=record["original_text"],
                        proposed_slug=record["original_slug"], status=status or "",
                        reason=record.get("reason", "no bilateral occurrence binding"),
                        source_file=record.get("source_file", ""),
                        source_revision=record.get("source_revision", ""),
                        occurrence=record.get("occurrence", ""),
                    ))
                live = target is not None and any(
                    r.get("target_id") == target.canonical_id for r in occurrences
                )
            else:
                live = target is not None and binds(link.display, target)
            if live:
                kept.append(link)
                if status:
                    plan.link_status.setdefault(concept.slug, {})[
                        (link.slug, link.relation)
                    ] = status
                live_edges.append((concept, link, status or ""))
                continue
            if occurrences is not None:
                continue
            reason = (
                "no page in this wiki is bound to this label"
                if target is None
                else f"the proposed target page is titled {target.title!r}"
            )
            plan.unresolved.setdefault(concept.slug, []).append(UnresolvedRelation(
                relation=link.relation, display=(link.display or "").strip() or link.slug,
                proposed_slug=link.slug, status=status or "", reason=reason,
            ))
        concept.related = kept

    forward = {(concept.slug, link.slug) for concept, link, _s in live_edges}
    seen: set[tuple[str, str]] = set()
    for concept, link, status in live_edges:
        pair = (link.slug, concept.slug)
        if pair in forward or pair in seen:
            continue
        seen.add(pair)
        display = (link.display or "").strip() or link.slug
        plan.derived.setdefault(link.slug, []).append(DerivedBacklink(
            slug=concept.slug,
            relation=_INVERSE.get(link.relation, "related_to"),
            display=concept.title,
            forward_text=f"{concept.title} {link.relation} {display}",
            status=status,
        ))

    from obsidian_llm_wiki.synth.review import member_label

    for moc in bundle.maps:
        parts = map_contributions.get(moc.slug, [])
        retained = {f: statuses[f] for f, _m, _t in parts if f in statuses}
        for filename, contribution, _titles in parts:
            if label_key(contribution.title) != label_key(moc.title):
                retained[filename] = "unbound"
                if filename not in statuses:
                    plan.errors.append(
                        f"publication:{filename}: MoC {moc.slug!r} is titled {moc.title!r}, "
                        f"but this source reviewed it as {contribution.title!r}"
                    )
        if retained:
            plan.moc_status[moc.slug] = retained
        reviewed: dict[str, list[tuple[str, str]]] = {}
        for filename, contribution, titles in parts:
            for member in contribution.concept_slugs:
                reviewed.setdefault(member, []).append(
                    (member_label(member, titles), filename)
                )
        members: dict[str, str] = {}
        kept_members = []
        for member in moc.concept_slugs:
            target = by_slug.get(member)
            records = reviewed.get(member)
            occurrences = member_occurrences.get((moc.slug, member))
            if occurrences is not None:
                unresolved = [
                    r
                    for r in occurrences
                    if target is None or r.get("target_id") != target.canonical_id
                ]
                plan.unresolved_members.setdefault(moc.slug, []).extend(unresolved)
                if len(unresolved) == len(occurrences):
                    continue
            if not records:
                members[member] = "derived"
                kept_members.append(member)
                continue
            if target is None:
                kept_members.append(member)  # rendered only if the page exists
                continue
            member_status = effective(member)
            bound = [
                filename for label, filename in records
                if binds(label, target) and member_status.get(filename) != "unbound"
            ]
            if not bound:
                logger.warning(
                    "MoC %r member %r: reviewed label does not name page %r; not published",
                    moc.slug, member, target.title,
                )
                continue
            admitted = [filename for filename in bound if filename not in member_status]
            members[member] = "reviewed" if admitted else member_status[bound[0]]
            kept_members.append(member)
        moc.concept_slugs = kept_members
        plan.moc_members[moc.slug] = members

    plan.fingerprint = _fingerprint(bundle)
    for error in plan.errors:
        logger.error("%s", error)
    return plan


def verify_publication(bundle: SynthesisBundle, plan: PublicationPlan) -> list[str]:
    """Report any planned page whose identity or links changed after planning."""
    current = _fingerprint(bundle)
    problems: list[str] = []
    for kind in ("concepts", "maps"):
        before, after = plan.fingerprint.get(kind, {}), current[kind]
        for slug in sorted(set(before) | set(after)):
            if before.get(slug) != after.get(slug):
                problems.append(
                    f"publication: {kind[:-1]} {slug!r} changed after the publication plan "
                    f"({before.get(slug)!r} -> {after.get(slug)!r}); reviewed content was "
                    "re-planned under its actual published identity"
                )
    return problems
