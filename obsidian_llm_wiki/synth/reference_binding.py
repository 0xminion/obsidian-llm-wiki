"""Revision-bound per-occurrence reference decisions, independent of page paths."""

from __future__ import annotations

from obsidian_llm_wiki.core.entity_identity import (
    IDENTITY_RECORD_VERSION,
    definition_snapshot,
    find_assessment,
)
from obsidian_llm_wiki.core.identity_context import MAX_CONTEXT_OWNERS, source_context
from obsidian_llm_wiki.core.identity_plan import digest
from obsidian_llm_wiki.core.span_catalog import build_span_catalog, source_revision
from obsidian_llm_wiki.synth.identity import assess_identity, assessor_fingerprint


def capture_references(syntheses):
    result = {}
    for synthesis in syntheses:
        from obsidian_llm_wiki.synth.review import collect_review_units

        units = {unit.path: unit for unit in collect_review_units(synthesis)}
        originals = {}
        old = (synthesis.review_receipt or {}).get("references", {})
        titles = {c.slug: c.title for c in synthesis.concepts}
        for i, concept in enumerate(synthesis.concepts):
            for j, link in enumerate(concept.related):
                key = f"concepts[{i}].related[{j}]"
                originals[key] = dict(
                    old.get(key)
                    or {
                        "original_text": link.display or link.slug,
                        "original_display": link.display,
                        "original_slug": link.slug,
                        "relation": link.relation,
                    }
                )
        for i, moc in enumerate(synthesis.maps):
            for j, slug in enumerate(moc.concept_slugs):
                key = f"maps[{i}].members[{j}]"
                originals[key] = dict(
                    old.get(key)
                    or {
                        "original_text": titles.get(slug, slug),
                        "original_slug": slug,
                        "relation": "member",
                    }
                )
        for key, record in originals.items():
            unit = units.get(key.replace(".members[", ".concept_slugs["))
            if unit is not None:
                record["review_unit"] = {
                    "key": unit.key,
                    "text": unit.text,
                    "subject": unit.subject,
                }
        result[synthesis.source_file] = originals
    return result


async def bind_references(
    syntheses, originals, registry, config, request_limiter=None, errors=None
):
    """Every external occurrence is assessed against prospective owner contexts.

    A recorded target ID survives only while that entity survives. It never falls
    back to a newly occupied slug. Insufficient occurrences retain original data.
    """
    by_id = {}
    by_slug = {}
    for synthesis in syntheses:
        for i, concept in enumerate(synthesis.concepts):
            by_id.setdefault(concept.canonical_id, []).append((synthesis, i, concept))
            by_slug[concept.slug] = concept.canonical_id
    for synthesis in syntheses:
        receipt = synthesis.review_receipt
        if not isinstance(receipt, dict):
            receipt = synthesis.review_receipt = {}
        revision = receipt.get("source_sha256", "")
        content = registry.identity_content(synthesis.source_file, revision)
        references = {}
        local = {c.slug: c.canonical_id for c in synthesis.concepts}
        occurrences = []
        for i, c in enumerate(synthesis.concepts):
            for j, link in enumerate(c.related):
                occurrences.append((f"concepts[{i}].related[{j}]", i, link.slug, link))
        for i, moc in enumerate(synthesis.maps):
            for j, slug in enumerate(moc.concept_slugs):
                occurrences.append((f"maps[{i}].members[{j}]", 0, slug, (moc, j)))
        for key, index, slug, item in occurrences:
            original = originals[synthesis.source_file][key]
            record = {
                **original,
                "occurrence": key,
                "source_file": synthesis.source_file,
                "source_revision": revision,
                "status": "unresolved",
                "target_id": "",
                "review_sha256": digest(
                    {
                        k: v
                        for k, v in receipt.items()
                        if k not in {"identity", "publication_binding", "references"}
                    }
                ),
            }
            previous_id = original.get("bound_entity_id") or original.get("target_id", "")
            candidate_id = previous_id or by_slug.get(slug, "")
            if previous_id and candidate_id not in by_id:
                record["reason"] = "previously bound entity no longer has a current owner"
            elif slug in local and (not previous_id or previous_id == local[slug]):
                # Source-local selectors refer to the exact reviewed local contribution,
                # not an external page's mutable label.
                record.update(
                    target_id=local[slug],
                    bound_entity_id=local[slug],
                    status="bound",
                    basis="reviewed local occurrence",
                )
            elif candidate_id in by_id:
                owners = []
                parts = by_id[candidate_id]
                for owner, ci, _concept in parts[:MAX_CONTEXT_OWNERS]:
                    owner_rev = (owner.review_receipt or {}).get("source_sha256", "")
                    owners.append(
                        source_context(
                            owner,
                            ci,
                            registry.identity_content(owner.source_file, owner_rev),
                            "candidate:" + candidate_id,
                        )
                    )
                candidate_context = {
                    "candidate_id": candidate_id,
                    "owners": owners,
                    "complete": len(owners) == len(parts) and all(o["complete"] for o in owners),
                }
                signature = digest(candidate_context)
                proof_receipt = {
                    **receipt,
                    "identity": {
                        "version": IDENTITY_RECORD_VERSION,
                        "assessments": [original.get("proof", {})],
                    },
                }
                old_proof = find_assessment(
                    proof_receipt,
                    subject_sha=definition_snapshot(record["original_text"], key),
                    candidate_id=candidate_id,
                    candidate_context_sha=signature,
                )
                terminal = original.get("proof", {})
                if (
                    terminal.get("verdict") in {"different", "insufficient"}
                    and terminal.get("candidate_id") == candidate_id
                    and terminal.get("candidate_context_sha256") == signature
                    and terminal.get("subject_sha256")
                    == definition_snapshot(record["original_text"], key)
                ):
                    old_proof = terminal
                if (
                    original.get("source_revision") == revision
                    and original.get("review_sha256") == record["review_sha256"]
                    and original.get("candidate_context_sha256") == signature
                    and original.get("assessor") == assessor_fingerprint(config)
                    and old_proof is not None
                    and old_proof.get("subject_context_sha256")
                    == digest(
                        source_context(
                            synthesis, index, content, occurrence=record.get("review_unit", {})
                        )
                    )
                ):
                    record = dict(original)
                elif (
                    isinstance(content, str)
                    and source_revision(content) == revision
                    and receipt.get("status") == "approved"
                ):
                    target = parts[0][2]
                    request = {
                        "concept_index": index,
                        "subject_title": record["original_text"],
                        "subject_summary": record.get("review_unit", {}).get(
                            "text", record["original_text"]
                        ),
                        "occurrence": record.get("review_unit", {}),
                        "subject_sha256": definition_snapshot(record["original_text"], key),
                        "candidate_id": candidate_id,
                        "candidate_title": target.title,
                        "candidate_summary": target.summary,
                        "candidate_sha256": definition_snapshot(target.title, target.summary),
                        "anchor_source": parts[0][0].source_file,
                        "candidate_context": candidate_context,
                    }
                    proof = (
                        await assess_identity(
                            synthesis,
                            [request],
                            build_span_catalog(content, synthesis.source_file),
                            config,
                            request_limiter=request_limiter,
                        )
                    )[0]
                    record.update(
                        proof=proof,
                        candidate_context_sha256=signature,
                        assessor=assessor_fingerprint(config),
                        reason=proof.get("reason", proof["verdict"]),
                    )
                    if proof["verdict"] == "unassessed" and errors is not None:
                        errors.append(
                            f"identity-reference:{synthesis.source_file}:{key}: "
                            f"{proof.get('reason', 'unassessed')}"
                        )
                    if proof["verdict"] == "same":
                        record.update(
                            target_id=candidate_id,
                            bound_entity_id=candidate_id,
                            status="bound",
                            basis="bilateral occurrence assessment",
                        )
                else:
                    record["reason"] = "reviewed source context unavailable"
            else:
                record["reason"] = "no candidate entity; original assertion retained"
            if record["target_id"] in by_id:
                target_slug = by_id[record["target_id"]][0][2].slug
                if isinstance(item, tuple):
                    item[0].concept_slugs[item[1]] = target_slug
                else:
                    item.slug = target_slug
            references[key] = record
        receipt["references"] = references
