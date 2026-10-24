"""Human-confirmed concept equivalences, keyed by durable registry IDs.

Assertions create relationships, never merge notes or translate their prose.
Similarity scores are not assertions and must not be written here automatically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from obsidian_llm_wiki.core.models import ConceptLink, SynthesisBundle

_CANONICAL_ID = re.compile(r"concept-[0-9a-f]{24}\Z")
_MAX_POLICY_BYTES = 1_000_000
_MAX_PAIRS = 10_000


def load_equivalences(path: Path) -> tuple[tuple[str, str], ...]:
    """Read a bounded, versioned local policy without modifying any vault file."""
    if not path.exists():
        return ()
    if path.stat().st_size > _MAX_POLICY_BYTES:
        raise ValueError("equivalences.json exceeds the policy byte limit")
    with path.open("rb") as stream:
        raw = stream.read(_MAX_POLICY_BYTES + 1)
    if len(raw) > _MAX_POLICY_BYTES:
        raise ValueError("equivalences.json exceeds the policy byte limit")
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or set(payload) != {"schema_version", "equivalent_concepts"}
    ):
        raise ValueError("equivalences.json requires schema_version=1 and equivalent_concepts")
    pairs = payload["equivalent_concepts"]
    if not isinstance(pairs, list) or len(pairs) > _MAX_PAIRS:
        raise ValueError("equivalent_concepts must be a bounded list")
    validated = set()
    for pair in pairs:
        if (
            not isinstance(pair, list) or len(pair) != 2
            or any(not isinstance(value, str) or not _CANONICAL_ID.fullmatch(value)
                   for value in pair)
            or pair[0] == pair[1]
        ):
            raise ValueError("each equivalence needs two distinct concept canonical IDs")
        validated.add(tuple(sorted(pair)))
    return tuple(sorted(validated))


def apply_equivalences(
    bundle: SynthesisBundle, pairs: tuple[tuple[str, str], ...]
) -> tuple[set[tuple[str, str]], list[str]]:
    """Link known concepts symmetrically; report unresolved IDs without guessing."""
    by_id = {concept.canonical_id: concept for concept in bundle.concepts if concept.canonical_id}
    resolved = set()
    errors = []
    for left_id, right_id in pairs:
        if left_id not in by_id or right_id not in by_id:
            errors.append(f"equivalence: unresolved canonical pair {left_id}, {right_id}")
            continue
        left, right = by_id[left_id], by_id[right_id]
        resolved.add((left.slug, right.slug))
        for source, target in ((left, right), (right, left)):
            if not any(link.slug == target.slug and link.relation == "equivalent_to"
                       for link in source.related):
                source.related.append(ConceptLink(target.slug, "equivalent_to", target.title))
    return resolved, errors
