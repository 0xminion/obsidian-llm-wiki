"""Revision-bound identity proof shared by registry and publication.

Names, aliases, ownership, slugs, and equal generic definitions are proposals,
not authority. A new entity receives a stable opaque ID. Its unchanged admitted
revision may retain an established binding; a changed or foreign contribution
requires a same assessment carrying complete bilateral source-backed context.
Proof tokens are scoped to the pair, source revisions and exact offered context.
Missing historical bytes and bounded incomplete context remain insufficient.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "IDENTITY_RECORD_VERSION",
    "SAME",
    "assessment_records",
    "definition_snapshot",
    "find_assessment",
    "identity_proposal",
    "label_key",
    "labels_of",
]

IDENTITY_RECORD_VERSION = "identity-assessment-v2-bilateral"
SAME = "same"
TERMINAL_VERDICTS = frozenset({"same", "different", "insufficient"})

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
# Only spacing and word-joining connectors between two word characters are
# typographic ("FlashAttention" / "Flash Attention" / "Flash-Attention").
# Every other punctuation mark is meaning-bearing ("C++", "C#", "F*", "3.11").
_CONNECTOR = re.compile(r"(?<=[^\W_])[\s\-\u2010-\u2015_\u00b7]+(?=[^\W_])", re.UNICODE)


def label_key(label: object) -> str:
    """Case/whitespace-insensitive identity of one human-visible label."""
    return " ".join(str(label or "").casefold().split())


_BILINGUAL = re.compile(r"(?P<a>[^()（）]+?)\s*[（(](?P<b>[^()（）]+)[)）]")


def _compact(key: str) -> str:
    return _CONNECTOR.sub("", key)


def _label_forms(label: object) -> set[str]:
    key = label_key(label)
    if not key:
        return set()
    forms = {key, "\x00" + _compact(key)}
    # An explicit bilingual title "English (中文)" names the entity in both
    # languages; a same-script parenthetical ("Python (snake)") is a
    # disambiguator and is never split.
    match = _BILINGUAL.fullmatch(key)
    if match:
        a, b = match.group("a").strip(), match.group("b").strip()
        if bool(_CJK.search(a)) != bool(_CJK.search(b)):
            for part in (a, b):
                forms.add(part)
                forms.add("\x00" + _compact(part))
    return forms


def labels_of(title: object, aliases: Iterable[object] = ()) -> set[str]:
    """Normalized reviewed labels of a contribution (title + aliases).

    Contains each label's case/whitespace-normalized form, its typographic
    variant (spacing/word connectors removed between word characters only;
    semantic punctuation such as ``+``, ``#``, ``*`` or ``.`` is preserved),
    and both halves of an explicit cross-script bilingual title. These are
    *naming* facts used for titles and relation displays, never identity proof.
    """
    forms: set[str] = set()
    for label in (title, *aliases):
        forms |= _label_forms(label)
    return forms


def contribution_snapshot(concept) -> str:
    """Exact reviewed subject payload, excluding mutable storage/proposal fields."""
    from obsidian_llm_wiki.core.models import concept_note_to_dict
    value = concept_note_to_dict(concept)
    for key in ("slug", "canonical_id", "is_new", "confidence", "provenance"):
        value.pop(key, None)
    for claim in value["claims"]:
        claim.pop("concept_slug", None)
    for link in value["related"]:
        link.pop("slug", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def definition_snapshot(title: object, summary: object) -> str:
    """Digest of one reviewed definition ``(title, summary)``; ``""`` if missing.

    A missing title or summary has no snapshot: it can neither match another
    definition nor be assessed, so it never proves identity.
    """
    title_key = label_key(title)
    text = " ".join(str(summary or "").split())
    if not title_key or not text:
        return ""
    material = json.dumps([title_key, text], ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _identity_block(receipt: object) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping):
        return {}
    block = receipt.get("identity")
    return block if isinstance(block, Mapping) else {}


def assessment_records(receipt: object) -> list[Mapping[str, Any]]:
    """Well-formed identity assessment records of a receipt (any verdict)."""
    block = _identity_block(receipt)
    if block.get("version") != IDENTITY_RECORD_VERSION:
        return []
    records = block.get("assessments")
    if not isinstance(records, list):
        return []
    return [
        record for record in records
        if isinstance(record, Mapping)
        and all(isinstance(record.get(key), str) for key in (
            "subject_sha256", "candidate_id", "candidate_sha256", "verdict", "source_sha256",
        ))
    ]


def find_assessment(
    receipt: object,
    *,
    subject_sha: str,
    candidate_id: str,
    candidate_sha: str | None = None,
    candidate_context_sha: str | None = None,
) -> Mapping[str, Any] | None:
    """The valid ``same`` assessment binding this subject to this candidate.

    Valid means: current record version, the receipt is an approved review
    receipt, the record belongs to the receipt's own source revision, verdict
    ``same`` with at least one support span, and it is bound to exactly this
    subject snapshot, candidate ID and (when given) candidate snapshot.
    """
    if not subject_sha or not candidate_id:
        return None
    if not isinstance(receipt, Mapping) or receipt.get("status") != "approved":
        return None
    revision = receipt.get("source_sha256")
    for record in assessment_records(receipt):
        from obsidian_llm_wiki.core.identity_context import bilateral_support
        from obsidian_llm_wiki.core.identity_plan import digest
        spans = record.get("span_ids")
        if (
            record["verdict"] == SAME
            and record["subject_sha256"] == subject_sha
            and record["candidate_id"] == candidate_id
            and (candidate_sha is None or record["candidate_sha256"] == candidate_sha)
            and record["source_sha256"] == revision
            and isinstance(spans, list)
            and spans
            and all(isinstance(span, str) and span for span in spans)
            and bilateral_support(record, spans)
            and record.get("subject_context", {}).get("source_sha256") == revision
            and record.get("subject_context_sha256") == digest(record.get("subject_context", {}))
            and record.get("candidate_context_sha256")
            == digest(record.get("candidate_context", {}))
            and (
                candidate_context_sha is None
                or record.get("candidate_context_sha256") == candidate_context_sha
            )
            and record.get("candidate_context", {}).get("candidate_id") == candidate_id
        ):
            return record
    return None


def identity_proposal(receipt: object, subject_sha: str) -> str:
    """The candidate ID the producer proposed for this subject (persisted hint)."""
    if not subject_sha:
        return ""
    proposals = _identity_block(receipt).get("proposals")
    if not isinstance(proposals, Mapping):
        return ""
    value = proposals.get(subject_sha)
    return value if isinstance(value, str) else ""
