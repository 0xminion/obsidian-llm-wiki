"""Cross-reference diagram builders for concepts and MoCs.

Extracted from render/obsidian.py for modularity. Renders typed-edge
relationship graphs as ASCII flow diagrams inside markdown code blocks.
"""

from __future__ import annotations

from obsidian_llm_wiki.core.models import ConceptNote

__all__ = [
    "build_cross_ref_diagram",
    "build_moc_cross_ref_diagram",
    # Backward-compat aliases
    "_build_cross_ref_diagram",
    "_build_moc_cross_ref_diagram",
]


def build_cross_ref_diagram(
    concept: ConceptNote,
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Build the typed-edge cross-reference section as an ASCII flow diagram.

    Renders inside a code block (```text) so Obsidian shows it as monospace
    with a copy icon.

    Format:
      Concept A
          ↓ relation
      Concept B → Concept C

      Cross-links: [[slug]] (descriptor)
    """
    lines: list[str] = []

    for link in concept.related:
        target = all_concepts.get(link.slug)
        if not target:
            continue

        target_display = link.display or target.slug
        relation_type = link.relation or "related_to"

        lines.append(concept.title)
        lines.append(f"    ↓ {relation_type}")

        # Target line with any sub-relationships
        target_sub = ""
        if target.related:
            sub_links = [
                r for r in target.related
                if r.slug != concept.slug
            ][:2]
            if sub_links:
                sub_parts = []
                for sl in sub_links:
                    sub_target = all_concepts.get(sl.slug)
                    if sub_target:
                        sub_parts.append(sub_target.title)
                if sub_parts:
                    target_sub = " → " + " → ".join(sub_parts)

        # Check for bidirectional edge
        reverse_links = [
            r for r in (target.related or [])
            if r.slug == concept.slug
        ]
        if reverse_links and not target_sub:
            rev_rel = reverse_links[0].relation or "related_to"
            lines.append(
                f"{target.title} ↔ {concept.title} ({rev_rel})"
            )
        elif target_sub:
            lines.append(f"{target_display}{target_sub}")
        else:
            lines.append(f"{target_display}")

    # Cross-links section
    if lines and concept.related:
        cross_link_lines: list[str] = []
        for link in concept.related:
            target = all_concepts.get(link.slug)
            if target:
                descriptor = link.relation or "related"
                cross_link_lines.append(
                    f"  [[{link.slug}]] ({descriptor})"
                )
        if cross_link_lines:
            lines.append("")
            lines.append("Cross-links:")
            lines.extend(cross_link_lines)

    return lines


def build_moc_cross_ref_diagram(
    moc_concepts: list[ConceptNote],
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Build ASCII flow diagram matching concept cross-ref format.

    Format (consistent with build_cross_ref_diagram):
      Concept A
          ↓ relation
      Concept B

      Cross-links:
        [[slug-a]] (relation)
    """
    lines: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    moc_slugs = {c.slug for c in moc_concepts}

    for concept in moc_concepts:
        for link in concept.related or []:
            if link.slug not in all_concepts:
                continue
            if link.slug not in moc_slugs:
                continue

            pair_key = tuple(sorted([concept.slug, link.slug]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            target = all_concepts[link.slug]
            relation = link.relation or "related_to"

            reverse = any(
                r.slug == concept.slug
                for r in (target.related or [])
            )

            lines.append(concept.title)
            lines.append(f"    ↓ {relation}")
            if reverse:
                lines.append(f"{target.title} ↔ {concept.title}")
            else:
                lines.append(f"{target.title}")
            lines.append("")

    # Cross-links section
    if lines:
        cross_links: list[str] = []
        seen_link_slugs: set[str] = set()
        for concept in moc_concepts:
            for link in concept.related or []:
                if link.slug not in moc_slugs:
                    continue
                pair = tuple(sorted([concept.slug, link.slug]))
                if pair in seen_link_slugs:
                    continue
                seen_link_slugs.add(pair)
                descriptor = link.relation or "related"
                cross_links.append(
                    f"  [[{link.slug}]] ({descriptor})"
                )
        if cross_links:
            if lines and lines[-1] == "":
                lines.pop()
            lines.append("")
            lines.append("Cross-links:")
            lines.extend(cross_links)

    return lines


# Backward-compat aliases
_build_cross_ref_diagram = build_cross_ref_diagram
_build_moc_cross_ref_diagram = build_moc_cross_ref_diagram

