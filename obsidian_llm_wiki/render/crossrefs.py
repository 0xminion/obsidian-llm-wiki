"""Cross-reference diagram builders for concepts and MoCs.

Extracted from render/obsidian.py for modularity. Renders typed-edge
relationship graphs as ASCII flow diagrams inside markdown code blocks.
"""

from __future__ import annotations

from obsidian_llm_wiki.core.models import ConceptNote

__all__ = [
    "build_cross_ref_diagram",
    "build_cross_links",
    "build_moc_cross_ref_diagram",
    "build_moc_cross_links",
    # Backward-compat aliases
    "_build_cross_ref_diagram",
    "_build_moc_cross_ref_diagram",
]


def build_cross_ref_diagram(
    concept: ConceptNote,
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Build the typed-edge cross-reference ASCII flow diagram.

    Returns only the diagram lines (for a ```text code block). Cross-link
    wikilinks are produced separately by :func:`build_cross_links` so they
    can be rendered as clickable markdown outside the code block.
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

    return lines


def build_cross_links(
    concept: ConceptNote,
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Return clickable wikilink lines for cross-references.

    These are rendered as regular markdown (outside any code block) so
    Obsidian treats the ``[[slug]]`` syntax as live, navigable links.
    """
    cross_link_lines: list[str] = []
    for link in concept.related:
        target = all_concepts.get(link.slug)
        if target:
            descriptor = link.relation or "related"
            display = link.display or target.title
            cross_link_lines.append(f"- [[{link.slug}|{display}]] ({descriptor})")
    return cross_link_lines


def build_moc_cross_ref_diagram(
    moc_concepts: list[ConceptNote],
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Build ASCII flow diagram for MoC cross-references.

    Returns only the diagram lines (for a ```text code block). Cross-link
    wikilinks are produced separately by :func:`build_moc_cross_links`.
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

    # Remove trailing blank line so the code block closes cleanly.
    if lines and lines[-1] == "":
        lines.pop()

    return lines


def build_moc_cross_links(
    moc_concepts: list[ConceptNote],
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Return clickable wikilink lines for MoC cross-references.

    Rendered as regular markdown outside the code block for navigation.
    """
    moc_slugs = {c.slug for c in moc_concepts}
    cross_links: list[str] = []
    seen_link_slugs: set[tuple[str, str]] = set()

    for concept in moc_concepts:
        for link in concept.related or []:
            if link.slug not in moc_slugs:
                continue
            pair = tuple(sorted([concept.slug, link.slug]))
            if pair in seen_link_slugs:
                continue
            seen_link_slugs.add(pair)
            target = all_concepts.get(link.slug)
            descriptor = link.relation or "related"
            display = link.display or (target.title if target else link.slug)
            cross_links.append(f"- [[{link.slug}|{display}]] ({descriptor})")

    return cross_links


# Backward-compat aliases
_build_cross_ref_diagram = build_cross_ref_diagram
_build_moc_cross_ref_diagram = build_moc_cross_ref_diagram

