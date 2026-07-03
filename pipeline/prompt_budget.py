"""Fair-share prompt budget management.

When assembling multiple source slices into a single concept-generation prompt,
the total content can exceed the configured budget.  This module provides a
proportional truncation strategy so every source contributes its fair share.

Ported from llm-wiki-compiler concept prompt assembly logic.
"""

from __future__ import annotations

import os

from pipeline.config import Config
from pipeline.okf_models import SourceSlice

# ──────────────────────────────────────────────────────────────────────────────
# Budgeted content assembly
# ──────────────────────────────────────────────────────────────────────────────


def build_budgeted_combined_content(
    concept: str,
    slices: list[SourceSlice],
    budget: int,
) -> str:
    """Assemble source slices into a combined prompt, respecting the budget.

    When total content exceeds *budget*, each source is proportionally
    truncated.  Every source always contributes at least one character (unless
    budget is zero), so no source is silently dropped.

    Args:
        concept: The concept name (used only for a descriptive header comment).
        slices: One or more ``SourceSlice`` objects with ``file`` and ``content``.
        budget: Maximum total character budget for the combined text.

    Returns:
        A formatted string with ``--- SOURCE: file.md ---`` headers and
        optionally truncated content.
    """
    if not slices:
        return ""

    # ── Compute lengths ──────────────────────────────────────────────────
    # Each slice has a fixed header overhead.
    header_overhead = 4  # "--- SOURCE: " + newline + trailing newline

    total_raw = sum(len(s.content) for s in slices)
    total_overhead = sum(header_overhead + len(s.file) for s in slices)
    separator_overhead = (len(slices) - 1) * 1  # blank line between sections
    total_structural = total_overhead + separator_overhead

    available_budget = budget - total_structural
    if available_budget < 0:
        # Budget is so small we can't even fit all headers.
        # Truncate each slice to 100 chars and mark truncated.
        parts: list[str] = []
        for s in slices:
            header = f"--- SOURCE: {s.file}.md ---"
            snippet = s.content[:100] if len(s.content) > 100 else s.content
            parts.append(f"{header}\n{snippet}\n\n[…truncated for prompt budget…]")
        return "\n\n".join(parts)

    if total_raw <= available_budget:
        # ── Everything fits — no truncation needed ────────────────────────
        parts = []
        for s in slices:
            header = f"--- SOURCE: {s.file}.md ---"
            parts.append(f"{header}\n{s.content}")
        return "\n\n".join(parts)

    # ── Proportional truncation ──────────────────────────────────────────
    # Each source gets: floor(raw_len / total_raw * available_budget)
    # Any remainder is distributed to the longest sources first.
    allocations: list[int] = []
    for s in slices:
        prop = int(len(s.content) / total_raw * available_budget)
        allocations.append(max(1, prop))  # at least 1 char per source

    # Distribute remainder
    allocated = sum(allocations)
    remainder = available_budget - allocated
    if remainder > 0:
        # Give to longest sources first (descending by original length).
        indexed = sorted(
            enumerate(slices), key=lambda x: len(x[1].content), reverse=True
        )
        for _ in range(remainder):
            idx = indexed[0][0]
            allocations[idx] += 1
            # Re-sort to maintain fairness
            indexed = sorted(
                [(i, slices[i]) for i in range(len(slices))],
                key=lambda x: len(x[1].content),
                reverse=True,
            )

    parts = []
    for i, s in enumerate(slices):
        header = f"--- SOURCE: {s.file}.md ---"
        budget_for_slice = allocations[i]
        content = s.content

        if len(content) <= budget_for_slice:
            parts.append(f"{header}\n{content}")
        else:
            truncated = content[:budget_for_slice]
            parts.append(f"{header}\n{truncated}\n\n[…truncated for prompt budget…]")

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Budget resolution
# ──────────────────────────────────────────────────────────────────────────────


def resolve_prompt_budget_chars(config: Config) -> int:
    """Resolve the effective prompt budget in characters.

    Checks the environment variable ``PROMPT_BUDGET_CHARS`` first, then falls
    back to ``config.prompt_budget_chars``.

    Args:
        config: The pipeline configuration.

    Returns:
        Integer character budget for prompt assembly.
    """
    env_val = os.getenv("PROMPT_BUDGET_CHARS")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return config.prompt_budget_chars
