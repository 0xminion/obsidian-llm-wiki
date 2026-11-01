"""Presentation-only normalization for human CLI text assertions."""
import re


def unstyle(text: str) -> str:
    """Remove ANSI SGR styling without normalizing content or whitespace."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)
