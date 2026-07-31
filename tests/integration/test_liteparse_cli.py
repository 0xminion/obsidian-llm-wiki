"""Real CLI contract for the optional LiteParse installation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.pdf"


@pytest.mark.integration
def test_liteparse_cli_parses_minimal_pdf_to_nonempty_markdown() -> None:
    """The installed ``lit`` CLI parses a real PDF without timing out."""
    lit = shutil.which("lit")
    if lit is None:
        pytest.skip("LiteParse CLI is not installed")

    command = [
        lit,
        "parse",
        str(FIXTURE),
        "--format",
        "text",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"LiteParse CLI timed out after {exc.timeout} seconds")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "LiteParse CLI produced no Markdown/text output"
