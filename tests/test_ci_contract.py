"""CI workflow contract tests."""

from pathlib import Path


def test_ci_workflow_runs_truthful_quality_gate():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "ruff check ." in workflow
    assert "pyflakes pipeline tests" in workflow
    assert "pytest -q" in workflow
    assert "python -m build --wheel" in workflow
    assert "pipeline init /tmp/obsidian-wheel-vault" in workflow
    assert "Meta/prompts/batch-create.prompt" in workflow
    assert "Meta/Templates/Entry.md" in workflow
