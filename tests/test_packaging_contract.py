"""Packaging contract tests that keep wheel installs honest without building a wheel."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def test_pyproject_includes_packaged_assets():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    wheel = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["pipeline"]
    assert "pipeline/assets/**/*" in wheel["include"]


def test_packaged_asset_directories_are_canonical_and_populated():
    asset_root = Path("pipeline/assets")
    prompts = sorted((asset_root / "prompts").glob("*"))
    templates = sorted((asset_root / "templates").glob("*"))

    assert prompts
    assert templates
    assert (asset_root / "prompts" / "batch-create.prompt").read_text(encoding="utf-8")
    assert (asset_root / "templates" / "Entry.md").read_text(encoding="utf-8")
