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


def test_packaged_asset_directories_match_source_assets():
    asset_root = Path("pipeline/assets")
    for source_dir_name in ["prompts", "templates"]:
        source_files = sorted(Path(source_dir_name).glob("*"))
        packaged_files = sorted((asset_root / source_dir_name).glob("*"))

        assert source_files
        assert [p.name for p in packaged_files] == [p.name for p in source_files]
        for source_file in source_files:
            packaged_file = asset_root / source_dir_name / source_file.name
            assert packaged_file.read_bytes() == source_file.read_bytes()
