"""Tests for pipeline.bundle_io — OKF bundle export/import."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from pipeline.bundle_io import EXCLUDED_NAMES, export_bundle, import_bundle

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_bundle(bundle_dir: Path) -> None:
    """Create a minimal but valid OKF bundle inside ``bundle_dir``."""
    for subdir in ("sources", "entries", "concepts", "mocs", "references"):
        (bundle_dir / subdir).mkdir(parents=True, exist_ok=True)
    # Root index.md with okf_version frontmatter (passes OKF-006).
    (bundle_dir / "index.md").write_text(
        "---\nokf_version: '0.1'\n---\n# Knowledge Bundle\n",
        encoding="utf-8",
    )
    # A conformant concept file.
    (bundle_dir / "concepts" / "foo.md").write_text(
        "---\n"
        "type: Concept\n"
        "title: Foo\n"
        "tags:\n"
        "- alpha\n"
        "timestamp: 2025-01-02T10:30:00\n"
        "---\n\n# Foo\n",
        encoding="utf-8",
    )


def _make_full_bundle(bundle_dir: Path) -> None:
    """Create a bundle that includes files which must be excluded from export."""
    _make_bundle(bundle_dir)
    # State directory — must be excluded.
    state_dir = bundle_dir / ".llmwiki"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text('{"sources": {}}', encoding="utf-8")
    # Compile lock — must be excluded.
    (bundle_dir / "compile.lock").write_text("locked", encoding="utf-8")
    # Bytecode cache — must be excluded.
    pycache = bundle_dir / "__pycache__"
    pycache.mkdir(parents=True, exist_ok=True)
    (pycache / "module.cpython-311.pyc").write_text("bytecode", encoding="utf-8")
    # Git directory — must be excluded.
    git_dir = bundle_dir / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")


def _archive_contains(tarball: Path, member_name: str) -> bool:
    """Return True if ``member_name`` is present in the tarball."""
    with tarfile.open(str(tarball), "r:gz" if str(tarball).endswith(".gz") else "r") as tar:
        return member_name in {m.name for m in tar.getmembers()}


# ── export_bundle ────────────────────────────────────────────────────────


def test_export_bundle_creates_tar_gz(tmp_path: Path):
    """export_bundle should produce a .tar.gz file by default."""
    bundle_dir = tmp_path / "mybundle"
    _make_bundle(bundle_dir)

    tarball = export_bundle(bundle_dir)

    assert tarball.exists()
    assert tarball.suffix == ".gz"
    assert tarball.name == "mybundle.tar.gz"
    # Verify it is a valid gzipped tarball.
    with tarfile.open(str(tarball), "r:gz") as tar:
        names = {m.name for m in tar.getmembers()}
    # Bundle files should be archived under the bundle name prefix.
    assert "mybundle/index.md" in names
    assert "mybundle/concepts/foo.md" in names


def test_export_bundle_uncompressed(tmp_path: Path):
    """When compress=False, the tarball should be an uncompressed .tar."""
    bundle_dir = tmp_path / "mybundle"
    _make_bundle(bundle_dir)

    tarball = export_bundle(bundle_dir, compress=False)
    assert tarball.exists()
    assert tarball.suffix == ".tar"
    # Should be readable as a plain tar.
    with tarfile.open(str(tarball), "r") as tar:
        assert len(tar.getmembers()) > 0


def test_export_bundle_custom_output_path(tmp_path: Path):
    """export_bundle should honour an explicit output_path."""
    bundle_dir = tmp_path / "mybundle"
    _make_bundle(bundle_dir)
    out = tmp_path / "custom" / "release.tar.gz"

    tarball = export_bundle(bundle_dir, output_path=out)
    assert tarball == out
    assert tarball.exists()


def test_export_bundle_nonexistent_dir(tmp_path: Path):
    """export_bundle should raise FileNotFoundError for a missing bundle dir."""
    with pytest.raises(FileNotFoundError):
        export_bundle(tmp_path / "does_not_exist")


def test_export_bundle_excludes_llmwiki(tmp_path: Path):
    """The .llmwiki/ directory must not appear in the exported tarball."""
    bundle_dir = tmp_path / "mybundle"
    _make_full_bundle(bundle_dir)

    tarball = export_bundle(bundle_dir)
    assert tarball.exists()

    with tarfile.open(str(tarball), "r:gz") as tar:
        names = {m.name for m in tar.getmembers()}

    # None of the excluded paths should be present.
    for member in names:
        for excluded in EXCLUDED_NAMES:
            assert excluded not in member.split("/"), (
                f"Excluded path component '{excluded}' found in member '{member}'"
            )


def test_export_bundle_excludes_compile_lock_and_pycache(tmp_path: Path):
    """compile.lock and __pycache__ must be excluded from the tarball."""
    bundle_dir = tmp_path / "mybundle"
    _make_full_bundle(bundle_dir)

    tarball = export_bundle(bundle_dir)

    with tarfile.open(str(tarball), "r:gz") as tar:
        names = {m.name for m in tar.getmembers()}

    assert not any("compile.lock" in m for m in names)
    assert not any("__pycache__" in m for m in names)
    assert not any(".git" in m for m in names)
    # Sanity: the real bundle content is still there.
    assert any("index.md" in m for m in names)


# ── import_bundle ───────────────────────────────────────────────────────


def test_import_bundle_extracts_and_verifies(tmp_path: Path):
    """import_bundle extracts the tarball, inits state dir, and returns a lint report."""
    bundle_dir = tmp_path / "mybundle"
    _make_bundle(bundle_dir)

    tarball = export_bundle(bundle_dir, output_path=tmp_path / "mybundle.tar.gz")
    target = tmp_path / "imported"

    result = import_bundle(tarball, target, verify=True)

    assert "bundle_path" in result
    assert "lint_report" in result
    bundle_path = Path(result["bundle_path"])
    assert bundle_path.is_dir()
    assert (bundle_path / "index.md").exists()
    assert (bundle_path / "concepts" / "foo.md").exists()
    # State directory should have been created.
    assert (bundle_path / ".llmwiki").is_dir()
    # Lint report should indicate the bundle was scanned.
    report = result["lint_report"]
    assert isinstance(report, dict)
    assert "passed" in report
    assert "files_checked" in report
    assert report["files_checked"] >= 2  # index.md + concepts/foo.md


def test_import_bundle_no_verify_skips_lint(tmp_path: Path):
    """When verify=False, no lint_report key should be present."""
    bundle_dir = tmp_path / "mybundle"
    _make_bundle(bundle_dir)

    tarball = export_bundle(bundle_dir, output_path=tmp_path / "mybundle.tar.gz")
    target = tmp_path / "imported"

    result = import_bundle(tarball, target, verify=False)

    assert "bundle_path" in result
    assert "lint_report" not in result
    bundle_path = Path(result["bundle_path"])
    assert (bundle_path / "index.md").exists()
    assert (bundle_path / ".llmwiki").is_dir()


def test_import_bundle_creates_target_dir(tmp_path: Path):
    """import_bundle should create the target directory if it does not exist."""
    bundle_dir = tmp_path / "mybundle"
    _make_bundle(bundle_dir)

    tarball = export_bundle(bundle_dir, output_path=tmp_path / "mybundle.tar.gz")
    target = tmp_path / "deep" / "nested" / "imported"

    result = import_bundle(tarball, target, verify=False)
    assert Path(result["bundle_path"]).is_dir()


def test_import_bundle_nonexistent_tarball(tmp_path: Path):
    """import_bundle should raise FileNotFoundError for a missing tarball."""
    with pytest.raises(FileNotFoundError):
        import_bundle(tmp_path / "nope.tar.gz", tmp_path / "target")


def test_roundtrip_preserves_content(tmp_path: Path):
    """Export then import should preserve all bundle content."""
    bundle_dir = tmp_path / "orig"
    _make_bundle(bundle_dir)
    # Add a references file for extra coverage.
    (bundle_dir / "references" / "r1.md").write_text(
        "---\ntype: Reference\n---\n# Ref\n",
        encoding="utf-8",
    )

    tarball = export_bundle(bundle_dir, output_path=tmp_path / "roundtrip.tar.gz")
    target = tmp_path / "restored"
    result = import_bundle(tarball, target, verify=True)

    bundle_path = Path(result["bundle_path"])
    # All original files should be present.
    assert (bundle_path / "index.md").exists()
    assert (bundle_path / "concepts" / "foo.md").exists()
    assert (bundle_path / "references" / "r1.md").exists()
    # index.md content should be preserved.
    content = (bundle_path / "index.md").read_text(encoding="utf-8")
    assert "okf_version" in content


# ── pytest entry ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
