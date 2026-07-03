"""OKF bundle import/export — tarball serialization for OKF bundles.

Exports an OKF bundle directory to a (optionally compressed) tarball,
excluding internal state/cache artifacts, and imports a tarball back into a
vault directory, with optional lint verification on import.

Public entry points: :func:`export_bundle`, :func:`import_bundle`.
Exported constant: :data:`EXCLUDED_NAMES` — path components filtered out
during export.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from pipeline.okf_lint import lint_bundle

__all__ = ["EXCLUDED_NAMES", "export_bundle", "import_bundle"]

# Path components (directory/file basenames) that are excluded from the
# exported tarball.  These are internal pipeline artifacts, caches, or VCS
# metadata that should not travel with a portable OKF bundle.
EXCLUDED_NAMES: frozenset[str] = frozenset({
    ".llmwiki",       # pipeline state, candidates, locks
    ".git",           # version control metadata
    "__pycache__",    # Python bytecode cache
    "compile.lock",   # compile-time PID lock file
    "lock",           # generic lock file
    "state.json",     # pipeline state file (redundant safety net)
    "candidates",     # review candidates directory content
})


# ── Helpers ───────────────────────────────────────────────────────────────


def _should_exclude(path: Path, root: Path) -> bool:
    """Return True if ``path`` should be excluded from the tarball.

    A file/directory is excluded if any path component between it and the
    bundle ``root`` (inclusive of its own basename) is in
    :data:`EXCLUDED_NAMES`.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part in EXCLUDED_NAMES for part in rel.parts)


def _collect_files(root: Path) -> list[Path]:
    """Recursively collect all files under ``root`` that are not excluded."""
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not _should_exclude(p, root):
            files.append(p)
    return files


# ── Export ────────────────────────────────────────────────────────────────


def export_bundle(
    bundle_dir: str | Path,
    output_path: str | Path | None = None,
    compress: bool = True,
) -> Path:
    """Export an OKF bundle directory to a tarball.

    Args:
        bundle_dir: Path to the OKF bundle root (e.g. ``vault/04-Wiki``).
        output_path: Destination tarball path.  Defaults to
            ``<parent>/<bundle_name>.tar.gz`` (or ``.tar`` when
            ``compress`` is ``False``).
        compress: When ``True`` (default), write a gzipped ``.tar.gz``;
            when ``False``, write an uncompressed ``.tar``.

    Returns:
        The :class:`~pathlib.Path` to the written tarball.
    """
    bd = Path(bundle_dir)
    if not bd.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bd}")

    if output_path is None:
        ext = ".tar.gz" if compress else ".tar"
        out = bd.parent / f"{bd.name}{ext}"
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)

    mode = "w:gz" if compress else "w"
    files = _collect_files(bd)

    with tarfile.open(out, mode) as tar:
        for fpath in files:
            arcname = str(fpath.relative_to(bd.parent))
            tar.add(str(fpath), arcname=arcname)

    return out


# ── Import ───────────────────────────────────────────────────────────────


def _lint_report_to_dict(report) -> dict:
    """Convert a LintReport dataclass to a plain dict for the return value."""
    return {
        "passed": report.passed,
        "errors": report.errors,
        "warnings": report.warnings,
        "files_checked": report.files_checked,
        "issues": [
            {
                "severity": issue.severity,
                "file": issue.file,
                "line": issue.line,
                "rule": issue.rule,
                "message": issue.message,
            }
            for issue in report.issues
        ],
    }


def _safe_extract(tar: tarfile.TarFile, target: str) -> None:
    """Extract a tarball safely, rejecting members with unsafe paths.

    This is a manual mitigation for CVE-2007-4559 (path traversal via
    tarfile.extractall) used as a fallback on Python 3.11 where the
    built-in ``filter`` parameter is not available.

    Rejects any member whose name:
      - Is an absolute path (starts with ``/``).
      - Contains a ``..`` path component (which could escape the target).
      - Contains a Windows-style drive prefix (e.g. ``C:\\``).
    """
    base = Path(target).resolve()
    for member in tar.getmembers():
        # Reject absolute paths.
        member_path = member.name
        if member_path.startswith("/") or member_path.startswith("\\"):
            raise ValueError(f"Refusing to extract absolute path: {member.name!r}")
        # Reject Windows drive letters (e.g. C:\foo).
        if len(member_path) > 1 and member_path[1] == ":" and member_path[2] in ("\\", "/"):
            raise ValueError(f"Refusing to extract drive-prefixed path: {member.name!r}")
        # Reject path-traversal via .. components.
        parts = member_path.replace("\\", "/").split("/")
        if ".." in parts:
            raise ValueError(f"Refusing to extract path with '..' component: {member.name!r}")
        # Verify the resolved destination stays within the target directory.
        dest = (base / member_path).resolve()
        try:
            dest.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Refusing to extract outside target: {member.name!r}") from exc
    tar.extractall(str(target))


def import_bundle(
    tarball: str | Path,
    target_dir: str | Path,
    verify: bool = True,
) -> dict:
    """Import an OKF tarball into a target directory.

    Extracts the tarball, creates the internal ``.llmwiki/`` state directory,
    then (unless ``verify`` is ``False``) runs the OKF linter over the
    extracted bundle and includes the lint report in the returned dict.

    Args:
        tarball: Path to the tarball file (``.tar``, ``.tar.gz``, etc.).
        target_dir: Directory to extract into.  Created if it doesn't exist.
        verify: When ``True`` (default), run lint verification after
            extraction and include the report in the result dict.

    Returns:
        A dict with keys:
          - ``"bundle_path"``: :class:`Path` to the extracted bundle.
          - ``"lint_report"`` (only when ``verify=True``): lint report dict
            with keys ``passed``, ``errors``, ``warnings``, ``files_checked``,
            ``issues``.
    """
    tb = Path(tarball)
    if not tb.is_file():
        raise FileNotFoundError(f"Tarball not found: {tb}")

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # ``"r"`` auto-detects compression (gz, bz2, xz, or plain).
    with tarfile.open(tb, "r") as tar:
        names = tar.getnames()
        try:
            tar.extractall(str(target), filter="data")  # py3.12+
        except TypeError:
            # py3.11 fallback — use a manual filter to reject path traversal.
            _safe_extract(tar, str(target))

    # Determine the extracted bundle directory.
    # The tarball may contain a single top-level directory (the bundle) or
    # multiple top-level entries (files + dirs).  If there's exactly one
    # top-level entry and it's a directory, use it; otherwise the bundle
    # root *is* the target directory itself.
    if names:
        top_components = {n.split("/")[0] for n in names if n}
        if len(top_components) == 1:
            candidate = target / next(iter(top_components))
            # Only treat the single top-level entry as the bundle dir if it
            # is actually a directory; if it's a file, the bundle root is
            # the target itself.
            extracted = candidate if candidate.is_dir() else target
        else:
            # Multiple top-level entries — the bundle root is target itself.
            extracted = target
    else:
        extracted = target

    # Create the internal state directory so the imported bundle is
    # immediately usable by the pipeline.
    if extracted.is_dir():
        (extracted / ".llmwiki").mkdir(parents=True, exist_ok=True)

    result: dict = {"bundle_path": extracted}

    if verify and extracted.is_dir():
        report = lint_bundle(extracted)
        result["lint_report"] = _lint_report_to_dict(report)

    return result
