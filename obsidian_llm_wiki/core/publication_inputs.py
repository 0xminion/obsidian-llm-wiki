"""Logical publication inputs, separate from opaque historical evidence paths.

Only this vault's allowlisted input namespace is read. Legacy absolute locators
are parsed lexically under one consistent origin, NEVER resolved or opened.
Results describe observable file inputs, not upstream/API freshness or semantic
admission. Callers must validate the committed generation before using them.
"""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from obsidian_llm_wiki.core.generated_paths import checked_generated_path
from obsidian_llm_wiki.core.source_files import validate_source_filename

INPUTS_FORMAT = "vault-relative-v1"
POLICY_INPUTS = frozenset({
    ".env", ".llmwiki/schema.yaml", "04-Wiki/.llmwiki/schema.yaml",
    "04-Wiki/.llmwiki/equivalences.json",
})


def _lexical_path(value):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError("invalid publication input locator")
    path = PurePosixPath(value)
    if "\\" in value or ":" in value or ".." in path.parts or path.as_posix() != value:
        raise ValueError("noncanonical publication input locator")
    return path


def _logical_path(value):
    path = _lexical_path(value)
    if value in POLICY_INPUTS:
        return value
    if path.parts[:2] == ("04-Wiki", "sources") and len(path.parts) == 3:
        validate_source_filename(path.name)
        return value
    raise ValueError("publication input outside the allowed logical namespace")


def logical_publication_inputs(manifest: Mapping) -> tuple[tuple[str, str | None], ...]:
    """Validate every row before IO; map only the two explicitly known formats."""
    if not isinstance(manifest, Mapping) or manifest.get("format") != "immutable-generation-v1":
        raise ValueError("unknown publication envelope format")
    if manifest.get("status") != "committed":
        raise ValueError("publication inputs are not committed")
    version = manifest.get("source_inputs_format", "legacy-absolute-v1")
    if version not in (INPUTS_FORMAT, "legacy-absolute-v1") or (
        version == "legacy-absolute-v1" and "source_inputs_format" in manifest
    ):
        raise ValueError("unknown publication input format")
    rows = manifest.get("source_inputs")
    if not isinstance(rows, (list, tuple)):
        raise ValueError("publication input inventory missing")
    parsed = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("invalid publication input row")
        name, sha = row
        path = _lexical_path(name)
        if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise ValueError("invalid publication input digest")
        parsed.append((path, sha))
    origin = None
    if version == "legacy-absolute-v1":
        anchors = [p for p, _ in parsed if p.name == ".env"]
        if len(anchors) != 1 or not anchors[0].is_absolute():
            raise ValueError("legacy publication input origin missing or ambiguous")
        origin = anchors[0].parent
    result = {}
    for path, sha in parsed:
        if origin is not None:
            try:
                path = path.relative_to(origin)
            except ValueError as exc:
                raise ValueError("inconsistent legacy publication input origins") from exc
        name = _logical_path(path.as_posix())
        if name in result:
            raise ValueError("duplicate logical publication input")
        result[name] = sha
    if not result.keys() >= POLICY_INPUTS:
        raise ValueError("publication input inventory missing required policy anchors")
    return tuple(sorted(result.items()))


def observable_source_paths(vault: Path) -> tuple[Path, ...]:
    """Use the loader's inventory rules, validating every child before discovery."""
    from obsidian_llm_wiki.ingest.sources import _discover_source_paths

    root = checked_generated_path(Path(vault) / "04-Wiki/sources", Path(vault))
    if not root.exists():
        return ()
    for child in root.iterdir():
        checked_generated_path(child, Path(vault))
    return tuple(_discover_source_paths(root))


def publication_input_drift(vault: Path, manifest: Mapping) -> tuple[str, ...]:
    """Return changed logical locators; malformed or unsafe inventories raise."""
    vault = Path(vault)
    inputs = logical_publication_inputs(manifest)
    checked_generated_path(vault, vault)
    if not vault.is_dir():
        raise ValueError("publication input vault is missing or not a directory")
    # Validate the complete path set before any content/stat read. Do not resolve
    # a source basename before checking symlink components (including parents).
    paths = [(name, checked_generated_path(vault / name, vault), sha) for name, sha in inputs]
    recorded = {name for name, _ in inputs}
    discovered = {_logical_path(p.relative_to(vault).as_posix())
                  for p in observable_source_paths(vault)}
    changed = sorted(discovered - recorded)
    for name, path, expected in paths:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            actual = None
        else:
            if not stat.S_ISREG(mode):
                raise ValueError(f"nonregular publication input: {name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            changed.append(name)
    return tuple(changed)


def input_freshness(vault_root, manifest) -> dict:
    """JSON result: current/drifted/unverifiable; no writes or provider calls.

    Recorded absent API source files remain captured snapshots. No observation
    here proves that an upstream URL/API has remained unchanged.
    """
    try:
        changed = list(publication_input_drift(Path(vault_root), manifest))
    except (ValueError, OSError, TypeError, RuntimeError) as exc:
        return {"status": "unverifiable", "changed_paths": [], "reasons": [str(exc)]}
    unobserved = [name for name, digest in logical_publication_inputs(manifest)
                  if name.startswith('04-Wiki/sources/') and digest is None]
    result = {"status": "drifted" if changed else "current", "changed_paths": changed,
              "reasons": ["source or policy file drift"] if changed else []}
    if unobserved:
        result.update(input_scope='captured_only', unobserved_sources=unobserved)
        result['reasons'].append('API-only captured sources have no observable live source file')
    return result
