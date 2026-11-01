"""Immutable generations: private workspaces, exclusive install, one commit pointer.

Only OLW-Publications is publisher-owned. Human inputs, legacy pages and policy
are read, never restored/replaced/deleted. A failed workspace is retained for
inspection. A committed generation is never revoked by recovery. File and
parent fsyncs provide POSIX ordering; power-loss/device durability is not a
claim of this module's process-crash tests.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from obsidian_llm_wiki.core.generated_paths import checked_generated_path
from obsidian_llm_wiki.core.identity_plan import canonical, digest, file_snapshot

FORMAT = "immutable-generation-v1"
NAMESPACE = "OLW-Publications"
OWNER = {"format": FORMAT, "owner": "obsidian-llm-wiki"}


def publication_root(wiki):
    return checked_generated_path(Path(wiki) / NAMESPACE, Path(wiki))


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive(path, data):
    with path.open("xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    _sync_dir(path.parent)


def _owned_root(wiki):
    root = publication_root(wiki)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        marker = checked_generated_path(root / "owner.json", root)
        if not marker.is_file() or json.loads(marker.read_bytes()) != OWNER:
            raise RuntimeError(f"publication namespace collision: {root}") from None
    else:
        _exclusive(root / "owner.json", canonical(OWNER).encode())
        _sync_dir(root.parent)
    for name in (".staging", "generations", "query-state"):
        checked_generated_path(root / name, root).mkdir(mode=0o700, exist_ok=True)
    return root


def _files(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        checked_generated_path(path, directory)
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        elif not path.is_dir():
            raise RuntimeError(f"nonregular generation entry: {path}")
    return result


class UnsupportedRegistryFormatError(RuntimeError):
    """Diagnostic inventory may report this; normal readers must refuse it."""


def registry_format_version(connection):
    """Read the exact format token; never coerce malformed version spellings."""
    import sqlite3

    try:
        rows = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchall()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"generation registry schema version unavailable: {exc}") from exc
    if len(rows) != 1 or not isinstance(rows[0][0], str) or not re.fullmatch(
        r"[1-9][0-9]*", rows[0][0]
    ):
        raise UnsupportedRegistryFormatError(
            "generation registry schema version missing or malformed"
        )
    return int(rows[0][0])


def validate_registry_format(connection):
    """Read-only compatibility gate shared by normal readers and preview."""
    from obsidian_llm_wiki.core.registry import _SCHEMA_VERSION

    version = registry_format_version(connection)
    if version != _SCHEMA_VERSION:
        raise UnsupportedRegistryFormatError(
            f"unsupported generation registry schema {version}; expected {_SCHEMA_VERSION}"
        )
    import sqlite3

    from obsidian_llm_wiki.core.registry import _SCHEMA

    # Derive required table/column types from the writer's versioned DDL, not
    # a permissive subset of whichever tables a reader happens to query. This
    # creates only an anonymous memory template, never a VaultRegistry or file.
    template = sqlite3.connect(":memory:")
    try:
        template.executescript(_SCHEMA)
        tables = template.execute("SELECT name FROM sqlite_master WHERE type='table'")
        actual = dict(connection.execute("SELECT name,type FROM sqlite_master"))
        for (table,) in tables:
            if actual.get(table) != "table":
                raise RuntimeError(f"generation registry schema missing table: {table}")
            required = set(template.execute("SELECT name,type FROM pragma_table_info(?)", (table,)))
            present = {tuple(row) for row in connection.execute(
                "SELECT name,type FROM pragma_table_info(?)", (table,)
            )}
            if not required <= present:
                missing = sorted(name for name, _ in required - present)
                raise RuntimeError(
                    f"generation registry schema invalid columns: {table}: {missing}"
                )
    finally:
        template.close()
    return version


def validate_generation(directory):
    """Check the original semantic plan against final cache/SQL/page consumers.

    SQLite is deserialized in memory: readers cannot migrate, checkpoint or
    create sidecars in a published generation. Stage-path evidence is retained
    as original evidence, not treated as an instruction to read/write there.
    This checks storage/consumer agreement, not admission or current policy.
    Callers certifying identity continuity must also verify the review receipt
    against historical source content and independently resolved review policy.
    """
    import sqlite3
    from types import SimpleNamespace

    from obsidian_llm_wiki.core.identity_plan import ProspectivePublicationPlan, registry_generation

    meta = directory / ".llmwiki"
    marker = json.loads((meta / "identity-publication.json").read_bytes())
    plan_id = marker.get("plan_id", "")
    if marker.get("status") != "committed" or not re.fullmatch(r"[0-9a-f]{64}", plan_id):
        raise RuntimeError("generation has no committed semantic plan")
    plan_data = json.loads((meta / "publications" / (plan_id + ".json")).read_bytes())
    if digest(plan_data) != plan_id:
        raise RuntimeError("semantic plan digest mismatch")
    for suffix in ("-wal", "-shm", "-journal"):
        if (meta / ("registry.db" + suffix)).exists():
            raise RuntimeError("generation registry has uncheckpointed sidecars")
    connection = sqlite3.connect(":memory:")
    try:
        raw = bytearray((meta / "registry.db").read_bytes())
        if len(raw) < 100 or not raw.startswith(b"SQLite format 3\x00"):
            raise RuntimeError("generation registry has invalid header")
        # No sidecars remain after writer close. WAL headers cannot deserialize
        # directly; adjust the private memory copy, never the published bytes.
        raw[18] = raw[19] = 1
        connection.deserialize(bytes(raw))
        connection.execute("PRAGMA query_only=ON")
        validate_registry_format(connection)
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("generation registry integrity failure")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("generation registry foreign-key failure")
        registry = SimpleNamespace(connection=connection)
        if registry_generation(registry) != marker["generation"]:
            raise RuntimeError("generation registry digest disagrees with semantic plan")
        plan = ProspectivePublicationPlan(**plan_data)
        plan.verify_consumers(
            registry, SimpleNamespace(llmwiki_dir=meta, concepts_dir=directory / "concepts")
        )
    except sqlite3.DatabaseError as exc:
        # Normalize malformed storage at the shared reader boundary, before any
        # caller can mistake a schema exception for an operational/write path.
        raise RuntimeError(f"generation registry schema/integrity failure: {exc}") from exc
    finally:
        connection.close()
    return marker


def current_publication(wiki, *, allow_uncommitted=False):
    """Resolve and validate a single pinned immutable snapshot, without writes."""
    snapshot = _publication_envelope(wiki, allow_uncommitted=allow_uncommitted)
    if snapshot:
        marker = validate_generation(snapshot[0])
        _validate_publication_inputs(snapshot[1], marker)
    return snapshot


def _validate_publication_inputs(manifest, marker):
    # Every reader must use the plan whose consumers were just validated, not
    # another archived plan selected independently by the outer pointer.
    selected = manifest.get("identity_plan_id")
    if not isinstance(selected, str) or selected != marker.get("plan_id"):
        raise ValueError("publication identity plan pointer differs from committed marker")
    from obsidian_llm_wiki.core.publication_inputs import logical_publication_inputs
    from obsidian_llm_wiki.core.source_files import validate_source_filename

    inputs = dict(logical_publication_inputs(manifest))
    required = {"04-Wiki/sources/" + validate_source_filename(name)
                for name in marker["source_payloads"]}
    if not required <= inputs.keys():
        raise ValueError("publication input inventory omits a committed source")


def _publication_envelope(wiki, *, allow_uncommitted=False):
    """Verify bytes/containment only. Diagnostic preview, NOT reader authority."""
    root = publication_root(wiki)
    path = checked_generated_path(root / "current.json", Path(wiki))
    if root.exists():
        marker = checked_generated_path(root / "owner.json", root)
        try:
            owned = json.loads(marker.read_bytes()) == OWNER
        except (OSError, ValueError):
            owned = False
        if not owned:
            raise RuntimeError(f"publication namespace collision: {root}")
    if not path.exists():
        # Ingest/query may have created only operational state. That is not a
        # pending publication. Any staged/installed generation still fails closed.
        pending = any(
            any(checked_generated_path(root / name, root).iterdir())
            for name in (".staging", "generations")
            if checked_generated_path(root / name, root).is_dir()
        )
        if pending and not allow_uncommitted:
            raise RuntimeError("pending publication: no committed generation")
        return None
    try:
        data = json.loads(path.read_bytes())
        if data["format"] != FORMAT or data["status"] != "committed":
            raise ValueError("not a committed immutable generation")
        if not re.fullmatch(r"[0-9a-f]{32}", data["generation"]):
            raise ValueError("invalid generation name")
        directory = checked_generated_path(root / "generations" / data["generation"], root)
        entrypoint = f'04-Wiki/{NAMESPACE}/generations/{data["generation"]}/index.md'
        if data.get("entrypoint") != entrypoint or not (directory / "index.md").is_file():
            raise ValueError("invalid current-generation entrypoint")
        if not directory.is_dir() or _files(directory) != data["files"]:
            raise ValueError("generation inventory/digest mismatch")
        return directory, data
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(f"corrupt or pending immutable publication: {exc}") from exc


def _preview_publication(wiki, *, allow_uncommitted=False):
    """Allow unknown-version inventory only; preserve all other refusal gates."""
    snapshot = _publication_envelope(wiki, allow_uncommitted=allow_uncommitted)
    if snapshot:
        # Preview records a blocker; this does not certify unsupported bytes.
        with suppress(UnsupportedRegistryFormatError):
            validate_generation(snapshot[0])
    return snapshot


def preview_publication_workflow(vault):
    """Read-only storage/editing inventory; not identity reassessment authority."""
    vault = Path(vault)
    wiki = checked_generated_path(vault / "04-Wiki", vault)
    root = publication_root(wiki)
    snapshot = _preview_publication(wiki, allow_uncommitted=True)
    legacy = sorted(
        p.relative_to(vault).as_posix()
        for folder in ("concepts", "entries", "mocs")
        for p in (wiki / folder).glob("*.md")
        if p.is_file()
    )
    legacy.extend(sorted(p.relative_to(vault).as_posix() for p in wiki.glob("*.md") if p.is_file()))
    annotations = checked_generated_path(vault / "03-Raw-Annotations/entities", vault)
    existing = sorted(
        p.relative_to(vault).as_posix() for p in annotations.glob("*.md") if p.is_file()
    )
    return {
        "format": FORMAT,
        "current_generation": snapshot[1]["generation"] if snapshot else None,
        "legacy_pages_retained": sorted(legacy),
        "existing_annotations_never_overwritten": existing,
        "legacy_action": "retain in place; link whole pages without attributing edits",
        "staging": "04-Wiki/OLW-Publications/.staging/<unique>/04-Wiki",
        "installation": "exclusive new directory; retain all prior generations",
        "reader": "one validated current.json; query pins only its generation",
        "annotation_template_policy": "optional exclusive creation; existing files untouched",
        "template_targets": sorted(
            "03-Raw-Annotations/entities/" + cid + ".md"
            for cid in (snapshot[1].get("entities", {}) if snapshot else {})
        ),
        "model_calls_made": 0,
        "model_call_cost": "unavailable until evidence/assessment planning; no bounded estimate",
        "storage_cost": (
            "full copies of generated pages, registry, caches and evidence per generation; "
            "no automatic pruning"
        ),
        "installed_generation_count": len(list((root / "generations").glob("*")))
        if root.exists()
        else 0,
        "native_ui_verified": False,
    }


def _install_exclusive(source, target):
    """Linux RENAME_NOREPLACE; deliberately no destructive rename fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, "renameat2", None)
    if fn is None:
        raise RuntimeError("exclusive directory install requires renameat2")
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))
    _sync_dir(target.parent)


def _checkpoint(event, **context):
    """Fault-injection seam. Production never executes external hooks."""


def _inputs(config, sources):
    paths = [
        config.vault / ".env",
        config.vault / ".llmwiki/schema.yaml",
        config.llmwiki_dir / "schema.yaml",
        config.llmwiki_dir / "equivalences.json",
    ]
    from obsidian_llm_wiki.core.publication_inputs import observable_source_paths
    from obsidian_llm_wiki.core.source_files import validate_source_filename

    paths.extend(config.sources_dir / validate_source_filename(name) for name in sources)
    paths.extend(observable_source_paths(config.vault))
    paths = sorted(set(paths))
    for path in paths:
        checked_generated_path(path, config.vault)
    return tuple((Path(name).relative_to(config.vault).as_posix(), sha)
                 for name, sha in file_snapshot(paths))


def _fingerprint(config, sources, inputs):
    # Store only the digest, never credentials or an .env copy in the generation.
    # Only actual location fields are omitted. URLs, quotes and arbitrary config
    # strings remain byte-for-byte provenance, even when they contain old paths.
    settings = asdict(config)
    settings.pop("vault_path")
    settings.pop("_vault")
    return digest(
        {
            "config": settings,
            "inputs": inputs,
            "sources": {name: asdict(source) for name, source in sources.items()},
        }
    )


def _compiler_versions(config):
    """Cover inner cache/checkpoint versions before permitting the outer no-op.

    Artifact versions already include mode-dependent producer/reviewer policy.
    Identity context is a separate inner checkpoint input, not an artifact
    version; keep it here too so context evolution triggers private replanning.
    Plan/receipt versions remain additional outer gates. Registry schema is
    checked by current_publication before this fast path is considered.
    """
    from obsidian_llm_wiki.core.identity_context import CONTEXT_VERSION
    from obsidian_llm_wiki.core.identity_plan import PLAN_VERSION
    from obsidian_llm_wiki.core.pipeline import _artifact_versions_for_config
    from obsidian_llm_wiki.synth.identity import assessor_fingerprint
    from obsidian_llm_wiki.synth.review import REVIEW_RECEIPT_VERSION

    return {
        "artifacts": asdict(_artifact_versions_for_config(config)),
        "identity_assessor": assessor_fingerprint(config),
        "identity_context": CONTEXT_VERSION,
        "identity_plan": PLAN_VERSION,
        "review_receipt": REVIEW_RECEIPT_VERSION,
    }


def _fingerprint_matches(config, sources, observed, fingerprint, manifest):
    """Known legacy fingerprint replay, without dereferencing historical roots."""
    # Older manifests without this stamp require one private revalidation. Never
    # let input equality bypass a changed producer/reviewer/identity contract.
    if manifest.get("compiler_versions") != _compiler_versions(config):
        return False
    if manifest.get("source_inputs_format") == "vault-relative-v1":
        return manifest["input_fingerprint"] == fingerprint
    from obsidian_llm_wiki.core.publication_inputs import logical_publication_inputs

    if logical_publication_inputs(manifest) != observed:
        return False
    # The mapper already proved one lexical absolute origin and the full allowed
    # namespace. Replay ONLY the two actual location fields in the old algorithm;
    # arbitrary strings (URLs, quotes, config values) remain untouched.
    origin = next(Path(name).parent for name, _ in manifest["source_inputs"]
                  if Path(name).name == ".env")
    settings = asdict(config)
    settings.update(vault_path=str(origin), _vault=origin)
    return manifest["input_fingerprint"] == digest({
        "config": settings,
        "inputs": manifest["source_inputs"],
        "sources": {name: asdict(source) for name, source in sources.items()},
    })


def _copy_tree(source, target):
    _files(source)  # fail closed on symlinks before copying any descendants
    shutil.copytree(source, target, dirs_exist_ok=True)


def _qualify_links(wiki, generation, previous=None):
    """Compatibility finalization seam; never rewrite rendered Markdown.

    Source text, evidence, human references and code are opaque. Generated
    navigation must be qualified at its typed emission site, not inferred from
    arbitrary text or basename priority after rendering.
    """
    from obsidian_llm_wiki.render.links import validate_link_inventory

    validate_link_inventory(wiki, f"04-Wiki/{NAMESPACE}/generations/{generation}")


def _annotation_links(wiki, config):
    from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

    entities = {}
    for folder in ("concepts", "mocs", "entries"):
        for page in (wiki / folder).glob("*.md"):
            text = page.read_bytes().decode("utf-8")
            cid = parse_frontmatter(text)[0].get("canonical_id", "")
            if not re.fullmatch(r"[a-z]+-[0-9a-f]{24}", str(cid)):
                continue  # entries without a logical entity ID are not entities
            entities[cid] = page.relative_to(wiki).as_posix()
            text = text.split("\n<!-- olw:annotations -->", 1)[0]
            text += "\n<!-- olw:annotations -->\n## Human notes\n\n"
            text += f"[[03-Raw-Annotations/entities/{cid}|Annotations]]\n"
            legacy = config.wiki_dir / folder / page.name
            if legacy.is_file():
                text += f"[[04-Wiki/{folder}/{page.stem}|Legacy page (retained in place)]]\n"
            page.write_text(text)
    return entities


def _annotation_templates(config, entities):
    for cid in entities:
        if not re.fullmatch(r"[a-z]+-[0-9a-f]{24}", str(cid)):
            raise RuntimeError("invalid annotation entity ID")
        target = checked_generated_path(
            config.annotations_dir / "entities" / (cid + ".md"), config.vault
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        prepared = publication_root(config.wiki_dir) / ".staging" / ("annotation-" + uuid4().hex)
        _exclusive(prepared, f"---\nentity_id: {cid}\n---\n# Human annotations\n\n".encode())
        try:
            # Install complete bytes atomically. A human file created even at
            # this last seam wins; there is never an open-for-write human path.
            _install_exclusive(prepared, target)
        except FileExistsError:
            pass  # existing human note is never read, rewritten or adopted
        finally:
            if prepared.exists():
                prepared.unlink()


async def publish_pipeline(vault_path, sources, config=None, **kwargs):
    """Run the existing semantic pipeline ONLY in a fresh private vault copy."""
    import fcntl

    from obsidian_llm_wiki.config import load_config
    from obsidian_llm_wiki.core.models import CompileResult
    from obsidian_llm_wiki.core.pipeline import _run_pipeline_workspace
    from obsidian_llm_wiki.core.source_files import source_file_path
    from obsidian_llm_wiki.ingest.sources import load_source_file

    vault = Path(vault_path).resolve()
    templates = kwargs.pop("annotation_templates", False)
    config = config or load_config(VAULT_PATH=str(vault))
    if config.vault != vault:
        raise ValueError("Pipeline vault does not match Config.vault")
    # Validate the complete legacy SQLite family before source/policy reads or
    # publisher namespace creation. exists()/stat() would follow a sidecar link.
    legacy_metadata = checked_generated_path(config.llmwiki_dir, vault)
    for suffix in ("", "-wal", "-shm", "-journal"):
        checked_generated_path(legacy_metadata / ("registry.db" + suffix), vault)
    # Materialize lazy source bodies from human inputs without opening mutable DBs.
    sources = dict(sources)
    for name, source in sources.items():
        if not source.content:
            loaded = load_source_file(source_file_path(config.sources_dir, name))
            if loaded is None:
                raise RuntimeError(f"missing source snapshot: {name}")
            sources[name] = loaded
    observed = _inputs(config, sources)
    fingerprint = _fingerprint(config, sources, observed)
    root = _owned_root(config.wiki_dir)
    lock = checked_generated_path(root / "writer.lock", root)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return CompileResult(errors=["lock: another compilation is running"])
        previous = current_publication(config.wiki_dir, allow_uncommitted=True)
        if (
            previous
            and _fingerprint_matches(config, sources, observed, fingerprint, previous[1])
            and not previous[1]["errors"]
            and not kwargs.get("force")
            and not kwargs.get("force_filenames")
            and not kwargs.get("refresh_render")
        ):
            if templates:
                _annotation_templates(config, previous[1].get("entities", {}))
            return CompileResult(skipped=len(sources))
        generation = uuid4().hex
        work = root / ".staging" / generation
        work.mkdir(mode=0o700)
        working = replace(config, vault_path=str(work), _vault=None)
        working.wiki_dir.mkdir()
        if previous:
            _copy_tree(previous[0], working.wiki_dir)
        elif config.llmwiki_dir.exists():
            # Bootstrap identity/cache evidence, never copy legacy human pages.
            for suffix in ("-wal", "-journal"):
                sidecar = config.llmwiki_dir / ("registry.db" + suffix)
                if sidecar.exists() and sidecar.stat().st_size:
                    raise RuntimeError(
                        "legacy registry is not a closed snapshot; migration preview required"
                    )
            _copy_tree(config.llmwiki_dir, working.llmwiki_dir)
        # Policies are read-only inputs. Staging uses the current versions.
        for relative in (
            "04-Wiki/.llmwiki/schema.yaml",
            "04-Wiki/.llmwiki/equivalences.json",
            ".llmwiki/schema.yaml",
        ):
            original = vault / relative
            target = work / relative
            if original.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
            elif target.exists():
                target.unlink()
        # Previous source copies are not live inputs; replace only private copies.
        if working.sources_dir.exists():
            shutil.rmtree(working.sources_dir)
        working.sources_dir.mkdir(parents=True)
        for name, source in sources.items():
            original = source_file_path(config.sources_dir, name)
            target = source_file_path(working.llmwiki_dir / "source-inputs", name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if original.is_file():
                target.write_bytes(original.read_bytes())
                loaded = load_source_file(target)
                if loaded is None or loaded.content.strip() != source.content.strip():
                    raise RuntimeError(f"stale source snapshot before staging: {name}")
            else:
                target.write_text(source.content)
        if _inputs(config, sources) != observed:
            raise RuntimeError("source or policy changed during staging; retry")
        _checkpoint("workspace_ready", work=work, generation=generation)
        options = dict(kwargs)
        options["unchanged_filenames"] = set()
        options["source_file_signatures"] = {}
        options["refresh_render"] = True  # every new generation is a complete projection
        options["link_prefix"] = f"04-Wiki/{NAMESPACE}/generations/{generation}"
        result = await _run_pipeline_workspace(work, sources, working, **options)
        marker = working.llmwiki_dir / "identity-publication.json"
        if not marker.is_file() or json.loads(marker.read_bytes()).get("status") != "committed":
            # No accepted plan (e.g. busy/internal failure): retain diagnostic workspace.
            return result
        _qualify_links(
            working.wiki_dir, generation, previous[1]["generation"] if previous else None
        )
        entities = _annotation_links(working.wiki_dir, config)
        # Our private presentation edits, not human adoption. Carry positive
        # ownership forward for the next staged copy's refresh.
        hashes_path = working.llmwiki_dir / "render-page-hashes.json"
        hashes = json.loads(hashes_path.read_text())
        hashes_path.write_text(
            canonical(
                {
                    name: hashlib.sha256((working.wiki_dir / name).read_bytes()).hexdigest()
                    for name in hashes
                    if (working.wiki_dir / name).is_file()
                }
            )
        )
        if _inputs(config, sources) != observed:
            raise RuntimeError("source or policy changed before publication; retry")
        semantic_marker = validate_generation(working.wiki_dir)
        files = _files(working.wiki_dir)
        manifest = {
            "format": FORMAT,
            "status": "committed",
            "generation": generation,
            "input_fingerprint": fingerprint,
            "compiler_versions": _compiler_versions(config),
            "files": files,
            "errors": result.errors,
            "entities": entities,
            "source_inputs": observed,
            "source_inputs_format": "vault-relative-v1",
            "identity_plan_id": semantic_marker["plan_id"],
            "previous": previous[1]["generation"] if previous else None,
            "entrypoint": f"04-Wiki/{NAMESPACE}/generations/{generation}/index.md",
        }
        # Flush every file/directory before the exclusive directory install.
        for name in files:
            with (working.wiki_dir / name).open("rb") as handle:
                os.fsync(handle.fileno())
        for directory in sorted(
            (p for p in working.wiki_dir.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            _sync_dir(directory)
        _sync_dir(working.wiki_dir)
        target = root / "generations" / generation
        _checkpoint("before_install", work=work, generation=generation, target=target)
        # Empty operational directories may be cleaned up during slow inference.
        # Revalidate ownership/containment and restore them at the install boundary,
        # not by catching ENOENT (which could hide a missing staged generation).
        _owned_root(config.wiki_dir)
        _install_exclusive(working.wiki_dir, target)
        _checkpoint("after_install", generation=generation, target=target)
        pending = root / (".commit-" + generation + ".json")
        _exclusive(pending, canonical(manifest).encode())
        _checkpoint("before_commit", generation=generation, target=target)
        # This pointer is app-owned, never a human homepage or legacy note.
        os.replace(pending, root / "current.json")
        _sync_dir(root)
        _checkpoint("after_commit", generation=generation, target=target)
        if templates:
            _annotation_templates(config, entities)
        result.pages = [target / Path(p).relative_to(working.wiki_dir) for p in result.pages]
        return result
    finally:
        os.close(fd)  # flock released even on exceptions; never unlink a lock path
