"""Recover only journalled app writes whose exact postimage still owns the path.

The journal's committed decision is the visibility fence. A pending journal is
not a committed generation, even if its last file write was a committed marker.
Unmanaged files are never enumerated or restored. Conflicts stay in place and
block recovery, with path-aware before/after/observed metadata in the journal.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from contextlib import closing
from contextvars import ContextVar
from pathlib import Path

from obsidian_llm_wiki.core.generated_paths import checked_generated_path

_ACTIVE: ContextVar = ContextVar("publication_transaction", default=None)
JOURNAL_VERSION = "owned-publication-writes-v3"


def _state(path):
    if not path.exists():
        return None
    st = path.stat()
    if not stat.S_ISREG(st.st_mode):
        raise RuntimeError(f"publication path is not a regular file: {path}")
    try:
        xattrs = {name: os.getxattr(path, name).hex() for name in os.listxattr(path)}
    except (AttributeError, OSError) as exc:
        raise RuntimeError(f"cannot inspect publication metadata: {path}: {exc}") from exc
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mode": stat.S_IMODE(st.st_mode), "mtime_ns": st.st_mtime_ns,
            "size": st.st_size, "uid": st.st_uid, "gid": st.st_gid, "xattrs": xattrs}


def _restore_metadata(fd, metadata):
    """Exact POSIX ownership, mode, binary xattrs (including POSIX ACLs).

    Permission/unsupported-metadata errors are fatal before rename. Inode,
    ctime, birthtime and atime are not promised by atomic file restoration.
    """
    current = os.fstat(fd)
    if (current.st_uid, current.st_gid) != (metadata["uid"], metadata["gid"]):
        os.fchown(fd, metadata["uid"], metadata["gid"])
    os.fchmod(fd, metadata["mode"])
    for name in os.listxattr(fd):
        if name not in metadata["xattrs"]:
            os.removexattr(fd, name)
    for name, value in metadata["xattrs"].items():
        os.setxattr(fd, name, bytes.fromhex(value))
    os.utime(fd, ns=(metadata["mtime_ns"], metadata["mtime_ns"]))


def _atomic_bytes(path, content, metadata=None):
    """Unjournalled journal/recovery write; preserve exact bytes and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publication-")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(content)
            out.flush()
            if metadata:
                _restore_metadata(out.fileno(), metadata)
            os.fsync(out.fileno())
        if metadata and _state(Path(temporary)) != metadata:
            raise RuntimeError(f"publication metadata restoration failed: {path}")
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save(path, data):
    _atomic_bytes(path, json.dumps(data, sort_keys=True).encode())


def publication_pending(llmwiki_dir):
    """Read-only status used before a reader opens/migrates the registry."""
    stage = Path(llmwiki_dir) / "publication-stage"
    manifest = checked_generated_path(stage / "manifest.json", Path(llmwiki_dir))
    if not manifest.exists():
        return False  # staging interrupted before any application write
    try:
        return json.loads(manifest.read_bytes()).get("status") != "committed"
    except (OSError, ValueError, TypeError, AttributeError):
        return True


def require_publication_ready(llmwiki_dir):
    if publication_pending(llmwiki_dir):
        raise RuntimeError(
            "pending or conflicted publication recovery; run build to recover, "
            "or inspect publication-stage/manifest.json before reading this generation"
        )


def _relative(config, path, *, source_output=False):
    path = checked_generated_path(path, config.vault)
    try:
        rel = path.relative_to(config.wiki_dir)
    except ValueError:
        return None
    # Policy, source inputs and append-only evidence are not rollback outputs.
    if rel.parts[0] == "sources" and rel.name != "index.md" and not source_output:
        return None
    if rel.parts[0] == ".llmwiki" and len(rel.parts) > 1 and (
        rel.parts[1] in {
            "publication-stage", "recovery-conflicts", "cache-history", "publications",
            "metrics.json", "compile.lock", "lock", "schema.yaml", "equivalences.json",
        } or rel.parts[1].startswith("registry.db") or path.name.endswith(".lock")
    ):
        return None
    return str(rel)


def publication_write_transaction_active():
    tx = _ACTIVE.get()
    return tx is not None and tx.active


def generated_source_output(path):
    """Renderer admission, after its existing-page ownership check succeeded.

    No glob or source filename pattern grants ownership. Human source inputs
    that merely resemble index shards are never admitted by this hook.
    """
    tx = _ACTIVE.get()
    if tx is not None and tx.active and Path(path).parent == tx.config.sources_dir:
        checked_generated_path(Path(path), tx.config.vault)
        tx.source_outputs.add(str(path))


def record_replace(path, temporary):
    """Called immediately before an app's atomic rename, never by human IO."""
    tx = _ACTIVE.get()
    if tx is not None and tx.active:
        tx.record(path, _state(Path(temporary)))


def preserve_replacement_metadata(path, temporary):
    """Carry existing editor metadata forward; retain the new content's mtime."""
    before = _state(Path(path))
    if before is not None:
        metadata = {**before, "mtime_ns": Path(temporary).stat().st_mtime_ns}
        with Path(temporary).open("r+b") as out:
            _restore_metadata(out.fileno(), metadata)
            os.fsync(out.fileno())


def record_delete(path):
    tx = _ACTIVE.get()
    if tx is not None and tx.active:
        tx.record(path, None)


def _registry_schema(connection):
    schema = list(connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ))
    tables = {row[1] for row in schema if row[0] == "table"}
    if not {"meta", "source", "synthesis", "concept", "moc"} <= tables:
        raise RuntimeError("publication registry backup has incomplete schema")
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def _load_registry_backup(stage, binding):
    """Pin read-only backup bytes, then validate before touching live outputs.

    Deserialize a private copy, not a second pathname open: SQLite can neither
    create an external database nor consult/create sidecars on the source.
    """
    path = checked_generated_path(stage / "registry-before.db", stage)
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = checked_generated_path(Path(str(path) + suffix), stage)
        if sidecar.exists():
            raise RuntimeError(f"publication registry backup has unexpected sidecar: {sidecar}")
    saved = None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise RuntimeError("publication registry backup is not a regular file")
            payload = source.read()
        if not binding or hashlib.sha256(payload).hexdigest() != binding.get("sha256"):
            raise RuntimeError("publication registry backup digest mismatch or missing binding")
        saved = sqlite3.connect(":memory:")
        saved.deserialize(payload)
        saved.execute("PRAGMA query_only=ON")
        if saved.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("publication registry backup integrity failure")
        if _registry_schema(saved) != binding.get("schema_sha256"):
            raise RuntimeError("publication registry backup schema mismatch")
        if saved.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("publication registry backup foreign key failure")
        return saved
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        if saved is not None:
            saved.close()
        raise RuntimeError(f"publication registry backup preflight failed: {exc}") from exc


def recover_publication(registry, config):
    stage = checked_generated_path(config.llmwiki_dir / "publication-stage", config.vault)
    manifest = checked_generated_path(stage / "manifest.json", config.vault)
    if not manifest.is_file():
        return False
    data = json.loads(manifest.read_bytes())
    if data.get("status") in {"committed", "rolled_back"}:
        shutil.rmtree(stage)
        return False
    if data.get("journal_version") != JOURNAL_VERSION:
        # Old whole-tree snapshots cannot prove which changes were app-owned.
        # Preserve every byte and the journal for deliberate operator recovery.
        raise RuntimeError(
            "legacy publication journal lacks write ownership; manual recovery required"
        )
    with closing(_load_registry_backup(stage, data.get("registry_backup"))) as saved:
        return _recover_validated(registry, config, stage, manifest, data, saved)


def _recover_validated(registry, config, stage, manifest, data, saved):
    records = data["writes"]
    conflicts = []
    targets = {}
    for rel, record in records.items():
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"invalid recovery path: {rel}")
        target = checked_generated_path(config.wiki_dir / rel, config.vault)
        if _relative(config, target, source_output=record.get("source_output", False)) != rel:
            raise RuntimeError(f"unowned recovery path: {rel}")
        observed = _state(target)
        targets[rel] = target
        if observed not in [record["before"], *record["postimages"]]:
            conflicts.append({"path": rel, "reason": "current path differs from owned postimage",
                              "before": record["before"], "after": record["after"],
                              "observed": observed})
        if record["before"] is not None:
            backup = checked_generated_path(stage / "files" / rel, stage / "files")
            if _state(backup) != record["before"]:
                conflicts.append({"path": rel, "reason": "preimage backup mismatch",
                                  "before": record["before"], "after": record["after"],
                                  "observed": _state(backup)})
    if conflicts:
        data.update(status="conflicted", conflicts=conflicts)
        _save(manifest, data)
        raise RuntimeError(
            "publication recovery conflict: " + ", ".join(c["path"] for c in conflicts)
        )
    # Preflight every target before restoring any path. A second check bounds
    # concurrent-editor races, but arbitrary uncoordinated IO is not atomic.
    for rel, record in records.items():
        target = targets[rel]
        current = _state(target)
        if current == record["before"]:
            continue
        if current not in record["postimages"]:
            raise RuntimeError(f"publication recovery conflict during restore: {rel}")
        if record["before"] is None:
            target.unlink()
        else:
            backup = checked_generated_path(stage / "files" / rel, stage / "files")
            _atomic_bytes(target, backup.read_bytes(), record["before"])
    registry.connection.rollback()
    saved.backup(registry.connection)
    registry.set_pipeline_checkpoint("")
    data["status"] = "rolled_back"
    _save(manifest, data)  # cleanup may remove backups before the manifest
    shutil.rmtree(stage)
    return True


class PublicationTransaction:
    def __init__(self, registry, config, plan):
        self.registry, self.config = registry, config
        self.stage = checked_generated_path(config.llmwiki_dir / "publication-stage", config.vault)
        if self.stage.exists():
            if (self.stage / "manifest.json").exists():
                raise RuntimeError("pending identity publication must be recovered first")
            shutil.rmtree(self.stage)  # interrupted staging; apply never started
        self.stage.mkdir(parents=True)
        database = self.stage / "registry-before.db"
        with closing(sqlite3.connect(database)) as backup:
            registry.connection.backup(backup)
            backup.execute("PRAGMA journal_mode=DELETE")
            schema = _registry_schema(backup)
        with database.open("rb") as backup_file:
            os.fsync(backup_file.fileno())
            backup_digest = hashlib.sha256(backup_file.read()).hexdigest()
        self.manifest = {"version": plan.version, "journal_version": JOURNAL_VERSION,
                         "plan_id": plan.plan_id, "status": "pending", "writes": {},
                         "registry_backup": {"sha256": backup_digest, "schema_sha256": schema}}
        _save(self.stage / "manifest.json", self.manifest)
        self.active = True
        self.source_outputs = set()
        self._token = _ACTIVE.set(self)

    def record(self, path, after):
        path = Path(path)
        source_output = str(path) in self.source_outputs
        rel = _relative(self.config, path, source_output=source_output)
        if rel is None:
            return
        before = _state(path)
        record = self.manifest["writes"].get(rel)
        if record is not None:
            if before != record["after"]:
                raise RuntimeError(f"publication recovery conflict before write: {rel}")
            record["after"] = after
            record["postimages"].append(after)
        else:
            if before is not None:
                dest = checked_generated_path(self.stage / "files" / rel, self.stage / "files")
                _atomic_bytes(dest, path.read_bytes(), before)
            self.manifest["writes"][rel] = {
                "before": before, "after": after, "postimages": [after],
                "source_output": source_output,
            }
        # Write-ahead intent allows both crash-before-rename and crash-after-
        # rename recovery. Neither a changed hash nor a snapshot grants ownership.
        _save(self.stage / "manifest.json", self.manifest)

    def _deactivate(self):
        self.active = False
        if _ACTIVE.get() is self:
            _ACTIVE.reset(self._token)

    def commit(self):
        self.manifest["status"] = "committed"
        _save(self.stage / "manifest.json", self.manifest)
        self._deactivate()
        shutil.rmtree(self.stage)

    def rollback(self):
        if self.active:
            self._deactivate()
            recover_publication(self.registry, self.config)
