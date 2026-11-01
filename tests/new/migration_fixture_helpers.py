"""Disposable immutable-preview fixtures; never open published SQL by pathname.

Resealing here is deliberate adversarial fixture construction, not a publisher
or a repair function. It changes checksums, never upgrades an invalid proof.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

from obsidian_llm_wiki.core.identity_plan import canonical, digest, registry_generation


def meta_for(cli):
    """Use the generation pinned before tampering, not a revalidated lookup."""
    return cli.generation / ".llmwiki"


@contextmanager
def closed_registry(meta, *, edit=False):
    """Inspect/alter only a private memory copy; always close the connection.

    Editing explicitly serializes fixture bytes back after commit. No writable
    VaultRegistry, schema migrations, WAL or filesystem SQLite connection.
    """
    db = meta / "registry.db"
    for suffix in ("-wal", "-shm", "-journal"):
        assert not db.with_name(db.name + suffix).exists(), "fixture registry is not closed"
    raw = bytearray(db.read_bytes())
    assert raw.startswith(b"SQLite format 3\x00")
    raw[18] = raw[19] = 1
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(bytes(raw))
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        if not edit:
            connection.execute("PRAGMA query_only=ON")
        yield SimpleNamespace(connection=connection)
        if edit:
            connection.commit()
            db.write_bytes(connection.serialize())
    finally:
        connection.close()


def reseal_outer(cli):
    """Refresh only the outer file inventory, leaving inner semantics untouched."""
    pointer = cli.wiki / "OLW-Publications" / "current.json"
    manifest = json.loads(pointer.read_text())
    assert manifest["generation"] == cli.generation.name
    manifest["files"] = {
        p.relative_to(cli.generation).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(cli.generation.rglob("*"))
        if p.is_file()
    }
    pointer.write_text(canonical(manifest))
    # Independent readback: this test must not pass on stale outer hashes.
    actual = json.loads(pointer.read_text())
    assert actual["files"] == {
        p.relative_to(cli.generation).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in cli.generation.rglob("*")
        if p.is_file()
    }


def reseal_inner_generation(cli):
    meta = meta_for(cli)
    marker_path = meta / "identity-publication.json"
    marker = json.loads(marker_path.read_text())
    with closed_registry(meta) as registry:
        marker["generation"] = registry_generation(registry)
    marker_path.write_text(canonical(marker))


def edit_receipt_and_reseal(committed, edit):
    """Keep plan/cache/SQL hashes consistent while preserving invalid evidence."""
    edit_payload_and_reseal(committed, lambda p: edit(p["review_receipt"]))


def edit_payload_and_reseal(committed, edit):
    """Raw payload variant also supports deliberately invalid collection shapes."""
    cli, _endpoint, _marker, raw = committed
    meta = meta_for(cli)
    payloads = {name: json.loads(payload) for name, payload in raw["sources"]}
    edit(payloads["reuse.md"])
    plan = dict(raw, sources=[(name, canonical(p)) for name, p in sorted(payloads.items())])
    plan_id = digest(plan)
    with closed_registry(meta, edit=True) as registry:
        registry.connection.execute(
            "UPDATE synthesis SET payload_json=? WHERE source_id=(SELECT id FROM source "
            "WHERE filename='reuse.md')",
            (json.dumps(payloads["reuse.md"]),),
        )
    cache = meta / "cache" / "reuse.md.json"
    data = json.loads(cache.read_text())
    data["payload"] = payloads["reuse.md"]
    cache.write_text(canonical(data))
    (meta / "publications" / (plan_id + ".json")).write_text(canonical(plan))
    marker_path = meta / "identity-publication.json"
    marker = json.loads(marker_path.read_text())
    marker.update(
        plan_id=plan_id, source_payloads={name: digest(p) for name, p in payloads.items()}
    )
    marker_path.write_text(canonical(marker))
    reseal_inner_generation(cli)
    pointer = cli.wiki / "OLW-Publications" / "current.json"
    manifest = json.loads(pointer.read_text())
    manifest["identity_plan_id"] = plan_id
    pointer.write_text(canonical(manifest))
    reseal_outer(cli)
    assert digest(json.loads((meta / "publications" / (plan_id + ".json")).read_text())) == plan_id


def pending_workspace(cli, raw, state):
    """Real private transaction in a fresh unpublished copy; keep current intact."""
    import shutil

    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.identity_plan import ProspectivePublicationPlan
    from obsidian_llm_wiki.core.publication_transaction import PublicationTransaction
    from obsidian_llm_wiki.core.registry import VaultRegistry

    vault = cli.wiki / "OLW-Publications" / ".staging" / ("preview-" + state)
    config = Config(vault_path=str(vault))
    shutil.copytree(cli.generation, config.wiki_dir)
    if state == "binary-stuck":
        binary = config.wiki_dir / "attachments" / "diagram.png"
        binary.parent.mkdir()
        binary.write_bytes(b"\x89PNG\xff\x00binary data")
    plan = ProspectivePublicationPlan(**raw)
    with VaultRegistry(config.llmwiki_dir) as registry:
        transaction = PublicationTransaction(registry, config, plan)
        try:
            if state == "pending-marker":
                plan.write_marker(registry, config, "pending")
            elif state == "interrupted-staging":
                (transaction.stage / "manifest.json").unlink()
            elif state == "corrupt-journal":
                (transaction.stage / "manifest.json").write_bytes(b"\xffbroken")
            elif state == "committed-cleanup":
                (transaction.stage / "manifest.json").write_text(
                    json.dumps(dict(transaction.manifest, status="committed"))
                )
        finally:
            # Simulated dead process: detach only, never recover its disk state.
            transaction._deactivate()
    return config
