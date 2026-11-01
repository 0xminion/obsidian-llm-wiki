"""Independent-B regressions: all recovery inputs preflight before mutation."""
import json
import os
import sqlite3
from pathlib import Path

import pytest

from obsidian_llm_wiki.core.publication_transaction import (
    PublicationTransaction,
    recover_publication,
)
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.render.frontmatter import atomic_write
from tests.new.test_publication_recovery_ownership import config_at, plan


@pytest.fixture(autouse=True)
def clear_transaction_context():
    from obsidian_llm_wiki.core.publication_transaction import _ACTIVE

    token = _ACTIVE.set(None)
    yield
    _ACTIVE.reset(token)


@pytest.mark.parametrize('damage', ['symlink', 'missing', 'corrupt', 'rows', 'schema', 'wal'])
def test_sqlite_backup_preflight_precedes_every_output_mutation(tmp_path, damage):
    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    page.write_bytes(b'old\r\n\xff')
    outside = tmp_path / 'external'
    outside.mkdir()
    with VaultRegistry(config.llmwiki_dir) as registry:
        registry.connection.execute('CREATE TABLE preservation_probe(value TEXT)')
        registry.connection.execute("INSERT INTO preservation_probe VALUES ('must survive')")
        registry.connection.execute(
            "INSERT INTO source(filename,content_hash,created_at,updated_at) "
            "VALUES ('a.md','source-hash','created','updated')"
        )
        registry.connection.commit()
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'app replacement')
        tx._deactivate()  # simulate the next process, without attempting rollback
        backup = tx.stage / 'registry-before.db'
        if damage == 'symlink':
            backup.unlink()
            backup.symlink_to(outside / 'must-not-create.db')
        elif damage == 'missing':
            backup.unlink()
        elif damage == 'corrupt':
            backup.write_bytes(b'not sqlite')
        elif damage == 'wal':
            Path(str(backup) + '-wal').symlink_to(outside / 'must-not-create-wal')
        else:
            with sqlite3.connect(backup) as db:
                db.execute('DELETE FROM source' if damage == 'rows' else 'DROP TABLE source')
        prior = page.read_bytes(), page.stat()
        rows = list(registry.connection.iterdump())
        for _ in range(2):
            with pytest.raises((RuntimeError, ValueError), match='backup|Unsafe'):
                recover_publication(registry, config)
            assert (page.read_bytes(), page.stat().st_mtime_ns) == (prior[0], prior[1].st_mtime_ns)
            assert list(registry.connection.iterdump()) == rows
            assert list(outside.iterdir()) == []
            assert (tx.stage / 'manifest.json').is_file()


@pytest.mark.parametrize('human_edit', [False, True])
def test_repeated_intent_keeps_owned_intermediate_without_adopting_human(tmp_path, human_edit):
    from obsidian_llm_wiki.core.publication_transaction import record_replace

    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    page.write_bytes(b'original')
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'legitimate intermediate')
        pending = tmp_path / 'prepared'
        pending.write_bytes(b'final intent')
        record_replace(page, pending)  # durable second intent, no rename
        tx._deactivate()
        if human_edit:
            page.write_bytes(b'human bytes after the second intent')
            with pytest.raises(RuntimeError, match='recovery conflict'):
                recover_publication(registry, config)
            assert page.read_bytes() == b'human bytes after the second intent'
        else:
            assert recover_publication(registry, config)
            assert page.read_bytes() == b'original'
            assert not recover_publication(registry, config)


def test_repeated_intent_rejects_human_preimage_before_record(tmp_path):
    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'first')
        page.write_bytes(b'human changed before second record')
        try:
            with pytest.raises(RuntimeError, match='recovery conflict'):
                atomic_write(page, 'second')
            assert page.read_bytes() == b'human changed before second record'
        finally:
            tx._deactivate()


@pytest.mark.parametrize('remove', [False, True])
def test_generated_source_navigation_and_sources_are_owned_not_human_inputs(tmp_path, remove):
    from obsidian_llm_wiki.core.identity_plan import output_snapshot
    from obsidian_llm_wiki.render.obsidian import (
        _render_directory_indexes,
        _write_generated_page,
    )

    config = config_at(tmp_path)
    config.sources_dir.mkdir()
    human = config.sources_dir / 'index-b00-000.md'
    human.write_bytes(b'human source, not a navigation shard')
    initial = [('a', 'A'), ('b', 'B')] if remove else [('a', 'A')]
    updated = [('a', 'A')] if remove else [('a', 'A'), ('b', 'B')]
    _render_directory_indexes('sources', config.sources_dir, initial, config.wiki_dir, shard_size=1)
    before = {p.name: p.read_bytes() for p in config.sources_dir.iterdir()}
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        _render_directory_indexes(
            'sources', config.sources_dir, updated, config.wiki_dir, shard_size=1,
        )
        generated = config.sources_dir / 'app-source.md'
        assert _write_generated_page(generated, 'generated source', config.wiki_dir)
        assert not _write_generated_page(human, 'do not overwrite', config.wiki_dir)
        outputs = dict(output_snapshot(config))
        assert str(generated) in outputs
        assert str(human) not in outputs
        assert 'sources/app-source.md' in tx.manifest['writes']
        assert 'sources/index-b00-000.md' not in tx.manifest['writes']
        human.write_bytes(b'human after publication intent')
        tx.rollback()
    before[human.name] = b'human after publication intent'
    assert {p.name: p.read_bytes() for p in config.sources_dir.iterdir()} == before


@pytest.mark.parametrize('change', ['xattr', 'acl', 'gid'])
def test_same_byte_human_extended_metadata_edit_blocks_in_place(tmp_path, change):
    import struct

    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    page.write_bytes(b'original\r\n\xff')
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'generated')
        tx._deactivate()
        if change == 'xattr':
            os.setxattr(page, 'user.human', b'\x00\xffhuman')
        elif change == 'acl':
            # Linux POSIX ACL xattr format: owner/named user/group/mask/other.
            mode = page.stat().st_mode & 0o777
            entries = [(1, mode >> 6, 0xffffffff), (2, 4, os.getuid() + 1),
                       (4, 0, 0xffffffff), (16, (mode >> 3) & 7, 0xffffffff),
                       (32, mode & 7, 0xffffffff)]
            acl = struct.pack('<I', 2) + b''.join(struct.pack('<HHI', *x) for x in entries)
            os.setxattr(page, 'system.posix_acl_access', acl)
            assert page.stat().st_mode & 0o777 == mode
        else:
            other_groups = [g for g in os.getgroups() if g != page.stat().st_gid]
            if not other_groups:
                pytest.skip('no alternate permitted group for native chown')
            os.chown(page, -1, other_groups[0])
        def preserved_page_state():
            # Reads may update atime. Preserve every contracted metadata field,
            # using nanoseconds rather than os.stat_result's tuple time values.
            data = page.read_bytes()
            status = page.stat()
            fields = ('st_mode', 'st_ino', 'st_dev', 'st_nlink', 'st_uid', 'st_gid',
                      'st_size', 'st_mtime_ns', 'st_ctime_ns')
            return (data, tuple(getattr(status, field) for field in fields),
                    {k: os.getxattr(page, k) for k in os.listxattr(page)})

        before = preserved_page_state()
        for _ in range(2):
            with pytest.raises(RuntimeError, match='recovery conflict'):
                recover_publication(registry, config)
            assert preserved_page_state() == before


def test_binary_preimage_restores_extended_metadata_exactly(tmp_path):
    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.bin'
    page.write_bytes(b'\x89PNG\x00\xff\r\n')
    page.chmod(0o640)
    os.setxattr(page, 'user.original', b'\xff\x00original')
    os.utime(page, ns=(1600000000000000000, 1600000000000000000))
    before = page.read_bytes(), page.stat()
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'replacement')
        tx.rollback()
    assert page.read_bytes() == before[0]
    assert page.stat().st_mode == before[1].st_mode
    assert page.stat().st_mtime_ns == before[1].st_mtime_ns
    assert (page.stat().st_uid, page.stat().st_gid) == (before[1].st_uid, before[1].st_gid)
    assert os.getxattr(page, 'user.original') == b'\xff\x00original'


@pytest.mark.parametrize('lazy', [False, True])
def test_inner_render_rollback_preserves_human_extended_metadata(tmp_path, lazy):
    from obsidian_llm_wiki.render.obsidian import _RenderTransaction

    page = tmp_path / 'generated.md'
    page.write_bytes(b'original')
    tx = _RenderTransaction(tmp_path, lazy=lazy)
    tx.before_change(page)
    atomic_write(page, 'app')
    tx.record_write(page, 'app')
    os.setxattr(page, 'user.human', b'human-after-write')
    expected = page.stat()
    tx.rollback()
    assert page.read_bytes() == b'app'
    assert os.getxattr(page, 'user.human') == b'human-after-write'
    assert page.stat().st_ino == expected.st_ino


def test_atomic_publication_preserves_existing_extended_metadata(tmp_path):
    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    page.write_bytes(b'original')
    page.chmod(0o640)
    os.setxattr(page, 'user.editor', b'\x00\xffeditor')
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'app')
        tx.commit()
    assert page.stat().st_mode & 0o777 == 0o640
    assert os.getxattr(page, 'user.editor') == b'\x00\xffeditor'


@pytest.mark.parametrize('kind', ['backup', 'lazy', 'full'])
def test_backups_and_inner_snapshots_restore_exact_binary_metadata(tmp_path, kind):
    import struct

    from obsidian_llm_wiki.core.backups import backup_file
    from obsidian_llm_wiki.core.publication_transaction import _state
    from obsidian_llm_wiki.render.obsidian import _RenderTransaction

    root = tmp_path / 'wiki'
    root.mkdir()
    page = root / 'binary.md'
    page.write_bytes(b'\xff\x00\r\n')
    page.chmod(0o640)
    groups = [g for g in os.getgroups() if g != page.stat().st_gid]
    assert groups, 'native alternate-group fixture requires a permitted supplementary group'
    os.chown(page, -1, groups[0])
    os.setxattr(page, 'user.original', b'\xff\x00')
    entries = [(1, 6, 0xffffffff), (2, 4, os.getuid() + 1),
               (4, 0, 0xffffffff), (16, 4, 0xffffffff), (32, 0, 0xffffffff)]
    acl = struct.pack('<I', 2) + b''.join(struct.pack('<HHI', *x) for x in entries)
    os.setxattr(page, 'system.posix_acl_access', acl)
    before = _state(page)
    if kind == 'backup':
        snapshot = backup_file(page, tmp_path / 'backups')
        assert _state(snapshot) == before
    else:
        tx = _RenderTransaction(root, lazy=kind == 'lazy')
        tx.before_change(page)
        atomic_write(page, 'app')
        tx.record_write(page, 'app')
        tx.rollback()
        assert _state(page) == before


@pytest.mark.parametrize('damage', ['integrity', 'schema'])
def test_bound_backup_still_requires_valid_sqlite_and_schema(tmp_path, damage):
    import hashlib

    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    page.write_bytes(b'old')
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'app')
        tx._deactivate()
        backup = tx.stage / 'registry-before.db'
        if damage == 'integrity':
            backup.write_bytes(b'invalid SQLite')
        else:
            with sqlite3.connect(backup) as saved:
                saved.execute('DROP TABLE source')
        manifest_path = tx.stage / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['registry_backup']['sha256'] = hashlib.sha256(backup.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        live = list(registry.connection.iterdump())
        with pytest.raises(RuntimeError, match='backup preflight'):
            recover_publication(registry, config)
        assert page.read_bytes() == b'app'
        assert list(registry.connection.iterdump()) == live


def test_rollback_cleanup_can_remove_database_before_terminal_manifest(tmp_path, monkeypatch):
    import shutil

    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    page.write_bytes(b'before')
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'app')
        real_rmtree = shutil.rmtree

        def interrupted_cleanup(stage):
            (stage / 'registry-before.db').unlink()
            raise OSError('cleanup interruption after database unlink')

        monkeypatch.setattr(shutil, 'rmtree', interrupted_cleanup)
        with pytest.raises(OSError, match='cleanup interruption'):
            tx.rollback()
        assert json.loads((tx.stage / 'manifest.json').read_text())['status'] == 'rolled_back'
        live = list(registry.connection.iterdump())
        monkeypatch.setattr(shutil, 'rmtree', real_rmtree)
        assert not recover_publication(registry, config)
        assert page.read_bytes() == b'before'
        assert list(registry.connection.iterdump()) == live


def test_unreadable_metadata_blocks_before_publication(tmp_path, monkeypatch):
    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    page.write_bytes(b'before')
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        real_listxattr = os.listxattr

        def denied(path, *args, **kwargs):
            if path == page:
                raise PermissionError('metadata inaccessible')
            return real_listxattr(path, *args, **kwargs)

        monkeypatch.setattr(os, 'listxattr', denied)
        with pytest.raises(RuntimeError, match='cannot inspect publication metadata'):
            atomic_write(page, 'app')
        assert page.read_bytes() == b'before'
        tx._deactivate()


def test_observed_uid_change_blocks_recovery_without_chown_privilege(tmp_path, monkeypatch):
    # Native uid mutation requires privilege; simulate only the stat boundary.
    config = config_at(tmp_path)
    page = config.wiki_dir / 'generated.md'
    page.write_bytes(b'before')
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, 'app')
        tx._deactivate()
        original = Path.stat

        def observed(path, *args, **kwargs):
            actual = original(path, *args, **kwargs)
            if path == page:
                from types import SimpleNamespace

                return SimpleNamespace(**{
                    name: (actual.st_uid + 1 if name == 'st_uid' else getattr(actual, name))
                    for name in dir(actual) if name.startswith('st_')
                })
            return actual

        monkeypatch.setattr(Path, 'stat', observed)
        with pytest.raises(RuntimeError, match='recovery conflict'):
            recover_publication(registry, config)
        assert page.read_bytes() == b'app'
