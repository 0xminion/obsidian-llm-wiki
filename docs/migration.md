# Migration, dry runs, backups, and recovery

Active operating procedure. There is **no `olw migrate` command**. [`VaultRegistry.__init__`](../obsidian_llm_wiki/core/registry.py) upgrades supported SQLite schemas on open; `migrate_legacy()` imports old JSON state/cache. A normal build invokes both paths. See [version namespaces](data-model.md#version-namespaces).

## Offline command rehearsal receipt

The documented snapshot, read-only plan, explicit migration, canonical-ID discovery/policy validation, and rollback snippets were executed on a new synthetic English/Chinese vault with empty credentials and live sockets blocked. The 7-file snapshot restored exactly; plan left paths/bytes/mtimes unchanged; schema 6 imported 2 sources/2 syntheses/2 concepts, integrity was `ok`, foreign-key violations 0, and repeated import returned false. The failed synthetic state was preserved beside the restored vault. See [receipts and limits](verification.md#documentation-rehearsal). Provider-calling build stages, real user data, and every historical DB schema were **not** exercised by this rehearsal.

## Before touching the production vault

1. Stop all `olw` processes, scheduled jobs, the desktop bridge, editors, and sync writers for this vault. A compiler lock is not a lock against an editor or cloud sync client.
2. Keep the old application environment/package version available for rollback. Do not run an old binary against the upgraded database; future-schema protection rejects that.
3. Snapshot the **entire vault**, including sources, notes, `.env`, policy, `.llmwiki`, backups, `registry.db`, and any `registry.db-wal`/`registry.db-shm`. Copying only the DB during activity can lose WAL state. Per-page automatic backups are not a migration snapshot.
4. Rehearse on a disposable copy. `olw build --dry-run` is now **read-only**: it reads source bodies and `state.json` without opening SQLite or contacting providers. It does **not** preview schema migration, version compatibility, cache validity, retry eligibility, or render changes. Missing JSON state classifies present sources as new; invalid state/inventory fails closed. `health` and normal registry opens can write maintenance/schema state, so use the snapshot first.

The following Bash/Python commands use the installed `olw` and the same activated Python environment. Replace only the vault path. They refuse symlinks rather than accidentally running a rehearsal through links back into production; handle a symlinked vault with an explicit filesystem snapshot strategy instead.

### 1. Create and verify an offline snapshot

Run only after writers have stopped; these commands do not stop them for you.

```bash
export VAULT="$HOME/MyVault"
export WORK="$(mktemp -d "${TMPDIR:-$HOME}/olw-upgrade.XXXXXXXX")"
export BACKUP="$WORK/backup"
export REHEARSAL="$WORK/rehearsal"
python - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path

vault = Path(os.environ['VAULT']).expanduser().absolute()
work = Path(os.environ['WORK']).resolve()
assert vault.is_dir() and not vault.is_symlink(), 'Use a real vault directory'
assert work != vault and vault not in work.parents, 'Snapshot must be outside vault'
for directory, dirs, files in os.walk(vault, followlinks=False):
    for name in dirs + files:
        assert not (Path(directory) / name).is_symlink(), 'Symlink: snapshot separately'

def manifest(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    stream, 'sha256'
                ).hexdigest()
    return result

before = manifest(vault)
backup = Path(os.environ['BACKUP'])
rehearsal = Path(os.environ['REHEARSAL'])
shutil.copytree(vault, backup)
assert before == manifest(backup) == manifest(vault), 'Snapshot mismatch; stop here'
shutil.copytree(backup, rehearsal)
assert before == manifest(rehearsal), 'Rehearsal copy mismatch'
(work / 'backup-manifest.json').write_text(json.dumps(before, indent=2) + '\n')
print(f'Verified {len(before)} files; snapshot={backup}; rehearsal={rehearsal}')
PY
```

Store `WORK` somewhere durable if `TMPDIR` is periodically pruned; the example location is suitable for rehearsal, not long-term backup retention. The snapshot contains credentials from `.env`; restrict access and do not upload it as audit evidence.

### 2. Rehearse schema and legacy import without LLM calls

First exercise the actual read-only CLI inventory on the copy (this does not migrate SQLite):

```bash
olw build "$REHEARSAL" --dry-run
```

Then explicitly exercise the internal migration API and database checks on that same copy. This is an executable Python maintenance snippet, **not a new CLI command**:

```bash
TARGET="$REHEARSAL" python - <<'PY'
import os
from pathlib import Path
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry

root = Path(os.environ['TARGET']) / '04-Wiki' / '.llmwiki'
with VaultRegistry(root) as registry:
    imported = registry.migrate_legacy(ArtifactVersions('legacy', 'legacy', 'legacy', 'legacy'))
    connection = registry.connection
    version = connection.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()[0]
    integrity = [row[0] for row in connection.execute('PRAGMA integrity_check')]
    foreign_keys = connection.execute('PRAGMA foreign_key_check').fetchall()
    assert integrity == ['ok'], integrity
    assert not foreign_keys, [tuple(row) for row in foreign_keys]
    counts = {table: connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
              for table in ('source', 'synthesis', 'concept', 'moc')}
    print({'schema': version, 'legacy_import_ran': imported, 'counts': counts,
           'integrity': integrity, 'foreign_key_violations': len(foreign_keys)})
PY
```

The current target schema is 6. Legacy import is marker-based and upsert-oriented; a second call should not re-import a completed migration. The legacy operation ledger is separately imported on registry open. Invalid/skipped legacy JSON needs manual inspection: a migration marker is not proof that every old cache entry was valid.

Do not pass current artifact labels to certify old output. Legacy material remains version-incompatible until it is successfully re-synthesized. Check expected source filenames and important notes, not only row counts or `integrity_check`.

### 3. Rehearse a real build and inspect output

These steps may contact the configured LLM/embedding service and incur cost. Review the copied `.env` and active environment first; keep the original frozen or take a new snapshot before later applying changes to it.

```bash
olw build "$REHEARSAL"
olw health "$REHEARSAL" --full --json
olw validate "$REHEARSAL" --strict
olw metrics "$REHEARSAL"
olw fix "$REHEARSAL" --dry-run --json
```

Inspect errors and per-source outcomes as well as process exit status. Build/recompile/ingest failures now exit 1, cancellation exits 130, and success/no-op exits 0. Ingest `--json` terminal events distinguish `failed` and `partial_failure`; partial success does not roll back successfully acquired sources. A full health scan can legitimately report findings requiring review. Verify:

- Original source bodies/provenance and human-edited body/frontmatter survived.
- Concept identities, ownership, language, evidence, and wikilinks make sense.
- Old artifact versions were not silently promoted by metadata-only normalization.
- Failed replacement syntheses retained last-good output and are still retryable.
- Re-running an unchanged build performs no new synthesis; a changed source updates its dependents.

Old notes with no `render-page-hashes.json` entry are intentionally protected. A build may preserve them instead of adopting/replacing them. Review and back up a legacy note before moving it out of a generated path; then compare regenerated output and merge your edits explicitly. There is no blanket “trust all old generated files” migration switch.

### 4. Apply to the original

After the copy has passed your checks, confirm production has not changed since the snapshot. If it has, stop writers and repeat the snapshot/rehearsal. Upgrade using the same tested environment:

```bash
olw build "$VAULT"
olw health "$VAULT" --full --json
olw validate "$VAULT" --strict
```

Do not copy just a rehearsal `registry.db` over production: caches, page hashes, source signatures, source ownership, and rendered files must agree. Keep the complete pre-upgrade snapshot until your acceptance checks are complete.

## Choose the right recovery operation

| Problem | Operation | Caveat |
| --- | --- | --- |
| Missing/damaged generated artifact with otherwise current synthesis | `olw build "$VAULT" --refresh-render` | Reconciles output without forcing synthesis. Protected edited/unknown files remain protected. |
| Need fresh inference for every source | `olw build "$VAULT" --force` | Expensive; re-reads/re-synthesizes everything and explicitly resets pending synthesis retry backoff. Does not steal live claims or bypass edit protection. |
| One failed source needs another synthesis attempt | `olw recompile "$VAULT" article.md`, then `olw build "$VAULT"` | Uses complete input; cannot recover text missing from extraction. |
| Source text was truncated or changed upstream | `olw ingest "$VAULT" --url URL --refresh` | Re-extract after checking limits/access; title/provenance-only refresh is retained too. A larger prompt cannot restore lost source text. |
| Interrupted ingestion | `olw ingest "$VAULT" --resume` | Check results and pending retry times; not an exactly-once guarantee. Repeating a known URL with synthesis enabled also builds the existing corpus; it need not extract a new URL to recover due synthesis work. |
| Whole migration or cross-store inconsistency | Restore the complete snapshot below | Do not mix old DB with new generated state. |

`--refresh-render` is **not an unconditional no-network mode**: changed inputs or incompatible synthesis versions still compile. If you need a strictly local check, use the rehearsal inventory/migration snippets instead of a build.

## Whole-vault rollback

Stop writers again. Restore using the **old compatible application version** after the directory is restored. The following preserves the failed vault beside the original and copies the verified snapshot back; it does not delete the failed state. Ensure sufficient disk space. A directory rename requires ordinary filesystem semantics; use your snapshot tooling for mount points.

```bash
python - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

vault = Path(os.environ['VAULT']).expanduser().absolute()
backup = Path(os.environ['BACKUP']).resolve()
expected = json.loads((Path(os.environ['WORK']) / 'backup-manifest.json').read_text())

def manifest(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    stream, 'sha256'
                ).hexdigest()
    return result

assert manifest(backup) == expected, 'Backup changed; do not restore blindly'
assert vault.is_dir() and not vault.is_symlink(), 'Unexpected vault path'
failed = vault.with_name(vault.name + '.before-recovery-' + uuid4().hex)
vault.rename(failed)
shutil.copytree(backup, vault)
assert manifest(vault) == expected, 'Restore incomplete; keep writers stopped'
print(f'Restored snapshot. Failed state preserved at {failed}')
PY
```

If the copy fails, leave writers stopped; the failed vault is preserved and the backup remains intact. Do not overlay arbitrary DB/cache files until you have chosen a consistent snapshot.

For a **single maintenance backup**, the supported command is:

```bash
olw fix "$VAULT" --restore "$VAULT/04-Wiki/.llmwiki/backups/PATH/FILE.bak"
```

Replace `PATH/FILE.bak` with an actual trusted backup path. This restores one supported backup, not the whole database or a migration. Review the target and make a full snapshot first; `--restore` cannot be combined with `--dry-run` or `--apply`.
