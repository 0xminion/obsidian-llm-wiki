# Migration, dry runs, backups, and recovery

Active operating procedure. There is **no `olw migrate` command**. See [identity-migration-preview.md](identity-migration-preview.md) for legacy identity inventory and schema-7 publication inspection: `olw identity-migration "$VAULT" --dry-run` is read-only, refuses busy/uncheckpointed-WAL registries, and has **no apply or recovery mode**. [`VaultRegistry.__init__`](../obsidian_llm_wiki/core/registry.py) upgrades supported SQLite schemas on writable open; `migrate_legacy()` imports old JSON state/cache. Public builds perform this work in a private workspace, not by migrating human legacy metadata or installed generations in place. See [immutable application callers](immutable-application-callers.md) and [version namespaces](data-model.md#version-namespaces).

## Offline command rehearsal receipt

The documented snapshot, read-only plan, explicit migration, canonical-ID discovery/policy validation, and rollback snippets were executed on a new synthetic English/Chinese vault with empty credentials and live sockets blocked. The 7-file snapshot restored exactly; plan left paths/bytes/mtimes unchanged; schema 6 imported 2 sources/2 syntheses/2 concepts, integrity was `ok`, foreign-key violations 0, and repeated import returned false. The failed synthetic state was preserved beside the restored vault. See [receipts and limits](verification.md#documentation-rehearsal). Provider-calling build stages, real user data, and every historical DB schema were **not** exercised by this rehearsal.

## Before touching the production vault

1. Stop all `olw` processes, scheduled jobs, the desktop bridge, editors, and sync writers for this vault. A compiler lock is not a lock against an editor or cloud sync client.
2. Keep the old application environment/package version available for rollback. Do not run an old binary against the upgraded database; future-schema protection rejects that.
3. Snapshot the **entire vault**, including sources, notes, `.env`, policy, `.llmwiki`, backups, `registry.db`, and any `registry.db-wal`/`registry.db-shm`. Copying only the DB during activity can lose WAL state. Per-page automatic backups are not a migration snapshot.
4. Rehearse on a disposable copy. `olw build --dry-run` is **read-only**: it validates current publication, then reads source bodies and current (or legacy fallback) `state.json` without writable SQLite opens or provider calls. Its content plan does **not** predict schema migration, synthesis compatibility, retry eligibility, or render changes. Missing JSON state classifies present sources as new; invalid state/inventory or corrupt publication fails closed. `health` is read-only; direct writable registry constructors can still migrate, so use only a disposable legacy copy for the internal snippet below.

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

### 2. Rehearse legacy schema import without LLM calls

First exercise the actual read-only CLI inventory on the copy (this does not migrate SQLite):

```bash
olw build "$REHEARSAL" --dry-run
```

For a **legacy flat-layout rehearsal only**, the following explicitly exercises the internal migration API and database checks on that disposable copy. It mutates the copy's legacy metadata and is **not a new CLI command or the current immutable publication path**. Do not point it at an installed generation. For an already-published vault, use `olw current "$REHEARSAL" --json` and `olw identity-migration "$REHEARSAL" --dry-run --json` instead; leave installed and legacy metadata intact.

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

The candidate target schema is **7**, adding exact historical identity source contexts and entity history. The schema-6 rehearsal receipt above is historical, not verification of this new contract. Legacy import is marker-based and upsert-oriented; a second call should not re-import a completed migration. The legacy operation ledger is separately imported on registry open. Invalid/skipped legacy JSON needs manual inspection: a migration marker is not proof that every old cache entry was valid. Missing historical context is not fabricated by preview.

Do not pass current artifact labels to certify old output. Legacy material remains version-incompatible until it is successfully re-synthesized. Check expected source filenames and important notes, not only row counts or `integrity_check`.

### 3. Rehearse a real build and inspect output

These steps may contact the configured LLM/embedding service and incur cost. Review the copied `.env` and active environment first; keep the original frozen or take a new snapshot before later applying changes to it.

```bash
olw build "$REHEARSAL"
olw current "$REHEARSAL" --json
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

Existing legacy notes stay in place regardless of render ownership hashes. New output lives under `04-Wiki/OLW-Publications/generations/<generation>/`; use the `entrypoint` returned by `current`, not the old flat `index.md`. Keep human edits in retained legacy notes or separate annotations; do not move or overwrite them merely to make room for regenerated pages. There is no blanket “trust all old generated files” migration switch.

### 4. Apply to the original

After the copy has passed your checks, confirm production has not changed since the snapshot. If it has, stop writers and repeat the snapshot/rehearsal. Upgrade using the same tested environment:

```bash
olw build "$VAULT"
olw current "$VAULT" --json
olw health "$VAULT" --full --json
olw validate "$VAULT" --strict
```

Do not copy just a rehearsal `registry.db` over production: caches, page hashes, source signatures, source ownership, and rendered files must agree. Keep the complete pre-upgrade snapshot until your acceptance checks are complete.

## Choose the right recovery operation

### Legacy identity publication journal gate

For legacy metadata before the first immutable publication, inspect the identity journal read-only:

```bash
olw identity-migration "$REHEARSAL" --dry-run --json
```

In that legacy layout, any `.llmwiki/publication-stage` is an explicit blocker, **even with a committed
`identity-publication.json` marker**. A pending, interrupted, malformed, binary-
blocked or cleanup-incomplete journal is not a clean generation. The preview
leaves every byte untouched, makes no provider calls and never attempts recovery.
Exit 3 inventories the blocker; exit 2 refuses an unsafe/busy database snapshot.
Neither is success/readiness. Preserve the whole vault and journal for diagnosis.
Do not delete a journal or sidecars to silence the gate. Rehearse recovery only
with a matching writer whose recovery behavior has been separately verified;
the audited candidate had human-file recovery and marker/journal ordering defects.
Normal builds and maintenance APIs may write and contact providers; they are not
read-only substitutes. See the preview guide for filesystem race, atime and
snapshot limits. This workflow does not certify production migration apply.

For immutable publications, `OLW-Publications/current.json` is the authoritative pointer. Resolve it with `olw current "$VAULT" --json`. A pending new private workspace does not invalidate a valid previous generation. Corrupt current generations are refused by build and readers; do not edit manifests, delete journals, or overwrite generation files to bypass validation. The earlier legacy journal findings above describe their historical audit, not a replacement for the current [publication contract](immutable-publication.md).

| Problem | Operation | Caveat |
| --- | --- | --- |
| Need a fresh projection from a valid current generation or legacy inputs | `olw build "$VAULT" --refresh-render` | Publishes a complete new immutable generation through normal review; compatible syntheses are reused, but changed/incompatible inputs may call models and incur cost. Existing human files and generations are retained. |
| Damaged/corrupt committed generation | Stop writers, preserve the whole state, and inspect a verified whole-vault snapshot | `--refresh-render` does not bypass current-generation validation or repair files in place. Do not mix individual DB, cache, page, or pointer backups. |
| Need fresh inference for every source | `olw build "$VAULT" --force` | Expensive; re-reads/re-synthesizes everything and explicitly resets pending synthesis retry backoff. Does not steal live claims or bypass edit protection. |
| One failed source needs another synthesis attempt | `olw recompile "$VAULT" article.md` | Forces the named complete input through synthesis/review and publishes a complete new generation; no follow-up build is needed. Cannot recover text missing from extraction. |
| Source text was truncated or changed upstream | `olw ingest "$VAULT" --url URL --refresh --preview` | Observational extraction may contact services. Review and save accepted text under a new input filename, preserving existing sources, then build. Mutating `--refresh` is refused. A larger prompt cannot restore lost source text. |
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

## Single-backup inspection and republishing

`olw fix --restore` is a **retained compatibility option that is always refused**, including for a trusted backup. It is not a working single-file restoration command. The refusal protects existing human files and immutable generations; supplying a different backup path does not enable restoration.

Inspect a backup read-only. If its content is useful, copy it manually to a **new, separate annotation file** under `03-Raw-Annotations/`, preserving the backup and all existing files. Do not overwrite a legacy note, an existing annotation, or any file inside `OLW-Publications/generations/`.

To regenerate output from source inputs when the current publication is valid, use `olw build "$VAULT" --refresh-render`. `olw fix "$VAULT" --apply` also republishes a complete projection (and refuses when there are no source inputs); it does not apply each dry-run text edit. Both use normal synthesis/review gates and **may call models and incur cost** for changed/incompatible inputs. They retain human files and old immutable generations and do not automatically resolve human-review findings. This is republishing, not restoring backup bytes. Whole-vault snapshot rollback above is a separate, explicit recovery operation.
