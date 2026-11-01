# Publication recovery preservation contract

## Status

The v3 owned-write journal fixes independent findings B2–B5 under the **quiescent/cooperating-writer contract below**. It does **not** satisfy unconditional preservation against an uncoordinated editor racing publication (L1). Do not describe this change as race-safe or approve the stronger no-human-edit-loss requirement on this evidence.

## Validated inputs and durable decisions

- Before rollback changes any output or live registry data, require an existing contained, regular, non-symlink `registry-before.db`. Reject SQLite WAL/SHM/journal sidecars, including symlinks. Open the backup through a read-only descriptor and deserialize its pinned bytes into a private SQLite connection; never ask SQLite to open an unvalidated backup pathname.
- Verify its journal-bound SHA-256, integrity, schema fingerprint and required tables, and foreign-key consistency. Recovery uses this validated private copy, not a later pathname reopen. There is no empty-database fallback. Missing, corrupt, unbound or unsafe backups preserve the current generation and require operator intervention. These hashes detect corruption/substitution; they are not authentication against a writer able to forge the entire journal.
- Preflight the output paths and preimage backups before restoring any of them. A path matching neither its preimage nor a journaled legitimate postimage is a conflict. The manifest records the exact conflict path and observed state; repeated recovery preserves it in place.
- Pending/conflicted journals still block cooperating readers. A durably committed journal is cleanup-only. Successful rollback durably records `rolled_back` before deleting its backup tree, so cleanup interruption cannot cause an empty/missing database to be restored on restart.
- Journal version v2 lacks the database binding, extended metadata and transition representation. It is deliberately **not** silently upgraded or automatically restored. Retain the journal, preimages and current files; inspect and reconcile them manually after taking an independent snapshot.

## Ownership and metadata

Only recorded application writes grant rollback ownership. A tree snapshot or a changed hash does not. Source navigation shards, collision-allocated indexes and generated source pages are explicitly admitted after the renderer's ownership check. Arbitrary human source Markdown—including filenames resembling shards—is not admitted. The durable generated-page inventory participates in output-generation agreement without scanning human inputs into the rollback set.

The recoverable file state is exact bytes (including binary content and CRLF), size, POSIX mode, nanosecond mtime, uid/gid and all visible binary extended attributes. On Linux this includes `system.posix_acl_access`. Human changes to these fields after an owned write are preserved in place and block recovery. Unsupported/unreadable metadata or inability to restore ownership/attributes is an error, not permission to discard it. Atomic text publication carries existing editor metadata forward; backups and both renderer snapshot modes preserve it too.

Atomic restoration creates/replaces an inode. **Inode number, ctime, birthtime, hard-link topology and atime are not restored or claimed to be restored.** Do not use a byte/metadata restoration receipt to claim these identities survived. The test host exercises native Linux xattrs, ACLs and supplementary-group changes; uid-change detection has a stat-boundary simulation because this unprivileged test cannot transfer ownership to another user. No native Windows/macOS metadata guarantee is established.

## L1 — remaining technical blocker

A separate actor writing after the durable intent and before `os.replace` is still overwritten. The real CLI replay pauses at exactly that boundary, records the actor's bytes/mode/inode, resumes publication, then invokes real recovery. Recovery restores the earlier pre-transaction file, not the actor's intervening edit. This remains demonstrated data loss against the stronger requirement, not merely a theoretical race.

Python's path-based `os.replace`/`unlink` provide no compare-and-swap condition on the destination inode/content/metadata. Rechecking immediately before those calls leaves the same final window. Backups taken before that window do not retain an inode newly installed during it. The implementation here adds no claimed closure for L1.

The precise supported contract therefore requires editors and other writers to remain quiescent while publication or recovery mutates generated paths (or to cooperate with the compiler's exclusive lock). Human edits already present when ownership checks run, edits made after a crashed writer, unmanaged creations/deletions and policy changes are preserved as tested; arbitrary mutations during the last check/use interval are outside this contract and remain a release blocker if unconditional preservation is required.

A future stronger primitive must retain the actual displaced inode durably (not just a preflight copy), make displacement/recovery conflicts observable, and avoid destructive restoration if the destination changes again. Platform-specific exchange/retained-displacement protocols also need crash tests at every syscall and directory durability boundary. Such a protocol was **not** delivered in this patch. No extra preflight is being represented as its substitute.

## Integration boundary

The idempotency worker's `write_marker` implementation is retained verbatim. A committed marker may legitimately retain the previous plan ID when every post-apply field except plan ID is identical and the old plan archive digest verifies. The transaction's plan ID need not equal that marker's plan ID. Reader acceptance continues to rely on the durable journal decision, not an invented equality rule. Public-query B1 changes and registry/entity-identity changes are separately owned and must be verified on the parent's integrated candidate.
