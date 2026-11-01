"""Application boundaries: pinned read projections and separate mutable state.

Publication metadata is never an operational database. These helpers do not
construct a registry on a reader path or recover/migrate a legacy database.
"""

from obsidian_llm_wiki.core.generated_paths import (
    checked_generated_path,
    checked_metadata_directory,
)
from obsidian_llm_wiki.core.immutable_publication import (
    _owned_root,
    current_publication,
    publication_root,
)
from obsidian_llm_wiki.core.publication_transaction import require_publication_ready


def read_publication(config):
    """Return one validated bundle and manifest (None for legacy inspection)."""
    snapshot = current_publication(config.wiki_dir)
    if snapshot:
        return snapshot
    require_publication_ready(config.llmwiki_dir)
    return config.wiki_dir, None


def operational_directory(config, *, create=False, kind="operations"):
    """Mutable ingestion/session/report state, never inside a generation.

    `create=False` is strictly observational, including on an unbuilt vault.
    A corrupt committed publication refuses before operational writes.
    """
    if kind not in {"operations", "query-state"}:
        raise ValueError("Unknown operational state kind")
    current_publication(config.wiki_dir, allow_uncommitted=True)
    root = _owned_root(config.wiki_dir) if create else publication_root(config.wiki_dir)
    directory = checked_generated_path(root / kind, config.vault)
    checked_metadata_directory(directory, config.vault)
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    return directory


def source_inventory(config):
    """Read human source bodies; no mutable SQLite constructor or flat state."""
    from obsidian_llm_wiki.ingest.sources import load_sources_incremental

    class Inventory:
        def source_filenames(self):
            return set()

        def source_file_states(self):
            return {}

    return load_sources_incremental(config.sources_dir, Inventory(), force=True)


def registry_inspection(config, bundle):
    """Read source hashes/render ownership without migrations or WAL side effects.

    Legacy WAL/lock guards are identical to migration preview. Published bytes
    were already validated by the caller's pinned read_publication resolution.
    """
    import sqlite3

    from obsidian_llm_wiki.core.identity_migration import (
        PreviewRefusedError,
        _Layout,
        _load_registry_copy,
        _registry_state,
    )

    metadata = checked_metadata_directory(bundle / ".llmwiki", config.vault)
    layout = _Layout(config.vault, bundle, config.sources_dir, metadata, metadata / "registry.db")
    facts, refusal = _registry_state(layout)
    if refusal:
        raise RuntimeError("Registry inspection refused: " + "; ".join(refusal))
    if facts["strategy"] == "absent":
        return {}, None
    try:
        connection, _ = _load_registry_copy(layout.db, facts)
    except PreviewRefusedError as exc:
        raise RuntimeError("Registry inspection refused: " + str(exc)) from exc
    try:
        hashes = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT filename,content_hash FROM source WHERE status='active'"
            )
        }
        rendered = {str(row[0]) for row in connection.execute("SELECT path FROM render_state")}
        return hashes, rendered
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            "Unsupported registry inspection schema; run identity-migration --dry-run"
        ) from exc
    finally:
        connection.close()


def install_source(config, source, content, *, preferred_stem=None):
    """Install complete bytes at a fresh source filename without reservations.

    The old empty-file reservation followed by replacement raced editors. A
    no-replace install preserves any editor-created candidate, including metadata.
    """
    from uuid import uuid4

    from obsidian_llm_wiki.core.immutable_publication import _exclusive, _install_exclusive
    from obsidian_llm_wiki.ingest.sources import _SOURCE_INDEX_SHARD, _SOURCE_METADATA_FILES
    from obsidian_llm_wiki.render.obsidian import slugify

    directory = checked_generated_path(config.sources_dir, config.vault)
    operations = operational_directory(config, create=True)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = operations / ("source-" + uuid4().hex)
    _exclusive(prepared, content.encode("utf-8"))
    stem = slugify(preferred_stem or source.title)
    try:
        for index in range(101):
            name = f"{stem}{'-' + str(index) if index else ''}.md"
            if name in _SOURCE_METADATA_FILES or _SOURCE_INDEX_SHARD.fullmatch(name):
                continue
            target = checked_generated_path(directory / name, config.vault)
            try:
                _install_exclusive(prepared, target)
                return target
            except FileExistsError:
                continue
        raise RuntimeError("Too many source filename collisions; choose a different source title")
    finally:
        if prepared.exists():
            prepared.unlink()
