"""Fixture arrangements at the legacy-to-immutable publication boundary.

Never mutate committed artifacts to simulate an ordinary retry. Missing legacy
caches/vectors are prepared in a real never-published workspace; public migration
must repair its private copy, retaining the legacy bytes as historical evidence.
"""

from obsidian_llm_wiki.core.pipeline import _run_pipeline_workspace


async def compile_legacy(config, sources, **kwargs):
    """Build a genuine flat-era fixture, without any immutable namespace."""
    namespace = config.wiki_dir / "OLW-Publications"
    assert not namespace.exists()
    result = await _run_pipeline_workspace(config.vault, sources, config, **kwargs)
    assert not namespace.exists()
    return result


def assert_private_policy_load(config, loads):
    """One policy load from the exact private generation workspace."""
    import json

    manifest = json.loads((config.wiki_dir / "OLW-Publications/current.json").read_text())
    workspace = config.wiki_dir / "OLW-Publications/.staging" / manifest["generation"]
    assert loads == [workspace]
    assert workspace != config.vault
    from tests.new.immutable_semantic_helpers import published_root

    for relative in (".llmwiki/schema.yaml", "04-Wiki/.llmwiki/schema.yaml"):
        source = config.vault / relative
        if source.exists():
            # The private 04-Wiki directory was moved into the generation.
            copied = (
                published_root(config) / ".llmwiki/schema.yaml"
                if relative.startswith("04-Wiki/") else workspace / relative
            )
            assert copied.read_bytes() == source.read_bytes()


def assert_input_hash(config, name, raw):
    """Bind original human bytes to the pointer AND private input capture."""
    import hashlib
    import json

    from tests.new.immutable_semantic_helpers import published_meta

    manifest = json.loads((config.wiki_dir / "OLW-Publications/current.json").read_text())
    expected_hash = hashlib.sha256(raw).hexdigest()
    assert manifest['source_inputs_format'] == 'vault-relative-v1'
    logical = (config.sources_dir / name).relative_to(config.vault).as_posix()
    assert dict(manifest["source_inputs"])[logical] == expected_hash
    assert (published_meta(config) / "source-inputs" / name).read_bytes() == raw
    assert (config.sources_dir / name).read_bytes() == raw
