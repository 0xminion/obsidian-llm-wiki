"""Strict fragment storage, raw bytes, and resealed publication consumer controls."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from obsidian_llm_wiki.core.models import source_synthesis_from_dict, source_synthesis_to_dict
from obsidian_llm_wiki.synth.review import verify_review_receipt_integrity
from tests.new.test_evidence_fragments import (
    CONTENT,
    approved,
    parsed,
)
from tests.new.test_evidence_fragments import (
    test_fragment_consumers_pipeline_cache_registry_preview_query_and_noop as build_publication,
)


def test_fragment_schema_has_its_own_current_compatibility_epoch():
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.pipeline import _artifact_versions_for_config

    assert _artifact_versions_for_config(Config()).schema == "synthesis-schema-v5-fragments"


@pytest.mark.parametrize(
    "content",
    [
        "Birds can\r\nswim. Gap. Birds can\r\nfly.",
        "Birds can\nswim. Gap. Birds can\nfly.",
        "青鸟可\r\n游泳。间隔。青鸟可\r\n飞行。",
        "青鸟可\n游泳。间隔。青鸟可\n飞行。",
    ],
)
def test_every_fragment_retains_internal_raw_newlines_and_utf8_offsets(content):
    catalog, _, synthesis = parsed(content)
    for fragment in synthesis.concepts[0].claims[0].evidence.fragments:
        assert content[fragment.start_offset : fragment.end_offset] == fragment.quote
        assert content.encode()[fragment.start_byte : fragment.end_byte] == fragment.quote.encode()
        assert "\n" in fragment.quote
    assert catalog.revision == synthesis.concepts[0].claims[0].evidence.fragments[1].source_hash


@pytest.mark.parametrize(
    "damage", ["unknown", "extra", "bool", "missing", "empty", "nested", "bad-id"]
)
def test_malformed_fragment_cache_is_not_repaired_or_silently_dropped(damage):
    _, synthesis, _, _ = approved()
    data = source_synthesis_to_dict(synthesis)
    evidence = data["concepts"][0]["claims"][0]["evidence"]
    fragment = evidence["fragments"][1]
    if damage == "unknown":
        evidence["version"] = "future"
    elif damage == "extra":
        evidence["quote"] = "joined fake"
    elif damage == "bool":
        fragment["start_byte"] = True
    elif damage == "missing":
        del fragment["source_hash"]
    elif damage == "empty":
        evidence["fragments"] = []
    elif damage == "nested":
        evidence["fragments"][1] = deepcopy(evidence)
    else:
        fragment["span_ids"] = [None, fragment["span_ids"][0]]
    with pytest.raises(ValueError):
        source_synthesis_from_dict(data)


def test_resealed_second_fragment_context_is_rejected():
    from obsidian_llm_wiki.synth.review_context import build_fragment_context_binding

    catalog, synthesis, _, _ = approved()
    binding = next(
        o["context_binding"]
        for o in synthesis.review_receipt["units"].values()
        if o["kind"] == "claim"
    )
    fragments = [
        {key: fragment[key] for key in ("span_ids", "start_offset", "end_offset")}
        for fragment in binding["cited_fragments"]
    ]
    fragments[1]["start_offset"] += 1
    # Rebuild all inner hashes/IDs around a source-exact but wrong citation offset.
    replacement = build_fragment_context_binding(
        content=CONTENT,
        source_file=catalog.source_file,
        source_sha256=catalog.revision,
        catalog_version=catalog.version,
        unit_key=binding["unit_key"],
        cited_fragments=fragments,
        visible_intervals=binding["visible_intervals"],
    )
    binding.clear()
    binding.update(replacement)
    assert verify_review_receipt_integrity(synthesis, CONTENT, "bird.md")


def test_resealed_generation_second_fragment_fails_preview_and_query(tmp_path, monkeypatch):
    from obsidian_llm_wiki.config import Config, LLMProviderConfig
    from obsidian_llm_wiki.core.identity_migration import preview_identity_migration
    from obsidian_llm_wiki.core.identity_plan import canonical, digest
    from obsidian_llm_wiki.query.graph import build_graph_from_vault
    from tests.new.immutable_semantic_helpers import published_root
    from tests.new.migration_fixture_helpers import (
        closed_registry,
        reseal_inner_generation,
        reseal_outer,
    )

    build_publication(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "offline-fragments")
    config = Config(vault_path=str(tmp_path), llm=LLMProviderConfig(model="offline-fragments"))
    root = published_root(config)
    cli = SimpleNamespace(wiki=config.wiki_dir, generation=root)
    meta = root / ".llmwiki"
    before = preview_identity_migration(tmp_path)
    binding = next(b for b in before["bindings"] if b["source_file"] == "bird.md")
    assert binding["evidence"]["historical_admission_integrity"] is True
    assert binding["evidence"]["catalog_span_citations"] == 2
    marker_path = meta / "identity-publication.json"
    marker = json.loads(marker_path.read_text())
    raw = json.loads((meta / "publications" / (marker["plan_id"] + ".json")).read_text())
    payloads = {name: json.loads(payload) for name, payload in raw["sources"]}
    payloads["bird.md"]["concepts"][0]["claims"][0]["evidence"]["fragments"][1]["quote"] = "伪造"
    plan = dict(raw, sources=[(name, canonical(p)) for name, p in sorted(payloads.items())])
    plan_id = digest(plan)
    with closed_registry(meta, edit=True) as registry:
        registry.connection.execute(
            "UPDATE synthesis SET payload_json=? "
            "WHERE source_id=(SELECT id FROM source WHERE filename='bird.md')",
            (json.dumps(payloads["bird.md"]),),
        )
    cache = meta / "cache" / "bird.md.json"
    data = json.loads(cache.read_text())
    data["payload"] = payloads["bird.md"]
    cache.write_text(canonical(data))
    (meta / "publications" / (plan_id + ".json")).write_text(canonical(plan))
    marker.update(
        plan_id=plan_id, source_payloads={name: digest(p) for name, p in payloads.items()}
    )
    marker_path.write_text(canonical(marker))
    reseal_inner_generation(cli)
    pointer = config.wiki_dir / "OLW-Publications" / "current.json"
    manifest = json.loads(pointer.read_text())
    manifest["identity_plan_id"] = plan_id
    pointer.write_text(canonical(manifest))
    reseal_outer(cli)
    report = preview_identity_migration(tmp_path)
    assert report["publication"]["verified"] is True  # Valid outer hashes are not semantic proof.
    binding = next(b for b in report["bindings"] if b["source_file"] == "bird.md")
    assert binding["status"] == "unresolved"
    assert binding["evidence"]["historical_admission_integrity"] is False
    assert any(b["kind"] == "invalid_admission_receipt" for b in report["blockers"])
    graph = build_graph_from_vault(config.wiki_dir, config=config)
    assert not any(p.current_eligible for p in graph.pages.values())
