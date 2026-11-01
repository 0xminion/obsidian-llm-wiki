"""Public CLI no-op guards that bypass the checkpoint-only fast path."""

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from tests.new.immutable_semantic_helpers import (
    assert_preserved,
    file_state,
    published_meta,
    published_registry,
    published_root,
)
from tests.new.test_publication_identity_cli import Endpoint, _legacy_merge
from tests.new.test_publication_identity_cli import cli as cli


def refresh(cli):
    return CliRunner().invoke(app, ["build", str(cli.wiki.parent), "--refresh-render"])


def payloads(cli):
    with published_registry(cli.wiki) as registry:
        return [
            tuple(r)
            for r in registry.connection.execute(
                "SELECT source_id,payload_json FROM synthesis ORDER BY source_id"
            )
        ]


def test_cli_split_terminal_different_survives_forced_refresh(cli, monkeypatch):
    _legacy_merge(cli, monkeypatch, "aspirin", "ibuprofen")
    endpoint = cli.use(Endpoint(identity={("Ibuprofen", "Aspirin"): "different"}))
    result = cli.build()
    assert result.exit_code == 0, result.output
    assert endpoint.requests == ["identity"]
    assert cli.concept_titles() == {"aspirin": "Aspirin", "ibuprofen": "Ibuprofen"}
    assert "Ibuprofen irritates the stomach lining. [1]" in cli.page("concepts/ibuprofen.md")
    assert "stomach lining" not in cli.page("concepts/aspirin.md")
    generation = published_root(cli.wiki)
    before, admitted = file_state(generation), payloads(cli)
    for _ in range(2):
        result = refresh(cli)
        assert result.exit_code == 0, result.output
        assert endpoint.requests == ["identity"]
        assert file_state(generation) == before
        assert payloads(cli) == admitted
        assert_preserved(cli.wiki, cli.legacy_state)
        assert cli.concept_titles() == {"aspirin": "Aspirin", "ibuprofen": "Ibuprofen"}
        assert "Ibuprofen irritates the stomach lining. [1]" in cli.page("concepts/ibuprofen.md")
        assert "stomach lining" not in cli.page("concepts/aspirin.md")


@pytest.mark.parametrize("coowner", [False, True])
def test_cli_assessed_revision_refresh_preserves_binding_payload_and_marker(cli, coowner):
    endpoint = cli.use(Endpoint(identity={("Aspirin", "Aspirin"): "same"}))
    cli.add("aspirin")
    assert cli.build().exit_code == 0
    if coowner:
        cli.add("aspirin2")
        assert cli.build().exit_code == 0
    original_generation = published_root(cli.wiki)
    original_state = file_state(original_generation)
    marker_path = published_meta(cli.wiki) / "identity-publication.json"
    old_marker = marker_path.read_bytes()
    source = cli.wiki / "sources" / "aspirin.md"
    source.write_text(source.read_text() + "A revised study confirms the same medicine.\n")
    old_calls = endpoint.requests.count("identity")
    result = cli.build()
    assert result.exit_code == 0, result.output
    assert endpoint.requests.count("identity") == old_calls + 1
    assert (published_meta(cli.wiki) / "identity-publication.json").read_bytes() != old_marker
    assert file_state(original_generation) == original_state
    assert marker_path.read_bytes() == old_marker
    assert cli.concept_titles() == {"aspirin": "Aspirin"}
    page = cli.page("concepts/aspirin.md")
    assert "Aspirin reduces fever in adults. [" in page
    if coowner:
        assert "Aspirin reduces fever in children. [" in page
    with published_registry(cli.wiki) as registry:
        admitted = registry.load_synthesis("aspirin.md")
        bindings = admitted.review_receipt["publication_binding"]
        assert {b["basis"] for b in bindings.values()} == {"assessed same entity"}
    saved_payloads = payloads(cli)
    generation = published_root(cli.wiki)
    before, calls = file_state(generation), list(endpoint.requests)
    marker_path = generation / ".llmwiki/identity-publication.json"
    marker = marker_path.read_bytes()
    for _ in range(2):
        result = refresh(cli)
        assert result.exit_code == 0, result.output
        assert endpoint.requests == calls
        assert file_state(generation) == before
        assert marker_path.read_bytes() == marker
        assert payloads(cli) == saved_payloads
        with published_registry(cli.wiki) as registry:
            assert (
                registry.load_synthesis("aspirin.md").review_receipt["publication_binding"]
                == bindings
            )
