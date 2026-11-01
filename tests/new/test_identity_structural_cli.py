"""Public CLI/HTTP verticals for revision retarget, evidence, and all consumers."""

import pytest

from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.models import source_synthesis_to_dict
from tests.new import test_publication_identity_cli as harness
from tests.new.grounding_doubles import cite, grounded_response
from tests.new.immutable_semantic_helpers import (
    file_state,
    published_meta,
    published_registry,
    published_root,
)

cli = harness.cli
KERNEL = "The Packet Scheduler is a kernel module that orders packets."
POSTAL = "The Packet Scheduler is a postal robot that sorts parcels."
LINKS = (
    "Router refers to the kernel module. "
    "Router depends on Packet Scheduler and packet-scheduler. "
    "Stack includes packet-scheduler."
)


def _producer(prompt, **kwargs):
    text = prompt.split("--- SOURCE DOCUMENT", 1)[-1]
    if "Router refers" in text:
        return grounded_response(
            summary="Router refers to the kernel module.",
            concepts=[
                {
                    "title": "Router",
                    "slug": "router",
                    "summary": "Router refers to the kernel module.",
                    "sections": [
                        {"heading": "Overview", "points": ["Router refers to the kernel module."]}
                    ],
                    "claims": [
                        {
                            "text": "Router refers to the kernel module.",
                            "evidence_span_ids": cite(prompt, "Router refers"),
                        }
                    ],
                    "related": [
                        {"slug": "packet-scheduler", "display": label, "relation": "depends_on"}
                        for label in ["Packet Scheduler", "packet-scheduler"]
                    ],
                }
            ],
            maps=[
                {
                    "title": "Stack",
                    "slug": "stack",
                    "summary": "Stack includes packet-scheduler.",
                    "concept_slugs": ["router", "packet-scheduler"],
                }
            ],
        )
    postal = "postal robot" in text
    claim = POSTAL if postal else KERNEL
    candidate = harness._candidate(prompt, "packet-scheduler")
    c = {
        "title": "Packet Scheduler",
        "slug": "packet-scheduler",
        "summary": claim,
        "sections": [{"heading": "Overview", "points": [claim]}],
        "claims": [
            {
                "text": claim,
                "evidence_span_ids": cite(prompt, "postal robot" if postal else "kernel module"),
            }
        ],
        "canonical_id": candidate,
    }
    return grounded_response(summary=claim, concepts=[c])


def _verdict(item):
    subject = " ".join(span["text"] for span in item["subject_context"]["spans"])
    candidates = " ".join(
        span["text"] for owner in item["candidate_context"]["owners"] for span in owner["spans"]
    )
    if ("postal robot" in subject) != ("postal robot" in candidates):
        return "different"
    return (
        "same" if "kernel module" in subject and "kernel module" in candidates else "insufficient"
    )


def _write(cli, filename, body):
    (cli.wiki / "sources" / filename).write_text(f"---\ntitle: {filename}\n---\n{body}\n")


@pytest.mark.parametrize("reverse", [False, True])
def test_real_cli_same_run_retarget_never_mixes_or_rebinds_foreign_occurrences(
    cli, monkeypatch, reverse
):
    monkeypatch.setattr(harness, "producer", _producer)
    endpoint = cli.use(harness.Endpoint(identity=_verdict))
    owner, reuse = ("a-owner.md", "z-reuse.md") if not reverse else ("z-owner.md", "a-reuse.md")
    _write(cli, owner, KERNEL)
    _write(cli, "router.md", LINKS)
    first = cli.build()
    assert first.exit_code == 0, first.output
    with published_registry(cli.wiki) as registry:
        old_id = registry.load_synthesis(owner).concepts[0].canonical_id
    assert "/concepts/packet-scheduler" in cli.page("concepts/router.md")
    _write(cli, owner, POSTAL)
    _write(cli, reuse, KERNEL)
    second = cli.build()
    assert second.exit_code == 0, second.output
    with published_registry(cli.wiki) as registry:
        syntheses = registry.load_syntheses({owner, reuse, "router.md"})
        assert syntheses[reuse].concepts[0].canonical_id == old_id
        assert syntheses[owner].concepts[0].canonical_id != old_id
        postal_slug = syntheses[owner].concepts[0].slug
        assert POSTAL not in cli.page("concepts/packet-scheduler.md")
        assert KERNEL not in cli.page(f"concepts/{postal_slug}.md")
        assert "router|" not in cli.page(f"concepts/{postal_slug}.md")
        for filename, synthesis in syntheses.items():
            assert source_synthesis_to_dict(
                load_synthesis(published_meta(cli.wiki), filename)
            ) == source_synthesis_to_dict(synthesis)
        refs = syntheses["router.md"].review_receipt["references"]
        assert len(refs) == 4
        assert all(
            r["target_id"] == old_id
            for key, r in refs.items()
            if r["original_slug"] == "packet-scheduler"
        )
    calls, snapshot = len(endpoint.requests), cli.snapshot()
    assert cli.build().exit_code == 0
    assert len(endpoint.requests) == calls and cli.snapshot() == snapshot


def test_foreign_occurrence_becomes_unresolved_not_same_name_retarget(cli, monkeypatch):
    monkeypatch.setattr(harness, "producer", _producer)
    cli.use(harness.Endpoint(identity=_verdict))
    _write(cli, "owner.md", KERNEL)
    _write(cli, "router.md", LINKS)
    assert cli.build().exit_code == 0
    _write(cli, "owner.md", POSTAL)
    result = cli.build()
    assert result.exit_code == 0, result.output
    router = cli.page("concepts/router.md")
    assert "/concepts/packet-scheduler" not in router
    assert "Packet Scheduler (depends_on)" in router
    assert "packet-scheduler (depends_on)" in router
    moc = cli.page("mocs/stack.md")
    assert "Unresolved members" in moc and "packet-scheduler — unresolved" in moc
    assert "/concepts/packet-scheduler" not in moc


def test_source_digest_defeats_same_size_same_mtime_noop(cli, monkeypatch):
    import os

    monkeypatch.setattr(harness, "producer", _producer)
    endpoint = cli.use(harness.Endpoint(identity=_verdict))
    # Equal length source buffers have deliberately identical stat signatures.
    length = max(len(KERNEL), len(POSTAL))
    _write(cli, "owner.md", KERNEL.ljust(length))
    assert cli.build().exit_code == 0
    path = cli.wiki / "sources" / "owner.md"
    stat = path.stat()
    calls = endpoint.requests.count("produce")
    _write(cli, "owner.md", POSTAL.ljust(length))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    result = cli.build()
    assert result.exit_code == 0, result.output
    assert endpoint.requests.count("produce") == calls + 1
    assert any(POSTAL in cli.page(f"concepts/{slug}.md") for slug in cli.concept_titles())


def test_checkpoint_cannot_omit_source_preconditions(cli, monkeypatch):
    import json
    import os

    monkeypatch.setattr(harness, "producer", _producer)
    endpoint = cli.use(harness.Endpoint(identity=_verdict))
    length = max(len(KERNEL), len(POSTAL))
    _write(cli, "owner.md", KERNEL.ljust(length))
    assert cli.build().exit_code == 0
    generation = published_root(cli.wiki)
    marker_path = generation / ".llmwiki" / "identity-publication.json"
    marker = json.loads(marker_path.read_text())
    marker["source_files"] = []
    marker_path.write_text(json.dumps(marker))
    source = cli.wiki / "sources" / "owner.md"
    stat = source.stat()
    _write(cli, "owner.md", POSTAL.ljust(length))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    before = file_state(generation)
    pointer = cli.wiki / "OLW-Publications/current.json"
    pointer_before = pointer.read_bytes()
    calls = list(endpoint.requests)
    result = cli.build()
    assert result.exit_code == 1, result.output
    assert "corrupt" in result.output and "inventory/digest mismatch" in result.output
    assert endpoint.requests == calls
    assert file_state(generation) == before
    assert pointer.read_bytes() == pointer_before
    assert POSTAL in source.read_text()


def test_source_change_after_plan_cannot_commit_success(cli, monkeypatch):
    from obsidian_llm_wiki.core import pipeline

    monkeypatch.setattr(harness, "producer", _producer)
    cli.use(harness.Endpoint(identity=_verdict))
    _write(cli, "owner.md", KERNEL)
    assert cli.build().exit_code == 0
    snapshot = cli.page("concepts/packet-scheduler.md")
    _write(cli, "new.md", KERNEL)
    real_render = pipeline.render_vault

    def edit_after_plan(*args, **kwargs):
        _write(cli, "owner.md", POSTAL)
        return real_render(*args, **kwargs)

    monkeypatch.setattr(pipeline, "render_vault", edit_after_plan)
    result = cli.build()
    assert result.exit_code != 0
    assert "source or policy changed before publication" in result.output
    assert cli.page("concepts/packet-scheduler.md") == snapshot
    # A human's changed source is not part of generated-output rollback.
    assert POSTAL in (cli.wiki / "sources" / "owner.md").read_text()
