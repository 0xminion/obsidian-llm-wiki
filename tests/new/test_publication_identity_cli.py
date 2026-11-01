"""Public ``olw build`` CLI over the real OpenAI-compatible HTTP client.

Only httpx's transport is replaced. Producer prompts (including the bounded
candidate block a model may copy IDs from), closed review schemas, parsing,
admission, registry identity, publication planning and rendering all run
production code. Exit codes are the CLI's real exit codes.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.providers import llm
from tests.new.grounding_doubles import cite, grounded_response, identity_answer, review_answer
from tests.new.immutable_semantic_helpers import (
    assert_preserved,
    file_state,
    generated_link,
    published_root,
    seed_legacy_cache,
)


def _candidate(prompt: str, slug: str) -> str:
    match = re.search(rf"^  - (concept-[0-9a-f]+) \| {re.escape(slug)} \|", prompt, re.M)
    return match.group(1) if match else ""


def _concept(title, slug, text, prompt, phrase, **extra):
    return {
        "title": title,
        "slug": slug,
        "summary": text,
        "sections": [{"heading": "Overview", "points": [text]}],
        "claims": [{"text": text, "evidence_span_ids": cite(prompt, phrase)}],
        **extra,
    }


def _with_id(response: str, concept_id: str) -> str:
    data = json.loads(response)
    data["concepts"][0]["canonical_id"] = concept_id
    return json.dumps(data, ensure_ascii=False)


SOURCES = {
    "aspirin": "Aspirin reduces fever in adults. It is taken by mouth as a tablet.\n",
    "ibuprofen": "Ibuprofen irritates the stomach lining. Like aspirin, it is taken by mouth.\n",
    "aspirin2": "Aspirin reduces fever in children and is swallowed by mouth.\n",
    "zh": "阿司匹林（aspirin）可以退烧，也可以缓解轻度的头痛和牙痛。\n",
    "lang": "Python is a programming language with dynamic types and garbage collection.\n",
    "snake": "Python is a family of nonvenomous constricting snakes in tropical forests.\n",
    "stack": "Router v2 depends on the Packet Scheduler for ordering.\n",
    "annealer": "The Quantum Annealer is a lab optimisation device.\n",
    "sched": "The Packet Scheduler is a kernel module that orders packets.\n",
    "drugs": "Aspirin is an analgesic drug. Ibuprofen is an analgesic drug too.\n",
}


def producer(prompt: str, target_slug: str = "quantum-annealer", moc_tag: str = "") -> str:
    body = prompt.split("--- SOURCE DOCUMENT", 1)[-1]
    if "Aspirin reduces fever in adults" in body:
        return grounded_response(
            summary="Aspirin reduces fever in adults.",
            concepts=[
                _concept(
                    "Aspirin",
                    "aspirin",
                    "Aspirin reduces fever in adults.",
                    prompt,
                    "reduces fever",
                )
            ],
        )
    if "stomach lining" in body:
        # Copies Aspirin's offered stable ID AND canonical slug (the prompt allows it).
        return _with_id(
            grounded_response(
                summary="Ibuprofen irritates the stomach lining.",
                concepts=[
                    _concept(
                        "Ibuprofen",
                        "aspirin",
                        "Ibuprofen irritates the stomach lining.",
                        prompt,
                        "stomach lining",
                    )
                ],
            ),
            _candidate(prompt, "aspirin"),
        )
    if "children" in body:
        return _with_id(
            grounded_response(
                summary="Aspirin reduces fever in children.",
                concepts=[
                    _concept(
                        "Aspirin",
                        "aspirin",
                        "Aspirin reduces fever in children.",
                        prompt,
                        "children",
                    )
                ],
            ),
            _candidate(prompt, "aspirin"),
        )
    if "退烧" in body:
        return _with_id(
            grounded_response(
                summary="阿司匹林可以退烧。",
                language="zh",
                concepts=[
                    _concept(
                        "阿司匹林",
                        "阿司匹林",
                        "阿司匹林可以退烧。",
                        prompt,
                        "退烧",
                        aliases=["aspirin"],
                    )
                ],
            ),
            _candidate(prompt, "aspirin"),
        )
    if "programming language" in body:
        return grounded_response(
            summary="Python is a programming language.",
            concepts=[
                _concept(
                    "Python",
                    "python",
                    "Python is a programming language with dynamic types.",
                    prompt,
                    "programming language",
                )
            ],
        )
    if "constricting snakes" in body:
        return _with_id(
            grounded_response(
                summary="Python is a family of snakes.",
                concepts=[
                    _concept(
                        "Python",
                        "python",
                        "Python is a family of nonvenomous constricting snakes.",
                        prompt,
                        "constricting",
                    )
                ],
            ),
            _candidate(prompt, "python"),
        )
    if "Router v2" in body:
        concept = _concept(
            "Router v2",
            "router-v2",
            "Router v2 depends on the Packet Scheduler.",
            prompt,
            "depends on",
        )
        concept["related"] = [
            {"slug": target_slug, "relation": "depends_on", "display": "Packet Scheduler"}
        ]
        return grounded_response(
            summary="Router v2 depends on the Packet Scheduler.", concepts=[concept]
        )
    if "Quantum Annealer" in body:
        return grounded_response(
            summary="The Quantum Annealer is a lab device.",
            concepts=[
                _concept(
                    "Quantum Annealer",
                    "quantum-annealer",
                    "The Quantum Annealer is a lab optimisation device.",
                    prompt,
                    "lab",
                )
            ],
        )
    if "kernel module" in body:
        return grounded_response(
            summary="The Packet Scheduler orders packets.",
            concepts=[
                _concept(
                    "Packet Scheduler",
                    "packet-scheduler",
                    "The Packet Scheduler is a kernel module that orders packets.",
                    prompt,
                    "kernel module",
                )
            ],
        )
    data = json.loads(
        grounded_response(
            summary="Aspirin is an analgesic drug.",
            concepts=[
                _concept(
                    "Aspirin drug",
                    "aspirin-drug",
                    "Aspirin is an analgesic drug.",
                    prompt,
                    "Aspirin is an analgesic",
                ),
                _concept(
                    "Ibuprofen drug",
                    "ibuprofen-drug",
                    "Ibuprofen is an analgesic drug too.",
                    prompt,
                    "Ibuprofen is an analgesic",
                ),
            ],
            maps=[
                {
                    "title": "Analgesic drug",
                    "slug": "analgesic-drug",
                    "summary": "Aspirin is an analgesic drug.",
                    "concept_slugs": ["aspirin-drug", "ibuprofen-drug"],
                }
            ],
        )
    )
    data["maps"][0]["tags"] = [moc_tag] if moc_tag else []
    return json.dumps(data, ensure_ascii=False)


class Endpoint:
    def __init__(self, verdicts=None, identity=None, **producer_options):
        self.requests: list[str] = []
        self.verdicts = verdicts or {}
        self.identity = identity or {}
        self.options = producer_options

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "gen-model"}]})
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        prompt = body["messages"][-1]["content"]
        if "reviews" in schema["properties"]:
            self.requests.append("review")
            content = json.dumps(review_answer(prompt, schema, self.verdicts), ensure_ascii=False)
        elif "assessments" in schema["properties"]:
            self.requests.append("identity")
            content = json.dumps(identity_answer(prompt, schema, self.identity), ensure_ascii=False)
        else:
            self.requests.append("produce")
            content = producer(prompt, **self.options)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            },
        )


@pytest.fixture
def cli(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    sources = vault / "04-Wiki" / "sources"
    sources.mkdir(parents=True)
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_PROVIDER=openai\nLLM_HOST=https://fixture.invalid/v1\n"
        "LLM_MODEL=gen-model\nEMBEDDINGS_ENABLED=false\nMIN_SOURCE_CHARS=1\nRETRY_COUNT=1\n",
        encoding="utf-8",
    )
    client_type = httpx.Client
    state = {"endpoint": Endpoint()}
    monkeypatch.setattr(
        llm.httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(lambda request: state["endpoint"](request)),
            trust_env=False,
            **kwargs,
        ),
    )

    class Cli:
        wiki = vault / "04-Wiki"

        def use(self, endpoint):
            state["endpoint"] = endpoint
            return endpoint

        def add(self, *names):
            for name in names:
                (sources / f"{name}.md").write_text(
                    f"---\ntitle: {name}\n---\n{SOURCES[name]}",
                    encoding="utf-8",
                )

        def build(self):
            return CliRunner().invoke(app, ["build", str(vault)])

        def page(self, rel):
            path = published_root(self.wiki) / rel
            return path.read_text(encoding="utf-8") if path.exists() else ""

        def concept_titles(self):
            return {
                path.stem: re.search(r"^# (.+)$", path.read_text(encoding="utf-8"), re.M).group(1)
                for path in sorted((published_root(self.wiki) / "concepts").glob("*.md"))
                if path.name != "index.md"
            }

        def snapshot(self):
            return {
                str(p.relative_to(self.wiki)): p.read_bytes()
                for p in sorted(self.wiki.rglob("*.md"))
                if p.name != "log.md"
            }

    return Cli()


def _two_step(cli, first, second, endpoint=None):
    cli.use(endpoint or Endpoint())
    cli.add(first)
    assert cli.build().exit_code == 0
    cli.add(second)
    result = cli.build()
    return result


def test_cli_forged_candidate_id_is_not_merge_authority_and_rebuild_is_noop(cli):
    endpoint = Endpoint()
    result = _two_step(cli, "aspirin", "ibuprofen", endpoint)
    assert result.exit_code == 0, result.output
    assert cli.concept_titles() == {"aspirin": "Aspirin", "ibuprofen": "Ibuprofen"}
    assert "stomach" not in cli.page("concepts/aspirin.md")
    assert "Ibuprofen irritates the stomach lining. [1]" in cli.page("concepts/ibuprofen.md")
    before, calls = cli.snapshot(), len(endpoint.requests)
    again = cli.build()
    assert again.exit_code == 0 and len(endpoint.requests) == calls
    assert cli.snapshot() == before


def test_cli_same_label_reuse_without_assessment_keeps_both_reviewed_claims_apart(cli):
    endpoint = Endpoint()  # the assessor double never asserts identity by default
    result = _two_step(cli, "aspirin", "aspirin2", endpoint)
    assert result.exit_code == 0, result.output
    assert endpoint.requests.count("identity") == 1
    titles = cli.concept_titles()
    assert sorted(titles.values()) == ["Aspirin", "Aspirin"] and "aspirin" in titles
    assert "children" not in cli.page("concepts/aspirin.md")
    other = next(slug for slug in titles if slug != "aspirin")
    assert "Aspirin reduces fever in children. [" in cli.page(f"concepts/{other}.md")


def test_cli_valid_same_label_reuse_merges_both_reviewed_claims(cli):
    endpoint = Endpoint(identity={("Aspirin", "Aspirin"): "same"})
    result = _two_step(cli, "aspirin", "aspirin2", endpoint)
    assert result.exit_code == 0, result.output
    assert cli.concept_titles() == {"aspirin": "Aspirin"}
    page = cli.page("concepts/aspirin.md")
    assert "Aspirin reduces fever in adults. [" in page
    assert "Aspirin reduces fever in children. [" in page
    assert "[!warning]" not in page
    before, calls = cli.snapshot(), len(endpoint.requests)
    assert cli.build().exit_code == 0 and len(endpoint.requests) == calls
    assert cli.snapshot() == before


def test_cli_bilingual_assessed_reuse_merges(cli):
    endpoint = Endpoint(identity={("阿司匹林", "Aspirin"): "same"})
    result = _two_step(cli, "aspirin", "zh", endpoint)
    assert result.exit_code == 0, result.output
    assert cli.concept_titles() == {"aspirin": "Aspirin"}
    page = cli.page("concepts/aspirin.md")
    assert "阿司匹林可以退烧。" in page and "阿司匹林" in page.split("---")[1]


def test_cli_bilingual_alias_without_assessment_stays_distinct(cli):
    result = _two_step(cli, "aspirin", "zh")
    assert result.exit_code == 0, result.output
    assert sorted(cli.concept_titles().values()) == ["Aspirin", "阿司匹林"]
    assert "退烧" not in cli.page("concepts/aspirin.md")


def test_cli_homonym_with_copied_id_keeps_two_pages(cli):
    endpoint = Endpoint(identity={("Python", "Python"): "different"})
    result = _two_step(cli, "lang", "snake", endpoint)
    assert result.exit_code == 0, result.output
    titles = cli.concept_titles()
    assert list(titles.values()) == ["Python", "Python"] and "python" in titles
    assert "snakes" not in cli.page("concepts/python.md")
    assert endpoint.requests.count("identity") == 1


@pytest.mark.parametrize("target_first", [True, False])
def test_cli_external_label_mismatch_has_no_live_edge_or_backlink(cli, target_first):
    first, second = ("annealer", "stack") if target_first else ("stack", "annealer")
    result = _two_step(cli, first, second)
    assert result.exit_code == 0, result.output
    router, annealer = cli.page("concepts/router-v2.md"), cli.page("concepts/quantum-annealer.md")
    assert (
        generated_link(cli.wiki, "concepts", "quantum-annealer", "Packet Scheduler") not in router
    )
    assert "/concepts/quantum-annealer" not in router and "Packet Scheduler (depends_on)" in router
    assert "router-v2" not in annealer


def test_cli_consistent_late_target_goes_live_with_typed_backlink(cli):
    result = _two_step(
        cli,
        "stack",
        "sched",
        Endpoint(
            target_slug="packet-scheduler",
            identity={("Packet Scheduler", "Packet Scheduler"): "same"},
        ),
    )
    assert result.exit_code == 0, result.output
    assert (
        f"- {generated_link(cli.wiki, 'concepts', 'packet-scheduler', 'Packet Scheduler')}"
        " (depends_on)"
        in cli.page("concepts/router-v2.md")
    )
    sched = cli.page("concepts/packet-scheduler.md")
    assert (
        f"- ← {generated_link(cli.wiki, 'concepts', 'router-v2', 'Router v2')}"
        " prerequisite_of — app-derived"
        in sched
    )
    assert "router-v2|prerequisite_of" not in sched  # not a reviewed frontmatter relation


def test_cli_moc_tag_and_members_are_reviewed(cli):
    cli.use(Endpoint(moc_tag="analgesic"))
    cli.add("drugs")
    assert cli.build().exit_code == 0
    moc = cli.page("mocs/analgesic-drug.md")
    assert generated_link(cli.wiki, "concepts", "aspirin-drug") in moc and "app-derived" not in moc


def test_cli_unsupported_moc_tag_fails_closed(cli):
    cli.use(Endpoint(verdicts={"demolished": "insufficient"}, moc_tag="demolished-structures"))
    cli.add("drugs")
    result = cli.build()
    assert result.exit_code == 1
    assert cli.page("mocs/analgesic-drug.md") == ""


# ── cache upgrade, support guards and assessor failure (public CLI) ─────────


def _legacy_merge(cli, monkeypatch, first, second):
    """Prepublication legacy fixture; no immutable generation is rewritten."""
    import asyncio

    from obsidian_llm_wiki.config import load_config
    from obsidian_llm_wiki.ingest.sources import load_sources_from_dir

    cli.use(Endpoint())
    cli.add(first, second)
    config = load_config(env_file=str(cli.wiki.parent / ".env"))
    sources = load_sources_from_dir(config.sources_dir)
    asyncio.run(
        seed_legacy_cache(
            config,
            {f"{first}.md": sources[f"{first}.md"], f"{second}.md": sources[f"{second}.md"]},
            merge=True,
        )
    )
    assert not (cli.wiki / "OLW-Publications").exists()
    assert list(cli.concept_titles()) == ["aspirin"]
    cli.legacy_state = file_state(cli.wiki)


@pytest.mark.parametrize(
    ("second", "identity", "merged"),
    [
        ("aspirin2", {("Aspirin", "Aspirin"): "same"}, True),  # valid reuse is kept
        ("aspirin2", {}, False),  # same label, no assessment: split, no fact lost
        ("ibuprofen", {("Ibuprofen", "Aspirin"): "different"}, False),  # forged reuse
    ],
)
def test_cli_cache_upgrade_reassesses_legacy_merges_without_producer_calls(
    cli,
    monkeypatch,
    second,
    identity,
    merged,
):
    _legacy_merge(cli, monkeypatch, "aspirin", second)
    endpoint = cli.use(Endpoint(identity=identity))
    result = cli.build()
    assert result.exit_code == 0, result.output
    assert endpoint.requests == ["identity"]  # receipts reused: no produce/review calls
    assert_preserved(cli.wiki, cli.legacy_state)
    titles = cli.concept_titles()
    assert (list(titles) == ["aspirin"]) is merged
    marker = "children" if second == "aspirin2" else "stomach lining"
    assert (marker in cli.page("concepts/aspirin.md")) is merged
    assert any(marker in cli.page(f"concepts/{slug}.md") for slug in titles)
    before = cli.snapshot()
    assert cli.build().exit_code == 0 and endpoint.requests == ["identity"]
    assert cli.snapshot() == before


def test_cli_assessed_reuse_keeps_the_support_guard(cli):
    endpoint = Endpoint(
        verdicts={"children": "insufficient"}, identity={("Aspirin", "Aspirin"): "same"}
    )
    result = _two_step(cli, "aspirin", "aspirin2", endpoint)
    assert result.exit_code == 1
    assert "identity" not in endpoint.requests  # unsupported output is never assessed
    assert "children" not in "".join(cli.page(f"concepts/{s}.md") for s in cli.concept_titles())


def test_cli_assessor_failure_fails_visibly_and_never_merges(cli):
    endpoint = Endpoint(identity=lambda item: "maybe")  # breaks the closed vocabulary
    result = _two_step(cli, "aspirin", "aspirin2", endpoint)
    assert result.exit_code == 1
    assert "identity:aspirin2.md" in result.output
    assert "children" not in cli.page("concepts/aspirin.md")
