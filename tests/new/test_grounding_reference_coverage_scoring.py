"""Coverage scorer: aggregates explicit judgments; never credits missing or unknown ones.

All obligations, ids and outputs below are invented for the test. They do not
reproduce any benchmark source, requirement, label or held-out item.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "benchmarks/grounding_reference/score_coverage.py"
spec = importlib.util.spec_from_file_location("grounding_reference_coverage", PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

OUT_S1 = b"invented summary for source one\n"
OUT_S2 = "\u865a\u6784\u6458\u8981\n".encode()
PRODUCER = "producer-run-x"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def req(rid, essential, group=None):
    r = {"id": rid, "text": f"invented requirement {rid}", "essential": essential}
    if group:
        r["group"] = group
    return r


def obligations():
    return {
        "version": "fixture-v2",
        "sources": [
            {
                "source_id": "S1",
                "summary_language_required": "en",
                "must_cover": [
                    {
                        "id": "S1-C1",
                        "hit_rule": {"type": "all_essential"},
                        "requirements": [
                            req("S1-C1.1", True),
                            req("S1-C1.2", True),
                            req("S1-C1.3", False),
                        ],
                    },
                    {
                        "id": "S1-C2",
                        "hit_rule": {"type": "at_least_k_groups", "k": 1, "groups": ["a", "b"]},
                        "requirements": [
                            req("S1-C2.1", True, "a"),
                            req("S1-C2.2", True, "b"),
                            req("S1-C2.3", False, "b"),
                        ],
                    },
                    {
                        "id": "S1-C3",
                        "hit_rule": {"type": "all_essential"},
                        "requirements": [req("S1-C3.1", True)],
                    },
                    {
                        "id": "S1-C4",
                        "hit_rule": {"type": "all_essential"},
                        "conditional": {
                            "applies_if": "the output states an invented figure",
                            "if_not_applicable": "excluded from n",
                            "rule": "CV4",
                        },
                        "requirements": [req("S1-C4.1", True)],
                    },
                ],
                "must_preserve_qualifiers": [{"id": "S1-Q1", "requirement": "keep the hedge"}],
                "must_not_assert": [{"id": "S1-N1", "assertion": "an invented falsehood"}],
            },
            {
                "source_id": "S2",
                "summary_language_required": "zh-Hans",
                "must_cover": [
                    {
                        "id": "S2-C1",
                        "hit_rule": {"type": "all_essential"},
                        "requirements": [req("S2-C1.1", True)],
                    }
                ],
                "must_preserve_qualifiers": [],
                "must_not_assert": [{"id": "S2-N1", "assertion": "another invented falsehood"}],
            },
        ],
    }


OBL_SHA = "0" * 64


def judgment_s1():
    return {
        "source_id": "S1",
        "output_sha256": sha(OUT_S1),
        "adjudication": {"judge_id": "judge-1", "independent": True, "method": "invented"},
        "language": {"observed": "en"},
        "applicability": {"S1-C4": "applicable"},
        "requirements": {
            "S1-C1.1": "hit",
            "S1-C1.2": "hit",
            "S1-C1.3": "hit",
            "S1-C2.1": "hit",
            "S1-C2.2": "hit",
            "S1-C2.3": "hit",
            "S1-C3.1": "hit",
            "S1-C4.1": "hit",
        },
        "qualifiers": {"S1-Q1": "respected"},
        "must_not": {"S1-N1": "absent"},
    }


def judgment_s2():
    return {
        "source_id": "S2",
        "output_sha256": sha(OUT_S2),
        "adjudication": {"judge_id": "judge-1", "independent": True, "method": "invented"},
        "language": {"observed": "zh-Hans"},
        "applicability": {},
        "requirements": {"S2-C1.1": "hit"},
        "qualifiers": {},
        "must_not": {"S2-N1": "absent"},
    }


def doc(*judgments):
    return {
        "reference_version": "fixture-v2",
        "coverage_obligations_sha256": OBL_SHA,
        "judgments": list(judgments),
    }


def hashes(s1=True, s2=True):
    out = {}
    if s1:
        out["S1"] = sha(OUT_S1)
    if s2:
        out["S2"] = sha(OUT_S2)
    return out


def run(judgments, output_hashes=None, obl=None):
    return module.score(
        obl or obligations(),
        judgments,
        output_hashes if output_hashes is not None else hashes(),
        producer=PRODUCER,
        obligations_sha256=OBL_SHA,
    )


def test_complete_full_coverage_passes_both_rubrics():
    rep = run(doc(judgment_s1(), judgment_s2()))
    s1 = rep["sources"]["S1"]
    assert rep["complete"] is True
    assert s1["status"] == "complete"
    assert s1["items_applicable"] == 4 and s1["threshold"] == 3
    assert s1["frozen_v2"]["source_pass"] is True
    assert s1["strict"]["source_pass"] is True
    assert s1["atomic_recall"] == 1.0
    agg = rep["aggregate"]
    assert agg["sources_expected"] == 2
    assert agg["frozen_v2"]["sources_passed"] == 2
    assert agg["strict"]["sources_passed"] == 2


def test_report_states_it_aggregates_judgments_not_truth():
    rep = run(doc(judgment_s1(), judgment_s2()))
    assert "not" in rep["disclaimer"] and "judgment" in rep["disclaimer"]


def test_strict_requires_every_bundle_group_while_frozen_rule_needs_k():
    j = judgment_s1()
    j["requirements"]["S1-C1.1"] = "miss"
    j["requirements"]["S1-C2.2"] = "miss"  # group b essential missed; group a hit
    s1 = run(doc(j, judgment_s2()))["sources"]["S1"]
    assert s1["items"]["S1-C2"]["frozen_hit"] is True
    assert s1["items"]["S1-C2"]["strict_hit"] is False
    assert s1["items"]["S1-C2"]["groups_hit"] == ["a"]
    assert s1["frozen_v2"]["item_hits"] == 3
    assert s1["frozen_v2"]["coverage_pass"] is True
    assert s1["strict"]["item_hits"] == 2
    assert s1["strict"]["coverage_pass"] is False
    assert s1["strict"]["all_items_hit"] is False
    assert s1["frozen_v2"]["source_pass"] is True
    assert s1["strict"]["source_pass"] is False


def test_nonessential_miss_costs_partial_credit_only():
    j = judgment_s1()
    j["requirements"]["S1-C1.3"] = "miss"
    s1 = run(doc(j, judgment_s2()))["sources"]["S1"]
    assert s1["items"]["S1-C1"]["frozen_hit"] is True
    assert s1["items"]["S1-C1"]["strict_hit"] is True
    assert s1["items"]["S1-C1"]["credit"] == round(2 / 3, 4)
    assert s1["requirements_hit"] == 7 and s1["requirements_applicable"] == 8
    assert s1["essential_recall"] == 1.0


def test_missing_conditional_applicability_never_defaults_to_not_applicable():
    j = judgment_s1()
    del j["applicability"]["S1-C4"]
    s1 = run(doc(j, judgment_s2()))["sources"]["S1"]
    assert s1["status"] == "incomplete"
    assert any("S1-C4" in r for r in s1["incomplete"])
    assert s1["items_applicable"] == 4  # stays in the denominator
    assert s1["not_applicable_items"] == []
    assert s1["items"]["S1-C4"]["applicability"] == "unjudged"
    assert s1["items"]["S1-C4"]["frozen_hit"] is False  # no credit
    assert s1["requirements_applicable"] == 8 and s1["requirements_hit"] == 7
    assert s1["frozen_v2"]["source_pass"] is None
    assert s1["strict"]["source_pass"] is None


@pytest.mark.parametrize("value", ["unknown"])
def test_unknown_applicability_is_incomplete(value):
    j = judgment_s1()
    j["applicability"]["S1-C4"] = value
    s1 = run(doc(j, judgment_s2()))["sources"]["S1"]
    assert s1["status"] == "incomplete"
    assert s1["items_applicable"] == 4


def test_explicit_not_applicable_removes_item_and_its_requirements():
    j = judgment_s1()
    j["applicability"]["S1-C4"] = "not_applicable"
    del j["requirements"]["S1-C4.1"]
    s1 = run(doc(j, judgment_s2()))["sources"]["S1"]
    assert s1["status"] == "complete"
    assert s1["not_applicable_items"] == ["S1-C4"]
    assert s1["items_applicable"] == 3 and s1["threshold"] == 3
    assert s1["requirements_applicable"] == 7


def test_not_applicable_requirement_marker_is_accepted_only_under_not_applicable_item():
    j = judgment_s1()
    j["applicability"]["S1-C4"] = "not_applicable"
    j["requirements"]["S1-C4.1"] = "not_applicable"
    assert run(doc(j, judgment_s2()))["sources"]["S1"]["status"] == "complete"
    j = judgment_s1()
    j["requirements"]["S1-C1.1"] = "not_applicable"
    with pytest.raises(module.CoverageInputError):
        run(doc(j, judgment_s2()))


def test_not_applicable_is_rejected_for_unconditional_items():
    j = judgment_s1()
    j["applicability"]["S1-C1"] = "not_applicable"
    with pytest.raises(module.CoverageInputError):
        run(doc(j, judgment_s2()))


def test_not_applicable_item_with_credited_requirement_is_rejected():
    j = judgment_s1()
    j["applicability"]["S1-C4"] = "not_applicable"  # S1-C4.1 still "hit"
    with pytest.raises(module.CoverageInputError):
        run(doc(j, judgment_s2()))


@pytest.mark.parametrize("how", ["missing", "unknown"])
def test_missing_or_unknown_requirement_gets_no_credit_and_is_incomplete(how):
    j = judgment_s1()
    if how == "missing":
        del j["requirements"]["S1-C3.1"]
    else:
        j["requirements"]["S1-C3.1"] = "unknown"
    s1 = run(doc(j, judgment_s2()))["sources"]["S1"]
    assert s1["status"] == "incomplete"
    assert s1["items"]["S1-C3"]["frozen_hit"] is False
    assert s1["requirements_hit"] == 7 and s1["requirements_applicable"] == 8
    assert s1["frozen_v2"]["source_pass"] is None


@pytest.mark.parametrize("section", ["qualifiers", "must_not", "language"])
def test_missing_qualifier_must_not_or_language_judgment_is_incomplete(section):
    j = judgment_s1()
    if section == "language":
        del j["language"]
    else:
        j[section] = {}
    rep = run(doc(j, judgment_s2()))
    s1 = rep["sources"]["S1"]
    assert s1["status"] == "incomplete"
    assert s1["frozen_v2"]["source_pass"] is None
    assert rep["complete"] is False
    assert rep["aggregate"]["sources_incomplete"] == 1


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("qualifiers", "S1-Q1", "violated"),
        ("must_not", "S1-N1", "violated"),
        ("language", "observed", "zh-Hans"),
    ],
)
def test_qualifier_must_not_or_language_failure_fails_source(section, key, value):
    j = judgment_s1()
    j[section][key] = value
    s1 = run(doc(j, judgment_s2()))["sources"]["S1"]
    assert s1["frozen_v2"]["coverage_pass"] is True
    assert s1["frozen_v2"]["source_pass"] is False
    assert s1["strict"]["source_pass"] is False


def test_omitted_qualifier_topic_is_respected():
    j = judgment_s1()
    j["qualifiers"]["S1-Q1"] = "omitted"
    assert run(doc(j, judgment_s2()))["sources"]["S1"]["frozen_v2"]["source_pass"] is True


def test_definite_failure_is_reported_even_when_incomplete():
    j = judgment_s1()
    j["must_not"]["S1-N1"] = "violated"
    del j["requirements"]["S1-C3.1"]
    s1 = run(doc(j, judgment_s2()))["sources"]["S1"]
    assert s1["status"] == "incomplete"
    assert s1["frozen_v2"]["source_pass"] is False


def test_missing_output_stays_in_every_denominator():
    rep = run(doc(judgment_s1()), output_hashes=hashes(s2=False))
    s2 = rep["sources"]["S2"]
    assert s2["status"] == "missing_output"
    assert s2["frozen_v2"]["source_pass"] is False
    agg = rep["aggregate"]
    assert agg["sources_expected"] == 2 and agg["sources_missing_output"] == 1
    assert agg["frozen_v2"]["sources_passed"] == 1
    assert agg["frozen_v2"]["source_pass_rate"] == 0.5
    assert agg["requirements_applicable"] == 9 and agg["requirements_hit"] == 8


def test_output_without_judgment_is_incomplete_not_passed():
    rep = run(doc(judgment_s1()))
    s2 = rep["sources"]["S2"]
    assert s2["status"] == "incomplete"
    assert s2["frozen_v2"]["source_pass"] is None
    assert rep["aggregate"]["frozen_v2"]["sources_passed"] == 1
    assert rep["complete"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["judgments"][0].update(output_sha256="f" * 64),  # judged other bytes
        lambda d: d["judgments"].append(copy.deepcopy(d["judgments"][0])),
        lambda d: d["judgments"][0].update(source_id="S9"),
        lambda d: d["judgments"][0]["requirements"].update({"S1-C9.1": "hit"}),
        lambda d: d["judgments"][0]["requirements"].update({"S2-C1.1": "hit"}),
        lambda d: d["judgments"][0]["requirements"].update({"S1-C1.1": "partial"}),
        lambda d: d["judgments"][0]["requirements"].update({"S1-C1.1": True}),
        lambda d: d["judgments"][0]["qualifiers"].update({"S1-Q1": "fine"}),
        lambda d: d["judgments"][0]["qualifiers"].update({"S1-Q9": "respected"}),
        lambda d: d["judgments"][0]["must_not"].update({"S1-N1": "no"}),
        lambda d: d["judgments"][0]["applicability"].update({"S1-C4": "maybe"}),
        lambda d: d["judgments"][0]["applicability"].update({"S1-C9": "applicable"}),
        lambda d: d["judgments"][0]["adjudication"].update(judge_id=PRODUCER),
        lambda d: d["judgments"][0]["adjudication"].update(independent=False),
        lambda d: d["judgments"][0]["adjudication"].pop("independent"),
        lambda d: d["judgments"][0]["adjudication"].update(judge_id=""),
        lambda d: d["judgments"][0].pop("adjudication"),
        lambda d: d["judgments"][0].update(extra_field=1),
        lambda d: d.update(coverage_obligations_sha256="1" * 64),
        lambda d: d.update(reference_version="other"),
        lambda d: d.update(judgments={}),
    ],
)
def test_malformed_unbound_or_non_independent_judgments_are_rejected(mutate):
    d = doc(judgment_s1(), judgment_s2())
    mutate(d)
    with pytest.raises(module.CoverageInputError):
        run(d)


def test_judgment_for_source_without_output_is_rejected():
    with pytest.raises(module.CoverageInputError):
        run(doc(judgment_s1(), judgment_s2()), output_hashes=hashes(s2=False))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o["sources"][0]["must_cover"][1]["hit_rule"].update(k=3),
        lambda o: o["sources"][0]["must_cover"][1]["hit_rule"].update(groups=["a"]),
        lambda o: o["sources"][0]["must_cover"][1]["requirements"][0].update(essential=False),
        lambda o: o["sources"][0]["must_cover"][0]["hit_rule"].update(type="most"),
        lambda o: o["sources"][0]["must_cover"][3]["conditional"].pop("applies_if"),
        lambda o: o["sources"][0]["must_cover"][0]["requirements"].append(req("S1-C1.1", True)),
        lambda o: o["sources"].append(copy.deepcopy(o["sources"][1])),
    ],
)
def test_malformed_obligations_are_rejected(mutate):
    obl = obligations()
    mutate(obl)
    with pytest.raises(module.CoverageInputError):
        run(doc(judgment_s1(), judgment_s2()), obl=obl)


def write_cli_fixture(tmp_path, judgments):
    root = tmp_path / "ref"
    root.mkdir()
    obl_bytes = json.dumps(obligations(), ensure_ascii=False).encode()
    (root / "coverage_obligations.json").write_bytes(obl_bytes)
    (root / "MANIFEST.sha256").write_text(f"{sha(obl_bytes)}  coverage_obligations.json\n")
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "s1.md").write_bytes(OUT_S1)
    (out_dir / "s2.md").write_bytes(OUT_S2)
    (out_dir / "outputs.json").write_text(
        json.dumps(
            {
                "producer": PRODUCER,
                "outputs": [
                    {"source_id": "S1", "path": "s1.md"},
                    {"source_id": "S2", "path": "s2.md"},
                ],
            }
        )
    )
    d = doc(*judgments)
    d["coverage_obligations_sha256"] = sha(obl_bytes)
    (tmp_path / "judgments.json").write_text(json.dumps(d))
    return [
        "score_coverage",
        "--root",
        str(root),
        "--outputs",
        str(out_dir / "outputs.json"),
        "--judgments",
        str(tmp_path / "judgments.json"),
    ]


def test_cli_scores_hash_bound_outputs(tmp_path, monkeypatch, capsys):
    argv = write_cli_fixture(tmp_path, [judgment_s1(), judgment_s2()])
    monkeypatch.setattr(module.sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["complete"] is True
    assert rep["aggregate"]["strict"]["sources_passed"] == 2


def test_cli_rejects_output_bytes_that_differ_from_judged_hash(tmp_path, monkeypatch):
    argv = write_cli_fixture(tmp_path, [judgment_s1(), judgment_s2()])
    (tmp_path / "outputs" / "s1.md").write_bytes(OUT_S1 + b"edited after judging")
    monkeypatch.setattr(module.sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code not in (0, None)


def test_cli_require_complete_fails_on_incomplete_judgments(tmp_path, monkeypatch, capsys):
    argv = write_cli_fixture(tmp_path, [judgment_s1()])
    monkeypatch.setattr(module.sys, "argv", [*argv, "--require-complete"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code not in (0, None)
    assert json.loads(capsys.readouterr().out)["complete"] is False


def test_cli_rejects_obligations_not_matching_manifest(tmp_path, monkeypatch):
    argv = write_cli_fixture(tmp_path, [judgment_s1(), judgment_s2()])
    (tmp_path / "ref" / "MANIFEST.sha256").write_text("0" * 64 + "  coverage_obligations.json\n")
    monkeypatch.setattr(module.sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code not in (0, None)
