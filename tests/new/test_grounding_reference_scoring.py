"""The reference scorer must not reward abstention or accept manipulable gold.

All fixtures are invented; no benchmark case text, id or label is reproduced.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "benchmarks/grounding_reference/score_predictions.py"
spec = importlib.util.spec_from_file_location("grounding_reference_score", PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cases():
    return [
        {
            "id": "GR-CAL-901",
            "split": "calibration",
            "label": "supported",
            "task": "claim_support",
            "source_language": "en",
            "pair_group": "pair",
            "mutation": None,
        },
        {
            "id": "GR-CAL-902",
            "split": "calibration",
            "label": "unsupported",
            "task": "claim_support",
            "source_language": "en",
            "pair_group": "pair",
            "mutation": "actor",
        },
    ]


def v2_flags(case, **overrides):
    task = case["task"]
    quote = task != "claim_support"
    flags = {
        "supported_elsewhere": True if case["mutation"] == "wrong_span" else None,
        "multi_occurrence": (1 if case["label"] == "supported" else 0) if quote else None,
        "valid_source_ids": (["S1"] if case["label"] == "supported" else []) if quote else None,
        "primary_metric": case["label"] != "uncertain",
        "claim_graded": not quote,
        "claim_field": "claim" if not quote else "mutation_description",
        "notes_residual": None,
    }
    flags.update(overrides)
    return flags


def v2_cases():
    out = cases()
    out.append(
        {
            "id": "GR-CAL-903",
            "split": "calibration",
            "label": "uncertain",
            "task": "claim_support",
            "source_language": "zh-Hans",
            "pair_group": "solo",
            "mutation": None,
        }
    )
    for c in out:
        c["flags"] = v2_flags(c)
        c["context_spans"] = []
    return out


def test_missing_predictions_never_count_as_correct():
    result = module.score(cases(), {})["all"]
    assert result["accuracy"] == 0
    assert result["supported_recall"] == 0
    assert result["unsupported_recall"] == 0
    assert result["missing_predictions"] == 2


def test_partial_predictions_keep_all_gold_cases_in_denominator():
    result = module.score(cases(), {"GR-CAL-902": "unsupported"})["all"]
    assert result["accuracy"] == 0.5
    assert result["supported_recall"] == 0
    assert result["unsupported_recall"] == 1


def test_v2_primary_flag_excludes_only_uncertain_gold():
    result = module.score(v2_cases(), {"GR-CAL-901": "supported", "GR-CAL-902": "unsupported"})
    assert result["all"]["n"] == 2
    assert result["all"]["accuracy"] == 1
    assert list(result["uncertain_excluded"]) == ["GR-CAL-903"]


def test_v2_flags_are_reported_as_separate_slices():
    gold = v2_cases()
    gold[1]["mutation"] = "wrong_span"
    gold[1]["flags"] = v2_flags(gold[1])
    result = module.score(gold, {"GR-CAL-902": "supported"})
    assert result["flag:supported_elsewhere"]["n"] == 1
    assert result["flag:supported_elsewhere"]["unsupported_accepted"] == 1


@pytest.mark.parametrize(
    "mutate",
    [
        # Excluding a scorable case shrinks the denominator.
        lambda g: g[0]["flags"].update(primary_metric=False),
        # Including uncertain gold adds an unscorable case.
        lambda g: g[2]["flags"].update(primary_metric=True),
        lambda g: g[0]["flags"].update(primary_metric="yes"),
        lambda g: g[0]["flags"].pop("primary_metric"),
        lambda g: g[0].pop("flags"),  # mixed v1/v2 records
        lambda g: g[0]["flags"].update(claim_graded=False),
        lambda g: g[0]["flags"].update(supported_elsewhere=True),
        lambda g: g[0]["flags"].update(multi_occurrence=2),
        lambda g: g[0]["flags"].update(unexpected=1),
        lambda g: g[0].update(context_spans="none"),
    ],
)
def test_inconsistent_v2_flags_are_rejected(mutate):
    gold = v2_cases()
    mutate(gold)
    with pytest.raises(module.GoldError):
        module.score(gold, {})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda g: g.append(copy.deepcopy(g[0])),  # duplicate identity inflates n
        lambda g: g[1].update(id=g[0]["id"]),
        lambda g: g[0].update(id=""),
        lambda g: g[0].update(id=7),
        lambda g: g[0].pop("id"),
        lambda g: g[0].update(id="GR-HO-901"),  # prefix of another split
        lambda g: g[0].update(split="heldout"),
        lambda g: g[0].pop("split"),
        lambda g: g[0].update(label="Supported"),
        lambda g: g[0].update(label=None),
        lambda g: g[0].update(task="other"),
        lambda g: g[0].update(polarity="hard_negative"),
        lambda g: g[0].update(pair_group=None),
        lambda g: g[0].update(mutation="negation"),  # supported gold cannot be mutated
        lambda g: g[1].update(mutation=None),
        lambda g: g.append("not a record"),
    ],
)
def test_malformed_or_duplicate_gold_identities_are_rejected(mutate):
    gold = cases()
    mutate(gold)
    with pytest.raises(module.GoldError):
        module.score(gold, {})


def run_cli(tmp_path, monkeypatch, gold, predictions, *extra, other_split=None):
    (tmp_path / "cases.calibration.jsonl").write_text(
        "\n".join(json.dumps(c) for c in gold), encoding="utf-8"
    )
    if other_split is not None:
        (tmp_path / "cases.heldout.jsonl").write_text(
            "\n".join(json.dumps(c) for c in other_split), encoding="utf-8"
        )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("\n".join(json.dumps(p) for p in predictions), encoding="utf-8")
    argv = ["score", str(predictions_path), "--root", str(tmp_path), "--split", "calibration"]
    monkeypatch.setattr(module.sys, "argv", [*argv, *extra])
    with pytest.raises(SystemExit) as exc:
        module.main()
    return exc.value.code


@pytest.mark.parametrize(
    "predictions",
    [
        [{"id": "GR-CAL-901", "label": "supported"}, {"id": "GR-CAL-901", "label": "unsupported"}],
        [{"id": "not-offered", "label": "supported"}],
        [{"id": "GR-CAL-901", "label": ["supported"]}],
        ["GR-CAL-901"],
    ],
)
def test_cli_rejects_duplicate_unknown_or_malformed_predictions(tmp_path, monkeypatch, predictions):
    assert run_cli(tmp_path, monkeypatch, cases(), predictions) != 0


def test_cli_rejects_malformed_gold(tmp_path, monkeypatch):
    gold = cases()
    gold.append(copy.deepcopy(gold[0]))
    assert run_cli(tmp_path, monkeypatch, gold, []) != 0


def test_cli_rejects_identity_shared_across_split_files(tmp_path, monkeypatch):
    other = copy.deepcopy(cases()[:1])
    other[0]["split"] = "heldout"  # same id appears in the other split file
    assert run_cli(tmp_path, monkeypatch, cases(), [], other_split=other) != 0


def test_cli_verify_manifest_rejects_tampered_gold(tmp_path, monkeypatch, capsys):
    gold = cases()
    assert run_cli(tmp_path, monkeypatch, gold, []) == 0  # write the files once
    capsys.readouterr()
    digest = module.file_sha256(tmp_path / "cases.calibration.jsonl")
    manifest = f"{digest}  cases.calibration.jsonl\n"
    (tmp_path / "MANIFEST.sha256").write_text(manifest)
    assert run_cli(tmp_path, monkeypatch, gold, [], "--verify-manifest") == 0
    assert json.loads(capsys.readouterr().out)["all"]["n"] == 2
    gold[0]["label"] = "unsupported"
    gold[0]["mutation"] = "actor"
    # run_cli rewrites the gold file; the recorded manifest still names the old bytes.
    assert run_cli(tmp_path, monkeypatch, gold, [], "--verify-manifest") != 0


def test_cli_verify_manifest_requires_a_manifest(tmp_path, monkeypatch):
    assert run_cli(tmp_path, monkeypatch, cases(), [], "--verify-manifest") != 0


def test_cli_scores_valid_gold(tmp_path, monkeypatch, capsys):
    preds = [{"id": "GR-CAL-901", "label": "supported"}, {"id": "GR-CAL-902", "label": None}]
    assert run_cli(tmp_path, monkeypatch, cases(), preds) == 0
    result = json.loads(capsys.readouterr().out)["all"]
    assert result["accuracy"] == 0.5 and result["missing_predictions"] == 1
