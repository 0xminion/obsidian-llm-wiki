"""Portable reference-v2 verifier over an invented miniature dataset.

Every source body, claim, quote, id and label here is invented for the test;
nothing is copied from the private benchmark.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "benchmarks/grounding_reference/verify_reference_v2.py"
spec = importlib.util.spec_from_file_location("grounding_reference_verify_v2", PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

S1 = (
    "Alpha team shipped 3 widgets in May. Later, the team said it would ship more. "
    "Widgets are blue.\nThe Beta group tested them."
)
S2 = (
    "\u8fd9\u662f\u4e00\u4e2a\u865a\u6784\u7684\u6765\u6e90\u3002"
    "\u5b83\u6ca1\u6709\u771f\u5b9e\u5185\u5bb9\u3002"
)
LOCATOR_QUOTE = S2[S2.index("\u5b83") :]
BODIES = {"S1": S1, "S2": S2}
GUIDELINES = "\n".join(
    f"- **{r}** invented rule text." for r in ("E1", "E2", "E3", "E4", "C1", "X1", "X2", "U1")
)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def bsha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def span(sid, text, with_text=True):
    body = BODIES[sid]
    start = body.index(text)
    end = start + len(text)
    sp = {
        "source_id": sid,
        "start": start,
        "end": end,
        "byte_start": len(body[:start].encode("utf-8")),
        "byte_end": len(body[:end].encode("utf-8")),
        "sha256": sha(text),
        "chars": end - start,
    }
    if with_text:
        sp["text"] = text
    return sp


def sourcedoc(body):
    return json.dumps({"content": body, "provenance": {"content_sha256": sha(body)}}).encode()


def sources_meta():
    rows = []
    for sid, lang, body, alias in (
        ("S1", "en", S1, None),
        ("S2", "zh-Hans", S2, None),
        ("S3", "en", S1, "S1"),  # byte-identical duplicate, absent from the inputs dir
    ):
        rows.append(
            {
                "source_id": sid,
                "language": lang,
                "chars": len(body),
                "content_sha256": sha(body),
                "private_sourcedoc_file_sha256": None if alias else bsha(sourcedoc(body)),
                "content_identical_to": alias,
            }
        )
    return {"offset_unit": "code_point", "span_hash": "sha256", "sources": rows}


SHIPPED = "Alpha team shipped 3 widgets in May."
LATER = "Later, the team said it would ship more."
BLUE = "Widgets are blue."
BETA = "The Beta group tested them."


def base_case(cid, split, task, label, mutation, pair, evidence, gold, **extra):
    c = {
        "id": cid,
        "key": f"key-{cid}",
        "split": split,
        "task": task,
        "source_id": extra.pop("source_id", "S1"),
        "cited_source_id": extra.pop("cited_source_id", "S1"),
        "source_language": "en",
        "claim_language": "en",
        "claim": f"invented claim for {cid}",
        "label": label,
        "polarity": {"supported": "positive", "unsupported": "hard_negative"}.get(
            label, "uncertain"
        ),
        "mutation": mutation,
        "pair_group": pair,
        "origin": "constructed",
        "evidence": evidence,
        "gold_support_spans": gold,
        "rationale": "invented rationale",
        "adjudication": {"by": "fixture", "method": "invented", "date": "2000-01-01"},
    }
    c.update(extra)
    return c


def quote_extra(quote, occ):
    return {
        "candidate_quote": quote,
        "candidate_quote_sha256": sha(quote),
        "candidate_quote_occurrences": occ,
    }


def v1_cases():
    blue = span("S1", BLUE)
    return {
        "calibration": [
            base_case(
                "GR-CAL-901",
                "calibration",
                "claim_support",
                "supported",
                None,
                "p1",
                [span("S1", SHIPPED)],
                [],
            ),
            base_case(
                "GR-CAL-902",
                "calibration",
                "claim_support",
                "unsupported",
                "number",
                "p1",
                [span("S1", SHIPPED)],
                [span("S1", SHIPPED)],
            ),
        ],
        "heldout": [
            base_case(
                "GR-HO-901",
                "heldout",
                "quote_fidelity",
                "supported",
                None,
                "p2",
                [blue],
                [],
                **quote_extra(BLUE, [{"start": blue["start"], "end": blue["end"]}]),
            ),
            base_case(
                "GR-HO-902",
                "heldout",
                "quote_fidelity",
                "unsupported",
                "case_alteration",
                "p2",
                [],
                [blue],
                **quote_extra(BLUE.upper(), []),
            ),
            base_case(
                "GR-HO-903",
                "heldout",
                "claim_support",
                "uncertain",
                "actor",
                "p3",
                [span("S1", BETA)],
                [],
            ),
            base_case(
                "GR-HO-904",
                "heldout",
                "claim_support",
                "unsupported",
                "wrong_span",
                "p4",
                [blue],
                [span("S1", SHIPPED)],
                notes="supported_elsewhere=true",
            ),
            base_case(
                "GR-HO-905",
                "heldout",
                "citation_locator",
                "unsupported",
                "cross_source_id",
                "p5",
                [],
                [span("S2", LOCATOR_QUOTE)],
                source_id="S2",
                **quote_extra(LOCATOR_QUOTE, []),
            ),
        ],
    }


ROLES = {
    "GR-CAL-902": ("label_basis", "contradicts"),
    "GR-HO-902": ("raw_original", None),
    "GR-HO-903": ("label_basis", "does_not_entail"),
    "GR-HO-904": ("supported_elsewhere", None),
    "GR-HO-905": ("true_location_other_source", None),
}
REFS = {
    "GR-CAL-901": ["E1", "E3"],
    "GR-CAL-902": ["C1"],
    "GR-HO-901": ["X1"],
    "GR-HO-902": ["X1"],
    "GR-HO-903": ["E2"],
    "GR-HO-904": ["E4"],
    "GR-HO-905": ["X2"],
}
CLAIM_FIELD = {
    "claim_support": "claim",
    "quote_fidelity": "mutation_description",
    "citation_locator": "ungraded_context",
}


def v2_from_v1(v1):
    out = {}
    for split, rows in v1.items():
        out[split] = []
        for a in rows:
            c = copy.deepcopy(a)
            if c["id"] == "GR-HO-903":
                c.update(
                    label="unsupported",
                    polarity="hard_negative",
                    rationale="invented new rationale",
                    gold_support_spans=[span("S1", BETA)],
                )
                c["change_refs"] = ["CH-001", "CH-002", "CH-003", "CH-004"]
                c["adjudication_v2"] = {"by": "f", "decision_by": "p", "method": "m", "date": "d"}
            else:
                c["change_refs"] = []
            role, rel = ROLES.get(c["id"], (None, None))
            c["reference_spans"] = [
                {**g, "role": role, "relation": rel, "credits_citation": False}
                for g in c["gold_support_spans"]
            ]
            quote = "candidate_quote" in c
            valid = sorted(
                s
                for s, b in (("S1", S1), ("S2", S2), ("S3", S1))
                if quote and c["candidate_quote"] in b
            )
            c["flags"] = {
                "supported_elsewhere": True if c["mutation"] == "wrong_span" else None,
                "multi_occurrence": len(c["candidate_quote_occurrences"]) if quote else None,
                "valid_source_ids": valid if quote else None,
                "primary_metric": c["label"] != "uncertain",
                "claim_graded": c["task"] == "claim_support",
                "claim_field": CLAIM_FIELD[c["task"]],
                "notes_residual": None,
            }
            c["context_spans"] = []
            if c["id"] == "GR-CAL-901":
                c["context_spans"] = [{**span("S1", LATER), "rule": "E3", "use": "referent"}]
            c["guideline_refs"] = list(REFS[c["id"]])
            out[split].append(c)
    return out


def item_v1(iid, texts):
    return {
        "id": iid,
        "point": f"point {iid}",
        "gloss_en": f"gloss {iid}",
        "evidence": [span("S1", t, with_text=False) for t in texts],
    }


def cov_v1():
    return {
        "sources": [
            {
                "source_id": "S1",
                "content_sha256": sha(S1),
                "summary_language_required": "en",
                "must_cover": [
                    item_v1("S1-C1", [SHIPPED]),
                    item_v1("S1-C2", [LATER, BLUE]),
                    item_v1("S1-C3", [BETA]),
                ],
                "must_preserve_qualifiers": [
                    {
                        "id": "S1-Q1",
                        "requirement": "keep the hedge",
                        "evidence": [span("S1", LATER, with_text=False)],
                    }
                ],
                "must_not_assert": [
                    {
                        "id": "S1-N1",
                        "assertion": "invented falsehood",
                        "why": "w",
                        "origin": "constructed",
                        "contrary_evidence": [span("S1", SHIPPED, with_text=False)],
                    }
                ],
                "out_of_scope": [],
                "notes": None,
            }
        ]
    }


def requirement(rid, text, essential, group=None):
    r = {"id": rid, "text": f"req {rid}", "essential": essential, "evidence": [span("S1", text)]}
    if group:
        r["group"] = group
    return r


def cov_v2():
    v1 = cov_v1()["sources"][0]
    items = []
    for it in v1["must_cover"]:
        it = copy.deepcopy(it)
        it["evidence"] = [span("S1", BODIES["S1"][e["start"] : e["end"]]) for e in it["evidence"]]
        it["audit"] = {"reviewer_flagged": False, "evidence_changed": False}
        items.append(it)
    items[0]["hit_rule"] = {"type": "all_essential", "rule": "CV2"}
    items[0]["requirements"] = [requirement("S1-C1.1", SHIPPED, True)]
    items[1]["hit_rule"] = {
        "type": "at_least_k_groups",
        "k": 1,
        "groups": ["g1", "g2"],
        "rule": "CV3",
        "why": "separate sections",
    }
    items[1]["requirements"] = [
        requirement("S1-C2.1", LATER, True, "g1"),
        requirement("S1-C2.2", BLUE, True, "g2"),
        requirement("S1-C2.3", BLUE, False, "g2"),
    ]
    items[2]["hit_rule"] = {"type": "all_essential", "rule": "CV2"}
    items[2]["requirements"] = [requirement("S1-C3.1", BETA, True)]
    items[2]["conditional"] = {
        "applies_if": "the output states an invented figure",
        "if_not_applicable": "excluded from n",
        "rule": "CV4",
    }
    src = copy.deepcopy(v1)
    src["must_cover"] = items
    src["must_preserve_qualifiers"][0]["evidence"] = [span("S1", LATER)]
    src["must_not_assert"][0]["contrary_evidence"] = [span("S1", SHIPPED)]
    return {
        "version": "fixture-v2",
        "derived_from": {"version": "fixture-v1"},
        "visibility": "private",
        "scoring_rule": {"coverage": "invented"},
        "sources": [src],
    }


def changes(v1, v2):
    a = next(c for c in v1["heldout"] if c["id"] == "GR-HO-903")
    b = next(c for c in v2["heldout"] if c["id"] == "GR-HO-903")
    rows = []
    for n, field in enumerate(("label", "polarity", "rationale", "gold_support_spans"), 1):
        rows.append(
            {
                "change_id": f"CH-00{n}",
                "file": "cases.heldout.jsonl",
                "target": "GR-HO-903",
                "field": field,
                "original": a[field],
                "new": b[field],
                "rationale": "invented",
                "decided_by": "parent",
                "source_evidence": [{"source_id": "S1", "start": 0, "end": 5}],
            }
        )
    rows.append(
        {
            "change_id": "CH-005",
            "file": "*",
            "target": "*",
            "field": "scoring_rule",
            "original": None,
            "new": "x",
            "rationale": "r",
            "decided_by": "curator",
            "source_evidence": [],
        }
    )
    return rows


def dataset():
    v1 = v1_cases()
    v2 = v2_from_v1(v1)
    return {
        "v1": v1,
        "v2": v2,
        "cov1": cov_v1(),
        "cov2": cov_v2(),
        "changes": changes(v1, v2),
        "sources": sources_meta(),
    }


def jsonl(rows):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


def write(tmp_path, d, after=None):
    base, root, inputs = tmp_path / "v1", tmp_path / "v2", tmp_path / "inputs"
    for p in (base, root, inputs):
        p.mkdir()
    for sid, body in BODIES.items():
        (inputs / sid).mkdir()
        (inputs / sid / "source.json").write_bytes(sourcedoc(body))
    files1 = {
        "sources.json": json.dumps(d["sources"]),
        "coverage_obligations.json": json.dumps(d["cov1"], ensure_ascii=False),
    }
    for split, rows in d["v1"].items():
        files1[f"cases.{split}.jsonl"] = jsonl(rows)
    for name, text in files1.items():
        (base / name).write_text(text, encoding="utf-8")
    (base / "MANIFEST.sha256").write_text(
        "# fixture v1\n" + "".join(f"{sha(t)}  {n}\n" for n, t in files1.items())
    )
    files2 = {
        "sources.json": files1["sources.json"],
        "coverage_obligations.json": json.dumps(d["cov2"], ensure_ascii=False),
        "CHANGES.jsonl": jsonl(d["changes"]),
        "GUIDELINES.md": GUIDELINES,
        "README.md": "invented readme\n",
        "VERIFICATION.json": "{}\n",
    }
    for split, rows in d["v2"].items():
        raw1 = files1[f"cases.{split}.jsonl"].splitlines()
        for line, c in zip(raw1, rows, strict=False):
            c.setdefault("v1_record_sha256", sha(line))
        files2[f"cases.{split}.jsonl"] = jsonl(rows)
    for name, text in files2.items():
        (root / name).write_text(text, encoding="utf-8")
    if after:
        after(base, root, inputs)
    manifest_files = sorted(files2)
    (root / "MANIFEST.sha256").write_text(
        "# fixture v2\n"
        + "".join(f"{bsha((root / n).read_bytes())}  {n}\n" for n in manifest_files)
    )
    return base, root, inputs


def verify(tmp_path, d=None, after=None, post=None, **kw):
    base, root, inputs = write(tmp_path, d or dataset(), after)
    if post:
        post(base, root, inputs)
    return module.verify(root=root, baseline=base, sources_dir=inputs, **kw)


def test_clean_fixture_verifies(tmp_path):
    res = verify(tmp_path)
    assert res["errors"] == []
    assert res["checks"] > 100
    assert res["counts"]["cases"] == 7
    assert res["counts"]["coverage"]["essential_requirements"] == 4
    assert res["v1_unchanged"] is True


def test_cli_requires_explicit_root_baseline_and_sources(tmp_path, monkeypatch):
    base, root, inputs = write(tmp_path, dataset())
    for argv in (
        ["--root", str(root), "--baseline", str(base)],
        ["--root", str(root), "--sources", str(inputs)],
        ["--baseline", str(base), "--sources", str(inputs)],
    ):
        monkeypatch.setattr(module.sys, "argv", ["verify", *argv])
        with pytest.raises(SystemExit) as exc:
            module.main()
        assert exc.value.code != 0


def test_cli_exit_status(tmp_path, monkeypatch, capsys):
    base, root, inputs = write(tmp_path, dataset())
    argv = ["verify", "--root", str(root), "--baseline", str(base), "--sources", str(inputs)]
    monkeypatch.setattr(module.sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    (root / "README.md").write_text("edited")
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1


def manifest_pin(base):
    return bsha((base / "MANIFEST.sha256").read_bytes())


def test_pins_accept_matching_and_reject_other_hashes(tmp_path):
    base, root, inputs = write(tmp_path, dataset())
    pin = tmp_path / "pin.sha256"
    pin.write_bytes((base / "MANIFEST.sha256").read_bytes())
    good = module.verify(
        root=root,
        baseline=base,
        sources_dir=inputs,
        baseline_pin=pin,
        expect_manifest_sha256=bsha((root / "MANIFEST.sha256").read_bytes()),
    )
    assert good["errors"] == []
    pin.write_text("0" * 64 + "  sources.json\n")
    bad = module.verify(
        root=root,
        baseline=base,
        sources_dir=inputs,
        baseline_pin=pin,
        expect_manifest_sha256="0" * 64,
    )
    assert any("pin" in e for e in bad["errors"])
    assert any("expected MANIFEST" in e for e in bad["errors"])


def edit(path_fn, text=None, fn=None):
    def post(base, root, inputs):
        p = path_fn(base, root, inputs)
        p.write_bytes(fn(p.read_bytes()) if fn else text.encode())

    return post


@pytest.mark.parametrize(
    "post",
    [
        edit(lambda b, r, i: r / "README.md", "tampered after manifest"),
        edit(lambda b, r, i: b / "cases.heldout.jsonl", fn=lambda x: x + b"\n"),
        edit(lambda b, r, i: r / "UNLISTED.txt", "extra"),
        edit(lambda b, r, i: b / "UNLISTED.txt", "extra"),
        edit(lambda b, r, i: i / "S1" / "source.json", fn=lambda x: x.replace(b"Alpha", b"Alphb")),
        lambda b, r, i: (i / "S2" / "source.json").unlink(),
        edit(
            lambda b, r, i: r / "MANIFEST.sha256",
            fn=lambda x: b"\n".join(ln for ln in x.split(b"\n") if b"GUIDELINES" not in ln),
        ),
    ],
)
def test_hash_and_file_set_tampering_is_detected(tmp_path, post):
    assert verify(tmp_path, post=post)["errors"]


def mut(fn):
    def apply():
        d = dataset()
        fn(d)
        return d

    return apply


def case(d, cid):
    return next(c for rows in d["v2"].values() for c in rows if c["id"] == cid)


def v1case(d, cid):
    return next(c for rows in d["v1"].values() for c in rows if c["id"] == cid)


def item(d, iid):
    return next(i for i in d["cov2"]["sources"][0]["must_cover"] if i["id"] == iid)


CASE_MUTATIONS = {
    "v1 claim edited": lambda d: case(d, "GR-CAL-901").update(claim="edited"),
    "v1 field dropped": lambda d: case(d, "GR-CAL-901").pop("origin"),
    "unexpected field": lambda d: case(d, "GR-CAL-901").update(extra=1),
    "record hash": lambda d: case(d, "GR-CAL-901").update(v1_record_sha256="0" * 64),
    "ledger new mismatch": lambda d: d["changes"][0].update(new="supported"),
    "ledger original mismatch": lambda d: d["changes"][0].update(original="supported"),
    "ledger missing": lambda d: d["changes"].pop(3),
    "dangling change ref": lambda d: case(d, "GR-HO-903")["change_refs"].append("CH-999"),
    "change ref omitted": lambda d: case(d, "GR-HO-903")["change_refs"].pop(),
    "adjudication_v2 missing": lambda d: case(d, "GR-HO-903").pop("adjudication_v2"),
    "duplicate change id": lambda d: d["changes"].append(dict(d["changes"][4])),
    "ledger offsets": lambda d: d["changes"][0]["source_evidence"].append(
        {"source_id": "S1", "start": 0, "end": 10**6}
    ),
    "primary flag": lambda d: case(d, "GR-CAL-901")["flags"].update(primary_metric=False),
    "primary flag type": lambda d: case(d, "GR-CAL-901")["flags"].update(primary_metric=1),
    "claim_graded": lambda d: case(d, "GR-HO-901")["flags"].update(claim_graded=True),
    "claim_field": lambda d: case(d, "GR-HO-901")["flags"].update(claim_field="claim"),
    "supported_elsewhere": lambda d: case(d, "GR-HO-904")["flags"].update(supported_elsewhere=None),
    "supported_elsewhere stray": lambda d: case(d, "GR-CAL-902")["flags"].update(
        supported_elsewhere=False
    ),
    "multi_occurrence": lambda d: case(d, "GR-HO-901")["flags"].update(multi_occurrence=2),
    "multi_occurrence bool": lambda d: case(d, "GR-HO-901")["flags"].update(multi_occurrence=True),
    "valid ids omit alias": lambda d: case(d, "GR-HO-901")["flags"].update(valid_source_ids=["S1"]),
    "valid ids on claim": lambda d: case(d, "GR-CAL-901")["flags"].update(valid_source_ids=[]),
    "notes_residual type": lambda d: case(d, "GR-CAL-901")["flags"].update(notes_residual=1),
    "extra flag": lambda d: case(d, "GR-CAL-901")["flags"].update(extra=None),
    "missing flag": lambda d: case(d, "GR-CAL-901")["flags"].pop("claim_field"),
    "role for task": lambda d: case(d, "GR-HO-902")["reference_spans"][0].update(
        role="label_basis"
    ),
    "role unknown": lambda d: case(d, "GR-CAL-902")["reference_spans"][0].update(role="x"),
    "credits citation": lambda d: case(d, "GR-CAL-902")["reference_spans"][0].update(
        credits_citation=True
    ),
    "credits citation falsy": lambda d: case(d, "GR-CAL-902")["reference_spans"][0].update(
        credits_citation=0
    ),
    "relation contradicts on actor": lambda d: case(d, "GR-HO-903")["reference_spans"][0].update(
        relation="contradicts"
    ),
    "relation missing on number": lambda d: case(d, "GR-CAL-902")["reference_spans"][0].update(
        relation=None
    ),
    "relation unknown": lambda d: case(d, "GR-HO-904")["reference_spans"][0].update(
        relation="refutes"
    ),
    "reference != gold": lambda d: case(d, "GR-CAL-902")["reference_spans"].clear(),
    "true location wrong source": lambda d: case(d, "GR-HO-905")["reference_spans"][0].update(
        source_id="S1"
    ),
    "context rule": lambda d: case(d, "GR-CAL-901")["context_spans"][0].update(rule="E4"),
    "context use": lambda d: case(d, "GR-CAL-901")["context_spans"][0].update(use=""),
    "context hash": lambda d: case(d, "GR-CAL-901")["context_spans"][0].update(sha256="0" * 64),
    "context duplicates evidence": lambda d: case(d, "GR-CAL-901")["context_spans"].append(
        {**span("S1", SHIPPED), "rule": "E3", "use": "x"}
    ),
    "context without E3 ref": lambda d: case(d, "GR-CAL-901")["guideline_refs"].remove("E3"),
    "E3 ref without context": lambda d: case(d, "GR-CAL-902")["guideline_refs"].append("E3"),
    "context type": lambda d: case(d, "GR-CAL-902").update(context_spans=None),
    "undefined guideline": lambda d: case(d, "GR-CAL-902")["guideline_refs"].append("Z9"),
    "no guideline": lambda d: case(d, "GR-CAL-902").update(guideline_refs=[]),
    "evidence offset": lambda d: case(d, "GR-CAL-901")["evidence"][0].update(byte_end=1),
    "quote hash": lambda d: case(d, "GR-HO-901").update(candidate_quote_sha256="0" * 64),
    "pair straddles split": lambda d: case(d, "GR-HO-905").update(pair_group="p1"),
}


def flip_supported_to_unsupported(d):
    """Even a fully ledgered supported -> unsupported flip is forbidden."""
    a = v1case(d, "GR-CAL-901")
    b = case(d, "GR-CAL-901")
    b.update(label="unsupported", polarity="hard_negative")
    b["flags"]["primary_metric"] = True
    b["change_refs"] = ["CH-101", "CH-102"]
    b["adjudication_v2"] = {"by": "f", "decision_by": "p", "method": "m", "date": "d"}
    for cid, f in (("CH-101", "label"), ("CH-102", "polarity")):
        d["changes"].append(
            {
                "change_id": cid,
                "file": "cases.calibration.jsonl",
                "target": "GR-CAL-901",
                "field": f,
                "original": a[f],
                "new": b[f],
                "rationale": "r",
                "decided_by": "parent",
                "source_evidence": [],
            }
        )


CASE_MUTATIONS["supported->unsupported flip"] = flip_supported_to_unsupported


def ledgered_claim_edit(d):
    """Claims are outside the changeable field set even with a ledger entry."""
    a = v1case(d, "GR-HO-903")
    b = case(d, "GR-HO-903")
    b["claim"] = "edited"
    b["change_refs"].append("CH-201")
    d["changes"].append(
        {
            "change_id": "CH-201",
            "file": "cases.heldout.jsonl",
            "target": "GR-HO-903",
            "field": "claim",
            "original": a["claim"],
            "new": "edited",
            "rationale": "r",
            "decided_by": "parent",
            "source_evidence": [],
        }
    )


CASE_MUTATIONS["ledgered claim edit"] = ledgered_claim_edit


# Mutations that also trip generic invariants must still report their own rule.
EXPECTED_MESSAGE = {
    "supported->unsupported flip": "flip",
    "ledgered claim edit": "GR-HO-903.claim",
    "pair straddles split": "crosses splits",
    "true location wrong source": "true_location_other_source",
    "ledger missing": "GR-HO-903.gold_support_spans",
}


@pytest.mark.parametrize("name", sorted(CASE_MUTATIONS))
def test_case_level_violations_are_detected(tmp_path, name):
    res = verify(tmp_path, d=mut(CASE_MUTATIONS[name])())
    assert res["errors"], name
    if name in EXPECTED_MESSAGE:
        assert any(EXPECTED_MESSAGE[name] in e for e in res["errors"]), res["errors"]


COVERAGE_MUTATIONS = {
    "point changed": lambda d: item(d, "S1-C1").update(point="edited"),
    "item ids changed": lambda d: d["cov2"]["sources"][0]["must_cover"].pop(),
    "req evidence outside item": lambda d: item(d, "S1-C1")["requirements"][0].update(
        evidence=[span("S1", BETA)]
    ),
    "req without evidence": lambda d: item(d, "S1-C1")["requirements"][0].update(evidence=[]),
    "req id prefix": lambda d: item(d, "S1-C1")["requirements"][0].update(id="S1-C9.1"),
    "req duplicate id": lambda d: item(d, "S1-C2")["requirements"][1].update(id="S1-C2.1"),
    "essential type": lambda d: item(d, "S1-C1")["requirements"][0].update(essential="yes"),
    "no essential": lambda d: item(d, "S1-C1")["requirements"][0].update(essential=False),
    "group on non-bundle": lambda d: item(d, "S1-C1")["requirements"][0].update(group="g1"),
    "group lacks essential": lambda d: item(d, "S1-C2")["requirements"][0].update(essential=False),
    "k out of range": lambda d: item(d, "S1-C2")["hit_rule"].update(k=3),
    "k type": lambda d: item(d, "S1-C2")["hit_rule"].update(k=True),
    "groups mismatch": lambda d: item(d, "S1-C2")["hit_rule"].update(groups=["g1"]),
    "hit rule type": lambda d: item(d, "S1-C1")["hit_rule"].update(type="majority"),
    "conditional applies_if": lambda d: item(d, "S1-C3")["conditional"].pop("applies_if"),
    "conditional rule": lambda d: item(d, "S1-C3")["conditional"].update(rule="CV9"),
    "req evidence hash": lambda d: item(d, "S1-C1")["requirements"][0]["evidence"][0].update(
        sha256="0" * 64
    ),
    "unlogged evidence change": lambda d: item(d, "S1-C3").update(evidence=[span("S1", BLUE)]),
    "audit flag wrong": lambda d: item(d, "S1-C1")["audit"].update(evidence_changed=True),
    "qualifier changed": lambda d: d["cov2"]["sources"][0]["must_preserve_qualifiers"][0].update(
        requirement="edited"
    ),
    "unlogged qualifier evidence": lambda d: d["cov2"]["sources"][0]["must_preserve_qualifiers"][
        0
    ].update(evidence=[span("S1", BLUE)]),
    "must_not changed": lambda d: d["cov2"]["sources"][0]["must_not_assert"][0].update(why="e"),
    "must_not contrary changed": lambda d: d["cov2"]["sources"][0]["must_not_assert"][0].update(
        contrary_evidence=[span("S1", BLUE)]
    ),
    "language changed": lambda d: d["cov2"]["sources"][0].update(
        summary_language_required="zh-Hans"
    ),
    "content hash": lambda d: d["cov2"]["sources"][0].update(content_sha256="0" * 64),
}


@pytest.mark.parametrize("name", sorted(COVERAGE_MUTATIONS))
def test_coverage_structure_violations_are_detected(tmp_path, name):
    res = verify(tmp_path, d=mut(COVERAGE_MUTATIONS[name])())
    assert res["errors"], name
    if name == "unlogged evidence change":
        assert any("unlogged evidence change" in e for e in res["errors"])


def test_v2_sources_must_stay_byte_identical_even_with_a_consistent_manifest(tmp_path):
    def after(base, root, inputs):
        p = root / "sources.json"
        p.write_text(p.read_text() + " ")  # v2 MANIFEST is written after this edit

    res = verify(tmp_path, after=after)
    assert any("byte-identical" in e for e in res["errors"])
    assert not any("v2: hash mismatch" in e for e in res["errors"])


def test_baseline_pin_mismatch_marks_v1_changed(tmp_path):
    base, root, inputs = write(tmp_path, dataset())
    pin = tmp_path / "pin.sha256"
    pin.write_text("0" * 64 + "  sources.json\n")
    res = module.verify(root=root, baseline=base, sources_dir=inputs, baseline_pin=pin)
    assert res["v1_unchanged"] is False


def test_logged_evidence_change_with_audit_flag_is_accepted(tmp_path):
    d = dataset()
    it = item(d, "S1-C3")
    it["evidence"].append(span("S1", BLUE))
    it["audit"]["evidence_changed"] = True
    d["changes"].append(
        {
            "change_id": "CH-301",
            "file": "coverage_obligations.json",
            "target": "S1-C3",
            "field": "must_cover.evidence",
            "original": None,
            "new": None,
            "rationale": "r",
            "decided_by": "curator",
            "source_evidence": [],
        }
    )
    assert verify(tmp_path, d=d)["errors"] == []


def locators(spans):
    return [{"source_id": s["source_id"], "start": s["start"], "end": s["end"]} for s in spans]


def test_ledger_may_record_span_lists_as_exact_locators(tmp_path):
    d = dataset()
    row = next(c for c in d["changes"] if c["field"] == "gold_support_spans")
    row["new"] = locators(row["new"])
    row["original"] = locators(row["original"])
    assert verify(tmp_path, d=d)["errors"] == []


@pytest.mark.parametrize(
    "bad",
    [
        lambda spans: [{**s, "end": s["end"] - 1} for s in locators(spans)],
        lambda spans: locators(spans)[:0],
        lambda spans: [{**s, "extra": 1} for s in locators(spans)],
    ],
)
def test_ledger_locators_must_match_the_changed_spans(tmp_path, bad):
    d = dataset()
    row = next(c for c in d["changes"] if c["field"] == "gold_support_spans")
    row["new"] = bad(row["new"])
    res = verify(tmp_path, d=d)
    assert any("GR-HO-903.gold_support_spans" in e for e in res["errors"])
