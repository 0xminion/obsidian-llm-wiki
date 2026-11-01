#!/usr/bin/env python3
"""Aggregate explicit coverage judgments against reference coverage obligations.

This scorer does NOT judge anything and does NOT establish that a summary is
correct. It aggregates per-requirement judgments that an independent
adjudicator (human or agent, not the producer) has already made against the
private captures, after binding each judgment to the exact output bytes it was
made on (SHA-256). Judgment quality is the adjudicator's responsibility.

Two rubrics are reported side by side and never merged:

* ``frozen_v2`` - the frozen reference-v2 hit rules. ``all_essential`` items
  are hit when every essential requirement is hit; ``at_least_k_groups``
  bundles are hit when at least k groups have all their essential
  requirements hit.
* ``strict`` - every essential requirement of an item must be hit, across
  every bundle group. The frozen k is ignored.

Both use the same source threshold (hits >= ceil(0.75 * n_applicable)) and
the same qualifier, must-not and language gates. ``strict.all_items_hit``
additionally reports whether every applicable item was strictly hit.

Missing or ``"unknown"`` judgments never receive credit. They keep their
item/requirement in every denominator and mark the source ``incomplete``;
an incomplete source's pass is ``null`` unless it fails even if every unknown
were resolved favourably. A conditional item is excluded from n only when the
judgment says ``"not_applicable"`` explicitly; a missing applicability
judgment is never treated as not applicable. Every expected source stays in
the aggregate denominators; a source without an output fails.

Inputs (all JSON):
  --root DIR          coverage_obligations.json (+ MANIFEST.sha256, checked)
  --outputs FILE      {"producer": ID, "outputs": [{"source_id", "path"}]}
                      paths are relative to FILE; each file's bytes are hashed
  --judgments FILE    {"reference_version", "coverage_obligations_sha256",
                       "judgments": [one object per judged output]}

Judgment object:
  {"source_id", "output_sha256",
   "adjudication": {"judge_id": str (!= producer), "independent": true, ...},
   "language": {"observed": "en" | "zh-Hans" | ... | "unknown"},
   "applicability": {"<conditional item id>": "applicable"|"not_applicable"|"unknown"},
   "requirements": {"<requirement id>": "hit"|"miss"|"unknown"|"not_applicable"},
   "qualifiers": {"<qualifier id>": "respected"|"omitted"|"violated"|"unknown"},
   "must_not": {"<must_not id>": "absent"|"violated"|"unknown"}}

Exit status: 0 scored (check "complete"), 2 invalid input, 3 incomplete with
--require-complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from typing import NoReturn

DISCLAIMER = (
    "This report aggregates supplied independent judgments bound to output hashes; "
    "it does not itself establish that any output is true, faithful or complete. "
    "Missing or unknown judgments receive no credit."
)
REQ_VALUES = {"hit", "miss", "unknown", "not_applicable"}
APPLICABILITY_VALUES = {"applicable", "not_applicable", "unknown"}
QUALIFIER_VALUES = {"respected", "omitted", "violated", "unknown"}
MUST_NOT_VALUES = {"absent", "violated", "unknown"}
JUDGMENT_KEYS = {
    "source_id",
    "output_sha256",
    "adjudication",
    "language",
    "applicability",
    "requirements",
    "qualifiers",
    "must_not",
}
SHA_HEX = set("0123456789abcdef")


class CoverageInputError(ValueError):
    """Inputs are malformed, unbound or not independent; nothing is scored."""


def _fail(msg) -> NoReturn:
    raise CoverageInputError(msg)


def _is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= SHA_HEX


def _rate(num, den):
    return round(num / den, 4) if den else None


def threshold(n: int) -> int:
    return math.ceil(0.75 * n)


# ---------------------------------------------------------------- obligations
def index_obligations(obligations) -> dict:
    """Validate the hit-rule structure and index it by source id."""
    if not isinstance(obligations, dict) or not isinstance(obligations.get("sources"), list):
        _fail("obligations: 'sources' list required")
    out, seen_ids = {}, set()

    def claim_id(i):
        if not isinstance(i, str) or not i:
            _fail(f"obligations: id must be a non-empty string: {i!r}")
        if i in seen_ids:
            _fail(f"obligations: duplicate id {i}")
        seen_ids.add(i)

    for s in obligations["sources"]:
        sid = s.get("source_id")
        if sid in out:
            _fail(f"obligations: duplicate source {sid}")
        claim_id(sid)
        lang = s.get("summary_language_required")
        if not isinstance(lang, str) or not lang:
            _fail(f"{sid}: summary_language_required missing")
        items = []
        must_cover = s.get("must_cover")
        if not isinstance(must_cover, list) or not must_cover:
            _fail(f"{sid}: must_cover empty")
        for it in must_cover:
            iid = it.get("id")
            claim_id(iid)
            reqs = it.get("requirements")
            if not isinstance(reqs, list) or not reqs:
                _fail(f"{iid}: requirements empty")
            for r in reqs:
                claim_id(r.get("id"))
                if not isinstance(r.get("essential"), bool):
                    _fail(f"{r.get('id')}: essential must be a bool")
            hr = it.get("hit_rule") or {}
            if hr.get("type") == "all_essential":
                if any("group" in r for r in reqs):
                    _fail(f"{iid}: group on an all_essential item")
                groups = None
                k = None
            elif hr.get("type") == "at_least_k_groups":
                groups, k = hr.get("groups"), hr.get("k")
                if (
                    not isinstance(groups, list)
                    or len(set(groups)) != len(groups)
                    or not groups
                    or not isinstance(k, int)
                    or isinstance(k, bool)
                    or not 1 <= k <= len(groups)
                ):
                    _fail(f"{iid}: malformed at_least_k_groups rule")
                if {r.get("group") for r in reqs} != set(groups):
                    _fail(f"{iid}: requirement groups != hit_rule groups")
                for g in groups:
                    if not any(r["essential"] for r in reqs if r.get("group") == g):
                        _fail(f"{iid}: group {g} has no essential requirement")
            else:
                _fail(f"{iid}: unknown hit_rule type {hr.get('type')!r}")
            if not any(r["essential"] for r in reqs):
                _fail(f"{iid}: no essential requirement")
            cond = it.get("conditional")
            if cond is not None and (
                not isinstance(cond, dict)
                or not isinstance(cond.get("applies_if"), str)
                or not cond["applies_if"]
            ):
                _fail(f"{iid}: conditional block needs an applies_if condition")
            items.append(
                {
                    "id": iid,
                    "type": hr["type"],
                    "groups": groups,
                    "k": k,
                    "conditional": cond is not None,
                    "requirements": [
                        {"id": r["id"], "essential": r["essential"], "group": r.get("group")}
                        for r in reqs
                    ],
                }
            )
        quals = [q.get("id") for q in s.get("must_preserve_qualifiers", [])]
        nots = [n.get("id") for n in s.get("must_not_assert", [])]
        for x in quals + nots:
            claim_id(x)
        out[sid] = {"language": lang, "items": items, "qualifiers": quals, "must_not": nots}
    return out


# ------------------------------------------------------------------ judgments
def _check_mapping(j, key, allowed_ids, values, where):
    m = j.get(key)
    if not isinstance(m, dict):
        _fail(f"{where}: '{key}' must be an object")
    for k, v in m.items():
        if k not in allowed_ids:
            _fail(f"{where}: {key} names unknown id {k!r}")
        if not isinstance(v, str) or v not in values:
            _fail(f"{where}: {key}[{k}] = {v!r} not in {sorted(values)}")
    return m


def validate_judgments(doc, index, output_hashes, producer, obligations_sha256, version):
    if not isinstance(doc, dict):
        _fail("judgments: top-level object required")
    if doc.get("coverage_obligations_sha256") != obligations_sha256:
        _fail("judgments are bound to a different coverage_obligations.json")
    if version is not None and doc.get("reference_version") != version:
        _fail(f"judgments reference_version {doc.get('reference_version')!r} != {version!r}")
    rows = doc.get("judgments")
    if not isinstance(rows, list):
        _fail("judgments: 'judgments' must be a list")
    by_source = {}
    for n, j in enumerate(rows, 1):
        where = f"judgment {n}"
        if not isinstance(j, dict):
            _fail(f"{where}: not an object")
        extra = set(j) - JUDGMENT_KEYS
        if extra:
            _fail(f"{where}: unexpected fields {sorted(extra)}")
        sid = j.get("source_id")
        if sid not in index:
            _fail(f"{where}: unknown source {sid!r}")
        where = f"judgment {sid}"
        if sid in by_source:
            _fail(f"{where}: duplicate judgment for one source")
        if sid not in output_hashes:
            _fail(f"{where}: no output was supplied for this source")
        if not _is_sha(j.get("output_sha256")) or j["output_sha256"] != output_hashes[sid]:
            _fail(f"{where}: output_sha256 does not match the supplied output bytes")
        adj = j.get("adjudication")
        if not isinstance(adj, dict):
            _fail(f"{where}: adjudication block required")
        judge = adj.get("judge_id")
        if not isinstance(judge, str) or not judge.strip():
            _fail(f"{where}: adjudication.judge_id required")
        if adj.get("independent") is not True:
            _fail(f"{where}: adjudication.independent must be true")
        if producer is not None and judge == producer:
            _fail(f"{where}: judge_id equals the producer; judgments must be independent")
        src = index[sid]
        cond_ids = {i["id"] for i in src["items"] if i["conditional"]}
        all_items = {i["id"] for i in src["items"]}
        app = j.get("applicability", {})
        if isinstance(app, dict):
            for k in app:
                if k in all_items and k not in cond_ids:
                    _fail(f"{where}: applicability judged for unconditional item {k}")
        _check_mapping(j, "applicability", cond_ids, APPLICABILITY_VALUES, where)
        req_ids = {r["id"] for i in src["items"] for r in i["requirements"]}
        reqs = _check_mapping(j, "requirements", req_ids, REQ_VALUES, where)
        _check_mapping(j, "qualifiers", set(src["qualifiers"]), QUALIFIER_VALUES, where)
        _check_mapping(j, "must_not", set(src["must_not"]), MUST_NOT_VALUES, where)
        for it in src["items"]:
            na = j["applicability"].get(it["id"]) == "not_applicable"
            for r in it["requirements"]:
                v = reqs.get(r["id"])
                if na and v not in (None, "not_applicable"):
                    _fail(f"{where}: {r['id']} judged {v!r} under a not_applicable item")
                if not na and v == "not_applicable":
                    _fail(f"{where}: {r['id']} is not_applicable but its item is not")
        if "language" in j:
            lang = j["language"]
            if not isinstance(lang, dict) or not isinstance(lang.get("observed"), str):
                _fail(f"{where}: language must be {{'observed': <tag or 'unknown'>}}")
        by_source[sid] = j
    return by_source


# -------------------------------------------------------------------- scoring
def _item_eval(it, values):
    """values: requirement id -> True (hit) / False (miss) / None (no credit, unknown)."""

    def ess_ok(reqs, optimistic):
        return all(
            (values[r["id"]] is True) or (optimistic and values[r["id"]] is None)
            for r in reqs
            if r["essential"]
        )

    def frozen(optimistic):
        if it["type"] == "all_essential":
            return ess_ok(it["requirements"], optimistic), None
        hit_groups = [
            g
            for g in it["groups"]
            if ess_ok([r for r in it["requirements"] if r["group"] == g], optimistic)
        ]
        return len(hit_groups) >= it["k"], hit_groups

    f_hit, groups = frozen(False)
    f_best, _ = frozen(True)
    return {
        "frozen_hit": f_hit,
        "strict_hit": ess_ok(it["requirements"], False),
        "frozen_best": f_best,
        "strict_best": ess_ok(it["requirements"], True),
        "groups_hit": groups,
    }


def _gate_pass(cov_known, cov_best, gates_fail, complete):
    if gates_fail or not cov_best:
        return False
    if not complete:
        return None
    return cov_known


def score_source(sid, src, j, has_output):
    incomplete = []
    items_out = {}
    not_applicable = []
    req_hit = req_app = ess_hit = ess_app = 0
    hits = {"frozen": 0, "strict": 0, "frozen_best": 0, "strict_best": 0}
    n_app = 0
    for it in src["items"]:
        if j is not None and it["conditional"]:
            a = j["applicability"].get(it["id"])
            if a is None or a == "unknown":
                incomplete.append(f"{it['id']}: conditional applicability not judged")
                applicability = "unjudged"
            else:
                applicability = a
        else:
            applicability = "applicable" if j is not None else "no_judgment"
        if applicability == "not_applicable":
            not_applicable.append(it["id"])
            items_out[it["id"]] = {"applicability": "not_applicable"}
            continue
        n_app += 1
        values = {}
        for r in it["requirements"]:
            judged = j is not None and applicability == "applicable"
            v = j["requirements"].get(r["id"]) if judged else None
            if judged and v in (None, "unknown"):
                incomplete.append(f"{r['id']}: requirement not judged")
            values[r["id"]] = True if v == "hit" else (False if v == "miss" else None)
        # Unjudged applicability keeps the item applicable with no credit; its best
        # case is applicable-and-hit, which is never worse than dropping it from n.
        ev = _item_eval(it, values)
        n_hit = sum(values[r["id"]] is True for r in it["requirements"])
        req_hit += n_hit
        req_app += len(it["requirements"])
        ess_hit += sum(values[r["id"]] is True for r in it["requirements"] if r["essential"])
        ess_app += sum(r["essential"] for r in it["requirements"])
        hits["frozen"] += ev["frozen_hit"]
        hits["strict"] += ev["strict_hit"]
        hits["frozen_best"] += ev["frozen_best"] if j is not None else 0
        hits["strict_best"] += ev["strict_best"] if j is not None else 0
        items_out[it["id"]] = {
            "applicability": applicability,
            "frozen_hit": ev["frozen_hit"],
            "strict_hit": ev["strict_hit"],
            "groups_hit": ev["groups_hit"],
            "credit": round(n_hit / len(it["requirements"]), 4),
        }
    thr = threshold(n_app)

    quals = {"respected": [], "omitted": [], "violated": [], "unjudged": []}
    nots = {"violated": [], "unjudged": []}
    observed = None
    if j is not None:
        for q in src["qualifiers"]:
            v = j["qualifiers"].get(q)
            quals["unjudged" if v in (None, "unknown") else v].append(q)
        for x in src["must_not"]:
            v = j["must_not"].get(x)
            if v in (None, "unknown"):
                nots["unjudged"].append(x)
            elif v == "violated":
                nots["violated"].append(x)
        observed = (j.get("language") or {}).get("observed")
        incomplete += [f"{q}: qualifier not judged" for q in quals["unjudged"]]
        incomplete += [f"{x}: must_not not judged" for x in nots["unjudged"]]
        if observed in (None, "unknown"):
            incomplete.append("language not judged")
    lang_ok = None if observed in (None, "unknown") else observed == src["language"]
    gates_fail = bool(quals["violated"] or nots["violated"] or lang_ok is False)

    if not has_output:
        status = "missing_output"
    elif j is None:
        status = "incomplete"
        incomplete.append("no judgment supplied for this output")
    else:
        status = "incomplete" if incomplete else "complete"
    complete = status == "complete"

    def rubric(key):
        known = hits[key] >= thr
        best = hits[f"{key}_best"] >= thr if has_output and j is not None else known
        if not has_output:
            src_pass = False
        elif j is None:
            src_pass = None
        else:
            src_pass = _gate_pass(known, best, gates_fail, complete)
        out = {
            "item_hits": hits[key],
            "coverage_pass": known if complete else (False if not best else None),
            "source_pass": src_pass,
        }
        if not has_output:
            out["coverage_pass"] = False
        return out

    frozen, strict = rubric("frozen"), rubric("strict")
    all_items = hits["strict"] == n_app
    strict["all_items_hit"] = all_items if complete else (False if not has_output else None)
    if not complete and has_output and j is not None and hits["strict_best"] < n_app:
        strict["all_items_hit"] = False
    return {
        "status": status,
        "incomplete": incomplete,
        "items_declared": len(src["items"]),
        "items_applicable": n_app,
        "not_applicable_items": not_applicable,
        "threshold": thr,
        "frozen_v2": frozen,
        "strict": strict,
        "requirements_hit": req_hit,
        "requirements_applicable": req_app,
        "atomic_recall": _rate(req_hit, req_app),
        "essential_hit": ess_hit,
        "essential_applicable": ess_app,
        "essential_recall": _rate(ess_hit, ess_app),
        "qualifiers": {**quals, "ok": None if quals["unjudged"] else not quals["violated"]},
        "must_not": {**nots, "ok": None if nots["unjudged"] else not nots["violated"]},
        "language": {"required": src["language"], "observed": observed, "ok": lang_ok},
        "items": items_out,
    }


def score(
    obligations,
    judgments_doc,
    output_hashes,
    producer=None,
    obligations_sha256=None,
    reference_version=None,
):
    index = index_obligations(obligations)
    if not isinstance(output_hashes, dict):
        _fail("output hashes must be a mapping")
    for sid, h in output_hashes.items():
        if sid not in index:
            _fail(f"output supplied for unknown source {sid!r}")
        if not _is_sha(h):
            _fail(f"{sid}: output hash must be lowercase sha256 hex")
    if reference_version is None:
        reference_version = obligations.get("version")
    by_source = validate_judgments(
        judgments_doc, index, output_hashes, producer, obligations_sha256, reference_version
    )
    sources = {
        sid: score_source(sid, src, by_source.get(sid), sid in output_hashes)
        for sid, src in index.items()
    }
    n = len(sources)
    vals = list(sources.values())

    def agg(key):
        passed = sum(s[key]["source_pass"] is True for s in vals)
        out = {
            "sources_passed": passed,
            "sources_failed": sum(s[key]["source_pass"] is False for s in vals),
            "sources_undetermined": sum(s[key]["source_pass"] is None for s in vals),
            "source_pass_rate": _rate(passed, n),
            "item_hits": sum(s[key]["item_hits"] for s in vals),
            "items_applicable": sum(s["items_applicable"] for s in vals),
        }
        out["item_hit_rate"] = _rate(out["item_hits"], out["items_applicable"])
        return out

    strict = agg("strict")
    strict["sources_all_items_hit"] = sum(s["strict"]["all_items_hit"] is True for s in vals)
    rh = sum(s["requirements_hit"] for s in vals)
    ra = sum(s["requirements_applicable"] for s in vals)
    eh = sum(s["essential_hit"] for s in vals)
    ea = sum(s["essential_applicable"] for s in vals)
    return {
        "reference_version": reference_version,
        "coverage_obligations_sha256": obligations_sha256,
        "producer": producer,
        "disclaimer": DISCLAIMER,
        "complete": all(s["status"] == "complete" for s in vals),
        "aggregate": {
            "sources_expected": n,
            "sources_complete": sum(s["status"] == "complete" for s in vals),
            "sources_incomplete": sum(s["status"] == "incomplete" for s in vals),
            "sources_missing_output": sum(s["status"] == "missing_output" for s in vals),
            "frozen_v2": agg("frozen_v2"),
            "strict": strict,
            "requirements_hit": rh,
            "requirements_applicable": ra,
            "atomic_recall": _rate(rh, ra),
            "essential_hit": eh,
            "essential_applicable": ea,
            "essential_recall": _rate(eh, ea),
        },
        "sources": sources,
    }


# ------------------------------------------------------------------------ CLI
def _sha_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_inputs(root, outputs_path, judgments_path):
    obl_path = os.path.join(root, "coverage_obligations.json")
    obl_sha = _sha_file(obl_path)
    mpath = os.path.join(root, "MANIFEST.sha256")
    if not os.path.exists(mpath):
        _fail(f"{mpath} missing; refusing to score unpinned obligations")
    recorded = {}
    with open(mpath, encoding="utf-8") as f:
        for ln in f:
            if ln.strip() and not ln.startswith("#"):
                h, name = ln.split()
                recorded[name] = h
    if recorded.get("coverage_obligations.json") != obl_sha:
        _fail("coverage_obligations.json does not match MANIFEST.sha256")
    outputs = _load_json(outputs_path)
    if not isinstance(outputs, dict) or not isinstance(outputs.get("outputs"), list):
        _fail("outputs: {'producer', 'outputs': [...]} required")
    producer = outputs.get("producer")
    if not isinstance(producer, str) or not producer:
        _fail("outputs: producer id required (used to enforce judge independence)")
    base = os.path.dirname(os.path.abspath(outputs_path))
    hashes = {}
    for o in outputs["outputs"]:
        sid, rel = o.get("source_id"), o.get("path")
        if not isinstance(sid, str) or not isinstance(rel, str):
            _fail("outputs: each entry needs source_id and path")
        if sid in hashes:
            _fail(f"outputs: duplicate output for {sid}")
        hashes[sid] = _sha_file(os.path.join(base, rel))
    return _load_json(obl_path), obl_sha, hashes, producer, _load_json(judgments_path)


def main():
    ap = argparse.ArgumentParser(description="Aggregate independent coverage judgments.")
    ap.add_argument("--root", required=True, help="reference dir with coverage_obligations.json")
    ap.add_argument("--outputs", required=True, help="outputs manifest JSON")
    ap.add_argument("--judgments", required=True, help="independent judgments JSON")
    ap.add_argument("--require-complete", action="store_true")
    a = ap.parse_args()
    try:
        obl, obl_sha, hashes, producer, judgments = load_inputs(a.root, a.outputs, a.judgments)
        rep = score(obl, judgments, hashes, producer=producer, obligations_sha256=obl_sha)
    except (CoverageInputError, OSError, json.JSONDecodeError) as e:
        print(f"invalid coverage input: {e}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    sys.exit(3 if a.require_complete and not rep["complete"] else 0)


if __name__ == "__main__":
    main()
