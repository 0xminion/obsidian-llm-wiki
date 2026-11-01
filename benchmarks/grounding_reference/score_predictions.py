#!/usr/bin/env python3
"""Score label predictions against the frozen grounding reference (stdlib only).

predictions: JSONL with {"id": "GR-HO-001", "label": "supported"|"unsupported"}.
Any other string label, a null label, or a missing id counts as an
abstention/miss and never receives correctness credit. Duplicate ids, unknown
ids and non-string labels are rejected.

The gold file is validated before scoring. Malformed or duplicate gold
identities, inconsistent label/polarity/mutation fields and (for v2 records)
typed ``flags`` that disagree with the label are rejected instead of scored,
because each of them can silently change a denominator.

Uncertain-gold cases (v2: ``flags.primary_metric == false``, which must agree
with ``label == "uncertain"``) are excluded from primary metrics and reported
separately. Use --split calibration for tuning; report --split heldout once,
after tuning is frozen. "Detection" metrics treat hard negatives
(unsupported) as the positive class, because missing an unsupported claim is
the costly error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {"supported": "positive", "unsupported": "hard_negative", "uncertain": "uncertain"}
TASKS = {"claim_support", "quote_fidelity", "citation_locator"}
SPLIT_PREFIX = {"calibration": "GR-CAL-", "heldout": "GR-HO-"}
V2_FLAG_KEYS = {
    "supported_elsewhere",
    "multi_occurrence",
    "valid_source_ids",
    "primary_metric",
    "claim_graded",
    "claim_field",
    "notes_residual",
}
CLAIM_FIELD = {
    "claim_support": {"claim"},
    "quote_fidelity": {"mutation_description"},
    "citation_locator": {"ungraded_context"},
}


class GoldError(ValueError):
    """The gold file cannot be scored without risking a manipulated denominator."""


def file_sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _flag_errors(c, where) -> list[str]:
    errs = []
    f = c["flags"]
    if not isinstance(f, dict):
        return [f"{where}: flags must be an object"]
    if set(f) != V2_FLAG_KEYS:
        errs.append(f"{where}: flags keys {sorted(set(f) ^ V2_FLAG_KEYS)} differ from schema")
        return errs
    task, label, quote = c["task"], c["label"], c["task"] != "claim_support"
    if f["primary_metric"] is not (label != "uncertain"):
        errs.append(f"{where}: flags.primary_metric inconsistent with label {label!r}")
    if f["claim_graded"] is not (not quote):
        errs.append(f"{where}: flags.claim_graded inconsistent with task {task!r}")
    if f["claim_field"] not in CLAIM_FIELD[task]:
        errs.append(f"{where}: flags.claim_field invalid for task {task!r}")
    wrong_span = c["mutation"] == "wrong_span"
    if f["supported_elsewhere"] is not (True if wrong_span and not quote else None):
        errs.append(f"{where}: flags.supported_elsewhere inconsistent with mutation")
    if quote:
        occ, valid = f["multi_occurrence"], f["valid_source_ids"]
        if not _is_int(occ) or occ < 0:
            errs.append(f"{where}: flags.multi_occurrence must be a non-negative int")
        elif (label == "supported") != (occ > 0):
            errs.append(f"{where}: flags.multi_occurrence inconsistent with label")
        if not isinstance(valid, list) or not all(isinstance(s, str) for s in valid):
            errs.append(f"{where}: flags.valid_source_ids must be a list of ids")
        elif "cited_source_id" in c and (label == "supported") != (c["cited_source_id"] in valid):
            errs.append(f"{where}: flags.valid_source_ids inconsistent with label")
    elif f["multi_occurrence"] is not None or f["valid_source_ids"] is not None:
        errs.append(f"{where}: quote/locator flags set on a claim case")
    nr = f["notes_residual"]
    if nr is not None and not (isinstance(nr, str) and nr):
        errs.append(f"{where}: flags.notes_residual must be null or a non-empty string")
    return errs


def validate_gold(cases, split: str | None = None) -> None:
    """Raise GoldError listing every problem; return None for scorable gold."""
    errs: list[str] = []
    seen: Counter = Counter()
    versions = set()
    for n, c in enumerate(cases, 1):
        where = f"gold record {n}"
        if not isinstance(c, dict):
            errs.append(f"{where}: not an object")
            continue
        cid = c.get("id")
        if not isinstance(cid, str) or not cid.strip():
            errs.append(f"{where}: id must be a non-empty string")
            continue
        where = f"gold {cid}"
        seen[cid] += 1
        csplit = c.get("split")
        if csplit not in SPLIT_PREFIX:
            errs.append(f"{where}: split must be one of {sorted(SPLIT_PREFIX)}")
        else:
            if split is not None and csplit != split:
                errs.append(f"{where}: split {csplit!r} in the {split!r} file")
            if not cid.startswith(SPLIT_PREFIX[csplit]):
                errs.append(f"{where}: id prefix does not match split {csplit!r}")
        label = c.get("label")
        if label not in LABELS:
            errs.append(f"{where}: label {label!r} not in {sorted(LABELS)}")
            continue
        if "polarity" in c and c["polarity"] != LABELS[label]:
            errs.append(f"{where}: polarity inconsistent with label")
        if c.get("task") not in TASKS:
            errs.append(f"{where}: task {c.get('task')!r} not in {sorted(TASKS)}")
            continue
        for key in ("pair_group", "source_language"):
            if not isinstance(c.get(key), str) or not c[key]:
                errs.append(f"{where}: {key} must be a non-empty string")
        mutation = c.get("mutation", "<absent>")
        if mutation == "<absent>" or not (mutation is None or isinstance(mutation, str)):
            errs.append(f"{where}: mutation must be null or a string")
        elif label == "supported" and mutation is not None:
            errs.append(f"{where}: supported gold carries mutation {mutation!r}")
        elif label == "unsupported" and mutation is None:
            errs.append(f"{where}: unsupported gold lacks a mutation")
        versions.add("flags" in c)
        if "flags" in c:
            errs.extend(_flag_errors(c, where))
            if not isinstance(c.get("context_spans"), list):
                errs.append(f"{where}: context_spans must be a list")
    errs.extend(f"duplicate gold id {i!r} x{k}" for i, k in sorted(seen.items()) if k > 1)
    if len(versions) > 1:
        errs.append("gold mixes v1 records (no flags) with v2 records (typed flags)")
    if errs:
        raise GoldError("; ".join(errs))


def is_primary(c) -> bool:
    return c["flags"]["primary_metric"] if "flags" in c else c["label"] != "uncertain"


def _metrics(cs, preds):
    tp = fp = fn = tn = miss = 0
    abstentions = Counter()
    supported_total = sum(c["label"] == "supported" for c in cs)
    false_acceptances = 0
    for c in cs:
        p = preds.get(c["id"])
        if p not in ("supported", "unsupported"):
            miss += 1
            abstentions[c["label"]] += 1
            fn += c["label"] == "unsupported"
            continue  # No abstention receives correctness credit.
        if c["label"] == "unsupported":
            tp += p == "unsupported"
            fn += p != "unsupported"
            false_acceptances += p == "supported"
        else:
            fp += p == "unsupported"
            tn += p == "supported"
    pr, rc, f1 = prf(tp, fp, fn)
    return {
        "n": len(cs),
        "accuracy": round((tp + tn) / len(cs), 4) if cs else None,
        "unsupported_precision": pr,
        "unsupported_recall": rc,
        "unsupported_f1": f1,
        "supported_recall": round(tn / supported_total, 4) if supported_total else None,
        "missing_predictions": miss,
        "abstentions_by_gold": dict(abstentions),
        "supported_accepted": tn,
        "supported_rejected": fp,
        "unsupported_rejected": tp,
        "unsupported_accepted": false_acceptances,
    }


def score(cases, preds, split: str | None = None):
    validate_gold(cases, split)
    out = {}
    scored = [c for c in cases if is_primary(c)]
    groups = defaultdict(list)
    for c in scored:
        groups["all"].append(c)
        groups[f"task:{c['task']}"].append(c)
        groups[f"lang:{c['source_language']}"].append(c)
        f = c.get("flags")
        if f is not None:
            if f["supported_elsewhere"]:
                groups["flag:supported_elsewhere"].append(c)
            if (f["multi_occurrence"] or 0) > 1:
                groups["flag:multi_occurrence"].append(c)
            if c["context_spans"]:
                groups["flag:context_spans"].append(c)
    for name, cs in sorted(groups.items()):
        out[name] = _metrics(cs, preds)
    mut = defaultdict(Counter)
    for c in scored:
        if c["label"] == "unsupported":
            mut[c["mutation"] or "none"]["n"] += 1
            mut[c["mutation"] or "none"]["caught"] += preds.get(c["id"]) == "unsupported"
    out["unsupported_recall_by_mutation"] = {
        m: round(v["caught"] / v["n"], 4) for m, v in sorted(mut.items())
    }
    pairs = defaultdict(list)
    for c in scored:
        pairs[c["pair_group"]].append(c)
    full = [g for g in pairs.values() if {x["label"] for x in g} == {"supported", "unsupported"}]
    out["minimal_pairs_fully_correct"] = (
        f"{sum(all(preds.get(x['id']) == x['label'] for x in g) for g in full)}/{len(full)}"
    )
    unc = [c for c in cases if not is_primary(c)]
    out["uncertain_excluded"] = {c["id"]: preds.get(c["id"]) for c in unc}
    return out


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def verify_manifest(root, names) -> list[str]:
    mpath = os.path.join(root, "MANIFEST.sha256")
    if not os.path.exists(mpath):
        return [f"MANIFEST.sha256 missing in {root}"]
    recorded = {}
    with open(mpath, encoding="utf-8") as f:
        for ln in f:
            if ln.strip() and not ln.startswith("#"):
                h, name = ln.split()
                recorded[name] = h
    errs = []
    for name in names:
        if recorded.get(name) != file_sha256(os.path.join(root, name)):
            errs.append(f"manifest mismatch or missing entry: {name}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--split", choices=["calibration", "heldout"], required=True)
    ap.add_argument("--root", default=HERE)
    ap.add_argument(
        "--verify-manifest",
        action="store_true",
        help="require ROOT/MANIFEST.sha256 to match the gold case files before scoring",
    )
    a = ap.parse_args()
    name = f"cases.{a.split}.jsonl"
    other = "heldout" if a.split == "calibration" else "calibration"
    other_name = f"cases.{other}.jsonl"
    other_path = os.path.join(a.root, other_name)
    if a.verify_manifest:
        names = [name] + ([other_name] if os.path.exists(other_path) else [])
        errs = verify_manifest(a.root, names)
        if errs:
            ap.error("; ".join(errs))
    cases = read_jsonl(os.path.join(a.root, name))
    try:
        validate_gold(cases, a.split)
        if os.path.exists(other_path):
            others = read_jsonl(other_path)
            validate_gold(others, other)
            shared = {c["id"] for c in cases} & {c["id"] for c in others}
            if shared:
                raise GoldError(f"gold ids shared across split files: {sorted(shared)}")
    except GoldError as e:
        ap.error(f"invalid gold: {e}")
    preds = {}
    for r in read_jsonl(a.predictions):
        if not isinstance(r, dict) or not isinstance(r.get("id"), str):
            ap.error("Each prediction needs a string id")
        if r["id"] in preds:
            ap.error(f"Duplicate prediction id: {r['id']}")
        label = r.get("label")
        if label is not None and not isinstance(label, str):
            ap.error(f"Prediction label must be a string or null: {r['id']}")
        preds[r["id"]] = label
    unknown = set(preds) - {c["id"] for c in cases}
    if unknown:
        ap.error(f"Unknown prediction ids: {sorted(unknown)}")
    res = score(cases, preds, a.split)
    res["unknown_prediction_ids"] = sorted(unknown)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    sys.exit(0)


if __name__ == "__main__":
    main()
