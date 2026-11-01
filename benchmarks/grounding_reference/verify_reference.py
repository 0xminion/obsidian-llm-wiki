#!/usr/bin/env python3
"""Verify the frozen grounding reference (stdlib only, no network, no model).

Public mode (default) checks schema, split separation, counts/balance,
stored-snippet hashes, copyright exposure caps and the file manifest.

Private mode (--durable-dir DIR, --archive PATH or --capture-dir DIR) additionally re-reads the
nine captured SourceDocs and re-verifies every source hash, every span offset
and hash (cases, gold spans, obligations), and recomputes every candidate quote
occurrence list. The archive is read in memory; nothing is extracted.

Exit status is non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "sources.json",
    "cases.calibration.jsonl",
    "cases.heldout.jsonl",
    "coverage_obligations.json",
]
LABELS = {"supported": "positive", "unsupported": "hard_negative", "uncertain": "uncertain"}
TASKS = {"claim_support", "quote_fidelity", "citation_locator"}
REQUIRED_MUTATIONS = {
    "negation",
    "causality",
    "actor",
    "number",
    "qualifier",
    "wrong_span",
    "case_alteration",
    "language_alteration",
}
SPAN_TEXT_MAX = 320


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fsha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Checker:
    def __init__(self):
        self.errors: list[str] = []
        self.checks = 0

    def ok(self, cond: bool, msg: str):
        self.checks += 1
        if not cond:
            self.errors.append(msg)


def load(root):
    with open(os.path.join(root, "sources.json"), encoding="utf-8") as f:
        sources = json.load(f)
    cases = []
    for split in ("calibration", "heldout"):
        with open(os.path.join(root, f"cases.{split}.jsonl"), encoding="utf-8") as f:
            for _n, line in enumerate(f, 1):
                c = json.loads(line)
                c["_file_split"] = split
                cases.append(c)
    with open(os.path.join(root, "coverage_obligations.json"), encoding="utf-8") as f:
        obl = json.load(f)
    return sources, cases, obl


def all_spans(c):
    return c.get("evidence", []) + c.get("gold_support_spans", [])


def obligation_spans(obl):
    for o in obl["sources"]:
        for grp in ("must_cover", "must_preserve_qualifiers", "must_not_assert"):
            for it in o[grp]:
                for sp in it.get("evidence", []) + it.get("contrary_evidence", []):
                    yield o["source_id"], it["id"], sp
        for oos in o.get("out_of_scope", []):
            if "span" in oos:
                yield o["source_id"], oos["region"], oos["span"]


def public_checks(root, ck: Checker):
    sources, cases, obl = load(root)
    smeta = {s["source_id"]: s for s in sources["sources"]}
    ids, keys = Counter(c["id"] for c in cases), Counter(c["key"] for c in cases)
    ck.ok(all(v == 1 for v in ids.values()), "duplicate case ids")
    ck.ok(all(v == 1 for v in keys.values()), "duplicate case keys")
    for c in cases:
        k = c["key"]
        ck.ok(c["split"] == c["_file_split"], f"{k}: split field != file")
        ck.ok(
            c["id"].startswith("GR-CAL-" if c["split"] == "calibration" else "GR-HO-"),
            f"{k}: id prefix",
        )
        ck.ok(c["task"] in TASKS, f"{k}: task")
        ck.ok(LABELS.get(c["label"]) == c["polarity"], f"{k}: label/polarity")
        ck.ok(c["source_id"] in smeta and c["cited_source_id"] in smeta, f"{k}: unknown source id")
        ck.ok(
            bool(c.get("rationale")) and bool(c.get("adjudication", {}).get("by")),
            f"{k}: rationale/adjudication",
        )
        for sp in all_spans(c):
            ck.ok(sp["end"] - sp["start"] == sp["chars"], f"{k}: span length")
            if "text" in sp:
                ck.ok(len(sp["text"]) <= SPAN_TEXT_MAX, f"{k}: stored snippet too long")
                ck.ok(sha(sp["text"]) == sp["sha256"], f"{k}: snippet hash")
                ck.ok(len(sp["text"]) == sp["chars"], f"{k}: snippet chars")
        if "candidate_quote" in c:
            ck.ok(
                sha(c["candidate_quote"]) == c["candidate_quote_sha256"],
                f"{k}: candidate quote hash",
            )
            occ = c["candidate_quote_occurrences"]
            if c["label"] == "supported":
                ck.ok(len(occ) >= 1, f"{k}: supported quote has no occurrence")
                ev = sorted((e["start"], e["end"]) for e in c["evidence"])
                ck.ok(
                    ev == sorted((o["start"], o["end"]) for o in occ),
                    f"{k}: evidence != all occurrences",
                )
            elif c["label"] == "unsupported":
                ck.ok(len(occ) == 0, f"{k}: unsupported quote occurs in cited source")
                ck.ok(bool(c["gold_support_spans"]), f"{k}: negative quote lacks gold location")
        if c["task"] == "claim_support":
            if c["label"] == "supported":
                ck.ok(bool(c["evidence"]), f"{k}: supported claim lacks evidence")
            if c["label"] == "unsupported":
                ck.ok(
                    bool(c["gold_support_spans"]),
                    f"{k}: unsupported claim lacks gold/contrary spans",
                )

    # split separation: no key, id or pair group may straddle splits
    by_group = {}
    for c in cases:
        by_group.setdefault(c["pair_group"], set()).add(c["split"])
    ck.ok(all(len(v) == 1 for v in by_group.values()), "pair_group straddles calibration/heldout")
    ck.ok(
        not any(c["split"] == "calibration" and c["label"] == "uncertain" for c in cases),
        "uncertain case in calibration",
    )

    # counts / balance / coverage of required phenomena
    pol = Counter(c["polarity"] for c in cases)
    ck.ok(len(cases) >= 40, "fewer than 40 cases")
    ck.ok(
        0.8 <= pol["positive"] / max(1, pol["hard_negative"]) <= 1.25,
        "positives/hard negatives imbalanced",
    )
    for split in ("calibration", "heldout"):
        sp = Counter(c["polarity"] for c in cases if c["split"] == split)
        ck.ok(sp["positive"] > 0 and sp["hard_negative"] > 0, f"{split}: missing a polarity")
    muts = {c["mutation"] for c in cases if c["mutation"]}
    ck.ok(muts >= REQUIRED_MUTATIONS, f"missing mutations {REQUIRED_MUTATIONS - muts}")
    ck.ok("cross_source_id" in muts, "no cross-source wrong-ID case")
    ck.ok(
        any(c["origin"].startswith("historical") and c["label"] == "unsupported" for c in cases),
        "no historical unsupported output",
    )
    ck.ok(
        any(len(c.get("candidate_quote_occurrences", [])) > 1 for c in cases),
        "no repeated-text multi-ID case",
    )
    ck.ok(
        any(c["cited_source_id"] != c["source_id"] and c["label"] == "supported" for c in cases),
        "no duplicate-source valid-ID case",
    )
    langs = {c["source_language"] for c in cases}
    ck.ok({"en", "zh-Hans", "zh-Hant"} <= langs, "language coverage")

    # copyright exposure: unique source chars disclosed through stored text/quotes
    exposed = {s: set() for s in smeta}
    for c in cases:
        for sp in all_spans(c):
            if "text" in sp:
                exposed[sp["source_id"]].update(range(sp["start"], sp["end"]))
        for o in c.get("candidate_quote_occurrences", []):
            exposed[c["cited_source_id"]].update(range(o["start"], o["end"]))
    exposure = {}
    for s, chars in exposed.items():
        total = smeta[s]["chars"]
        cap = 0.10 if total >= 5000 else 0.30
        frac = len(chars) / total
        exposure[s] = round(frac, 4)
        ck.ok(frac <= cap, f"{s}: disclosed {frac:.1%} > cap {cap:.0%}")

    # obligations
    osrc = {o["source_id"] for o in obl["sources"]}
    ck.ok(len(osrc) == 8 and osrc <= set(smeta), "obligations must cover 8 known sources")
    for o in obl["sources"]:
        ck.ok(
            o["content_sha256"] == smeta[o["source_id"]]["content_sha256"],
            f"{o['source_id']}: obligation hash",
        )
        ck.ok(
            len(o["must_cover"]) >= 4 and len(o["must_not_assert"]) >= 1,
            f"{o['source_id']}: obligation counts",
        )
    for _sid, item, sp in obligation_spans(obl):
        ck.ok("text" not in sp, f"{item}: obligation span must be locator-only")
        ck.ok(sp["end"] - sp["start"] == sp["chars"], f"{item}: span length")

    # manifest
    mpath = os.path.join(root, "MANIFEST.sha256")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            lines = [ln.split() for ln in f if ln.strip() and not ln.startswith("#")]
        recorded = {name: h for h, name in lines}
        for name in FILES:
            ck.ok(
                recorded.get(name) == fsha(os.path.join(root, name)), f"manifest mismatch: {name}"
            )
    else:
        ck.ok(False, "MANIFEST.sha256 missing")

    summary = {
        "cases": len(cases),
        "by_split": dict(Counter(c["split"] for c in cases)),
        "by_polarity": dict(pol),
        "by_split_polarity": {
            f"{s}/{p}": n
            for (s, p), n in sorted(Counter((c["split"], c["polarity"]) for c in cases).items())
        },
        "by_task_polarity": {
            f"{t}/{p}": n
            for (t, p), n in sorted(Counter((c["task"], c["polarity"]) for c in cases).items())
        },
        "by_mutation": dict(sorted(Counter(c["mutation"] or "none" for c in cases).items())),
        "by_source": dict(sorted(Counter(c["source_id"] for c in cases).items())),
        "by_source_language": dict(Counter(c["source_language"] for c in cases)),
        "historical_origin": dict(
            Counter(c["polarity"] for c in cases if c["origin"].startswith("historical"))
        ),
        "multi_occurrence_cases": sum(
            1 for c in cases if len(c.get("candidate_quote_occurrences", [])) > 1
        ),
        "cross_source_wrong_id_cases": sum(1 for c in cases if c["mutation"] == "cross_source_id"),
        "pair_groups": len(by_group),
        "disclosed_fraction_by_source": exposure,
        "obligation_sources": len(osrc),
        "obligation_items": sum(
            len(o[g])
            for o in obl["sources"]
            for g in ("must_cover", "must_preserve_qualifiers", "must_not_assert")
        ),
    }
    return sources, cases, obl, summary


def read_bodies(sources, archive=None, capture_dir=None, durable_dir=None):
    want = {s["private_sourcedoc_member"]: s["source_id"] for s in sources["sources"]}
    raw = {}
    if durable_dir:
        for s in sources["sources"]:
            p = os.path.join(durable_dir, s["source_id"], "source.json")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    raw[s["source_id"]] = f.read()
        return raw
    if archive:
        with tarfile.open(archive, "r:gz") as tf:
            for m in tf:
                if m.name in want and m.isfile():
                    fh = tf.extractfile(m)
                    assert fh is not None
                    raw[want[m.name]] = fh.read()
    else:
        assert capture_dir is not None
        for member, sid in want.items():
            with open(os.path.join(capture_dir, member), "rb") as f:
                raw[sid] = f.read()
    return raw


def private_checks(sources, cases, obl, raw, ck: Checker):
    bodies = {}
    aliased = []
    for s in sources["sources"]:
        sid = s["source_id"]
        alias = s.get("content_identical_to")
        if sid not in raw and alias and alias in raw:
            # e.g. T29 (absent from the durable set): body is byte-identical to T28's
            body = json.loads(raw[alias].decode("utf-8"))["content"]
            ck.ok(sha(body) == s["content_sha256"], f"{sid}: alias content hash")
            bodies[sid] = body
            aliased.append(sid)
            continue
        ck.ok(sid in raw, f"{sid}: SourceDoc member not found")
        if sid not in raw:
            continue
        ck.ok(
            hashlib.sha256(raw[sid]).hexdigest() == s["private_sourcedoc_file_sha256"],
            f"{sid}: SourceDoc file hash",
        )
        body = json.loads(raw[sid].decode("utf-8"))["content"]
        ck.ok(
            sha(body) == s["content_sha256"] and len(body) == s["chars"],
            f"{sid}: content hash/chars",
        )
        bodies[sid] = body

    def check_span(label, sp):
        b = bodies[sp["source_id"]]
        t = b[sp["start"] : sp["end"]]
        ck.ok(sha(t) == sp["sha256"], f"{label}: span hash at offsets")
        ck.ok(
            len(b[: sp["start"]].encode("utf-8")) == sp["byte_start"]
            and len(b[: sp["end"]].encode("utf-8")) == sp["byte_end"],
            f"{label}: byte offsets",
        )
        if "text" in sp:
            ck.ok(t == sp["text"], f"{label}: stored snippet != source")

    n_spans = 0
    for c in cases:
        for sp in all_spans(c):
            check_span(c["key"], sp)
            n_spans += 1
        if "candidate_quote" in c:
            b, q, occ, i = bodies[c["cited_source_id"]], c["candidate_quote"], [], -1
            while True:
                i = b.find(q, i + 1)
                if i == -1:
                    break
                occ.append({"start": i, "end": i + len(q)})
            ck.ok(
                occ == c["candidate_quote_occurrences"],
                f"{c['key']}: recomputed occurrences differ",
            )
    for _sid, item, sp in obligation_spans(obl):
        check_span(item, sp)
        n_spans += 1
    positives = [c for c in cases if c["label"] == "supported"]
    pos_spans = sum(len(c["evidence"]) for c in positives)
    return {
        "bodies_verified": len(bodies),
        "bodies_verified_via_identical_alias": aliased,
        "spans_verified": n_spans,
        "positive_cases": len(positives),
        "positive_evidence_spans_verified": pos_spans,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--archive", help="private-evidence.tar.gz (read-only, in-memory)")
    ap.add_argument(
        "--capture-dir", help="directory containing wiki-live50-live-synthesis/inputs/*/source.json"
    )
    ap.add_argument(
        "--durable-dir",
        help="reports/obsidian-grounded/reference-inputs (8 SourceDocs; T29 via T28 alias)",
    )
    a = ap.parse_args()
    ck = Checker()
    sources, cases, obl, summary = public_checks(a.root, ck)
    if a.archive or a.capture_dir or a.durable_dir:
        raw = read_bodies(sources, a.archive, a.capture_dir, a.durable_dir)
        summary["private"] = private_checks(sources, cases, obl, raw, ck)
    summary["checks"] = ck.checks
    summary["errors"] = ck.errors
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    sys.exit(1 if ck.errors else 0)


if __name__ == "__main__":
    main()
