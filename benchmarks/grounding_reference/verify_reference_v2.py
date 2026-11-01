#!/usr/bin/env python3
"""Portable verifier for grounding-reference-v2 (stdlib only; no network, no model).

All locations are explicit arguments; nothing is read from hard-coded paths:

  --root DIR      the v2 reference directory (private)
  --baseline DIR  the frozen v1 reference directory it was derived from
  --sources DIR   captured SourceDocs as DIR/<source_id>/source.json
  --baseline-pin FILE         optional committed copy of the v1 MANIFEST.sha256
  --expect-manifest-sha256 H  optional pinned SHA-256 of ROOT/MANIFEST.sha256

Checks:
* hashes: v2 and v1 MANIFEST files cover exactly the expected file sets and
  match on disk; no unlisted files; optional pins; v2 sources.json is
  byte-identical to v1; every SourceDoc file/content hash (identical-content
  aliases verified through their target).
* unchanged v1: every v1 record line hash, every v1 field value, case order,
  pair groups and splits. A v1 field may differ only if it is in the
  changeable set and a CHANGES.jsonl entry records exactly that original and
  new value, the case cites it in change_refs and carries adjudication_v2.
  Labels may change only away from "uncertain" (no supported<->unsupported).
* typed v2 fields: flags (exact key set, types, recomputed occurrence counts
  and valid source ids), reference_spans roles/relations/credits_citation,
  E3 context_spans, guideline_refs defined in ROOT/GUIDELINES.md.
* coverage: item ids/points unchanged, atomic requirement structure, hit
  rules (all_essential / at_least_k_groups), conditional blocks, qualifiers
  and must-not items unchanged, evidence changes logged and audit-flagged.
* every span (cases, context, coverage, ledger offsets) against SourceDoc
  bodies, by code-point offsets, UTF-8 byte offsets and SHA-256.

Prints a JSON summary (counts only, no case text). Exit 1 on any error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

SPLITS = ("calibration", "heldout")
SPLIT_PREFIX = {"calibration": "GR-CAL-", "heldout": "GR-HO-"}
V1_FILES = [
    "sources.json",
    "cases.calibration.jsonl",
    "cases.heldout.jsonl",
    "coverage_obligations.json",
]
V2_FILES = V1_FILES + ["CHANGES.jsonl", "GUIDELINES.md", "README.md", "VERIFICATION.json"]
LABELS = {"supported": "positive", "unsupported": "hard_negative", "uncertain": "uncertain"}
TASKS = {"claim_support", "quote_fidelity", "citation_locator"}
CHANGEABLE_FIELDS = {"label", "polarity", "rationale", "gold_support_spans"}
NEW_REQUIRED = {
    "reference_spans",
    "flags",
    "context_spans",
    "guideline_refs",
    "v1_record_sha256",
    "change_refs",
}
NEW_OPTIONAL = {"adjudication_v2"}
FLAG_KEYS = {
    "supported_elsewhere",
    "multi_occurrence",
    "valid_source_ids",
    "primary_metric",
    "claim_graded",
    "claim_field",
    "notes_residual",
}
CLAIM_FIELD = {
    "claim_support": "claim",
    "quote_fidelity": "mutation_description",
    "citation_locator": "ungraded_context",
}
REF_EXTRA = {"role", "relation", "credits_citation", "note"}
RELATIONS = {None, "contradicts", "does_not_entail"}
CONTRADICTION_MUTATIONS = {"number", "negation"}
SPAN_KEYS = {"source_id", "start", "end", "byte_start", "byte_end", "sha256", "chars"}
LEDGER_KEYS = {
    "change_id",
    "file",
    "target",
    "field",
    "original",
    "new",
    "rationale",
    "decided_by",
    "source_evidence",
}
ADJ_V2_KEYS = ("by", "decision_by", "method", "date")
SPAN_LIST_FIELDS = {"gold_support_spans"}
LOCATOR_KEYS = {"source_id", "start", "end"}
RULE_RE = re.compile(r"\*\*([A-Z]{1,2}\d+[a-z]?)\b")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fsha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def nonempty_str(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def span_key(sp):
    return (sp.get("source_id"), sp.get("start"), sp.get("end"))


def occurrences(body: str, q: str):
    out, i = [], body.find(q)
    while i >= 0 and q:
        out.append({"start": i, "end": i + len(q)})
        i = body.find(q, i + 1)
    return out


class Checker:
    def __init__(self):
        self.errors: list[str] = []
        self.checks = 0
        self.spans = Counter()

    def ok(self, cond, msg) -> bool:
        self.checks += 1
        if not cond:
            self.errors.append(msg)
        return bool(cond)


# --------------------------------------------------------------------- hashes
def read_manifest(path, ck: Checker, label):
    entries = {}
    if not ck.ok(os.path.isfile(path), f"{label}: MANIFEST.sha256 missing"):
        return entries
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.split()
            if ck.ok(len(parts) == 2 and len(parts[0]) == 64, f"{label}: malformed manifest line"):
                ck.ok(parts[1] not in entries, f"{label}: duplicate manifest entry {parts[1]}")
                entries[parts[1]] = parts[0]
    return entries


def check_dir(d, expected, ck: Checker, label):
    if not ck.ok(os.path.isdir(d), f"{label}: directory missing: {d}"):
        return {}
    entries = read_manifest(os.path.join(d, "MANIFEST.sha256"), ck, label)
    ck.ok(
        set(entries) == set(expected),
        f"{label}: manifest lists {sorted(set(entries) ^ set(expected))} unexpectedly",
    )
    listing = set(os.listdir(d)) - {"MANIFEST.sha256"}
    ck.ok(listing == set(expected), f"{label}: unexpected files {sorted(listing ^ set(expected))}")
    for name in expected:
        p = os.path.join(d, name)
        ck.ok(os.path.isfile(p) and entries.get(name) == fsha(p), f"{label}: hash mismatch {name}")
    return entries


def load_bodies(sources, sources_dir, ck: Checker):
    bodies, aliased = {}, []
    rows = sources.get("sources", []) if isinstance(sources, dict) else []
    ck.ok(bool(rows), "sources.json: no sources")
    meta = {s.get("source_id"): s for s in rows}
    ck.ok(len(meta) == len(rows), "sources.json: duplicate source ids")
    for s in rows:
        sid = s.get("source_id")
        p = os.path.join(sources_dir, str(sid), "source.json")
        if not os.path.isfile(p):
            continue
        ck.ok(fsha(p) == s.get("private_sourcedoc_file_sha256"), f"{sid}: SourceDoc file hash")
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            body = doc["content"]
        except (OSError, ValueError, KeyError, TypeError):
            ck.ok(False, f"{sid}: unreadable SourceDoc")
            continue
        ck.ok(
            sha(body) == s.get("content_sha256") and len(body) == s.get("chars"),
            f"{sid}: content hash/chars",
        )
        prov = doc.get("provenance") if isinstance(doc, dict) else None
        if isinstance(prov, dict) and "content_sha256" in prov:
            ck.ok(prov["content_sha256"] == sha(body), f"{sid}: provenance content hash")
        bodies[sid] = body
    for s in rows:
        sid = s.get("source_id")
        if sid in bodies:
            continue
        alias = s.get("content_identical_to")
        if ck.ok(alias in bodies, f"{sid}: SourceDoc missing and no verified identical alias"):
            ck.ok(sha(bodies[alias]) == s.get("content_sha256"), f"{sid}: alias content hash")
            bodies[sid] = bodies[alias]
            aliased.append(sid)
    return bodies, aliased


def check_span(ck: Checker, bodies, sp, where, kind, require_text=False):
    ck.spans[kind] += 1
    if not ck.ok(isinstance(sp, dict) and set(sp) >= SPAN_KEYS, f"{where}: malformed span"):
        return
    body = bodies.get(sp["source_id"])
    if not ck.ok(body is not None, f"{where}: unknown source {sp['source_id']!r}"):
        return
    a, b = sp["start"], sp["end"]
    if not ck.ok(is_int(a) and is_int(b) and 0 <= a < b <= len(body), f"{where}: bad offsets"):
        return
    t = body[a:b]
    ck.ok(sp["sha256"] == sha(t), f"{where}: span hash at offsets")
    ck.ok(sp["chars"] == b - a, f"{where}: span chars")
    ck.ok(
        sp["byte_start"] == len(body[:a].encode("utf-8"))
        and sp["byte_end"] == len(body[:b].encode("utf-8")),
        f"{where}: byte offsets",
    )
    if "text" in sp:
        ck.ok(sp["text"] == t, f"{where}: stored text != source")
    elif require_text:
        ck.ok(False, f"{where}: span text required")


def read_json(path, ck: Checker, label):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        ck.ok(False, f"{label}: unreadable JSON ({type(e).__name__})")
        return None


def read_jsonl(path, ck: Checker, label):
    raw, rows = [], []
    try:
        with open(path, encoding="utf-8") as f:
            raw = [ln for ln in f.read().split("\n") if ln.strip()]
        rows = [json.loads(ln) for ln in raw]
    except (OSError, ValueError) as e:
        ck.ok(False, f"{label}: unreadable JSONL ({type(e).__name__})")
        return [], []
    return raw, rows


# ---------------------------------------------------------------------- cases
def ledger_value_matches(field, recorded, actual) -> bool:
    """Exact value, or (span-list fields only) the exact ordered locator projection.

    The full spans are hash-verified against the SourceDocs separately, so a
    locator list binds the same content without duplicating quoted text.
    """
    if recorded == actual:
        return True
    if field not in SPAN_LIST_FIELDS or not isinstance(recorded, list):
        return False
    if not isinstance(actual, list) or len(recorded) != len(actual):
        return False
    return all(
        isinstance(r, dict)
        and set(r) == LOCATOR_KEYS
        and isinstance(a, dict)
        and span_key(r) == span_key(a)
        for r, a in zip(recorded, actual, strict=True)
    )


def check_flags(ck: Checker, c, bodies, occ, where):
    f = c.get("flags")
    if not ck.ok(isinstance(f, dict) and set(f) == FLAG_KEYS, f"{where}: flags key set"):
        return
    task, label, mutation = c.get("task"), c.get("label"), c.get("mutation")
    quote = task != "claim_support"
    ck.ok(f["primary_metric"] is (label != "uncertain"), f"{where}: flags.primary_metric")
    ck.ok(f["claim_graded"] is (task == "claim_support"), f"{where}: flags.claim_graded")
    ck.ok(f["claim_field"] == CLAIM_FIELD.get(task), f"{where}: flags.claim_field")
    want_se = True if (not quote and mutation == "wrong_span") else None
    ck.ok(f["supported_elsewhere"] is want_se, f"{where}: flags.supported_elsewhere")
    marker = "supported_elsewhere=true" in (c.get("notes") or "")
    ck.ok(marker == bool(f["supported_elsewhere"]), f"{where}: supported_elsewhere vs notes")
    nr = f["notes_residual"]
    ck.ok(nr is None or nonempty_str(nr), f"{where}: flags.notes_residual type")
    if quote:
        ck.ok(
            is_int(f["multi_occurrence"]) and f["multi_occurrence"] == len(occ),
            f"{where}: flags.multi_occurrence != recomputed occurrences",
        )
        valid = sorted(s for s, b in bodies.items() if c.get("candidate_quote", "\0") in b)
        ck.ok(
            isinstance(f["valid_source_ids"], list) and f["valid_source_ids"] == valid,
            f"{where}: flags.valid_source_ids != recomputed",
        )
        cited = c.get("cited_source_id")
        ck.ok(
            (label == "supported") == (cited in valid) == (len(occ) > 0),
            f"{where}: label vs valid_source_ids/occurrences",
        )
    else:
        ck.ok(
            f["multi_occurrence"] is None and f["valid_source_ids"] is None,
            f"{where}: quote flags on a claim case",
        )


def allowed_roles(c):
    if c["task"] == "quote_fidelity":
        return {"raw_original"}
    if c["task"] == "citation_locator":
        return {"true_location_other_source"}
    if c.get("mutation") == "wrong_span":
        return {"supported_elsewhere", "cited_partial_support"}
    return {"label_basis"}


def check_case_semantics(ck: Checker, c, bodies, rules):
    cid = c.get("id")
    where = str(cid)
    task, label, mutation = c.get("task"), c.get("label"), c.get("mutation")
    if not ck.ok(task in TASKS and label in LABELS, f"{where}: task/label"):
        return
    ck.ok(c.get("polarity") == LABELS[label], f"{where}: polarity/label")
    ck.ok((mutation is None) == (label == "supported"), f"{where}: mutation/label")
    ck.ok(
        c.get("source_id") in bodies and c.get("cited_source_id") in bodies,
        f"{where}: unknown source id",
    )
    for key in ("evidence", "gold_support_spans", "reference_spans", "context_spans"):
        ck.ok(isinstance(c.get(key), list), f"{where}: {key} must be a list")
    evidence = c.get("evidence") if isinstance(c.get("evidence"), list) else []
    gold = c.get("gold_support_spans") if isinstance(c.get("gold_support_spans"), list) else []
    refs = c.get("reference_spans") if isinstance(c.get("reference_spans"), list) else []
    ctx = c.get("context_spans") if isinstance(c.get("context_spans"), list) else []
    for e in evidence:
        check_span(ck, bodies, e, f"{where}.evidence", "case_evidence")
    for e in gold:
        check_span(ck, bodies, e, f"{where}.gold_support_spans", "case_gold_support")
    for e in refs:
        check_span(ck, bodies, e, f"{where}.reference_spans", "case_reference")
    for e in ctx:
        check_span(ck, bodies, e, f"{where}.context_spans", "case_context")

    # reference spans: typed alias of gold_support_spans
    stripped = [
        {k: v for k, v in r.items() if k not in REF_EXTRA} for r in refs if isinstance(r, dict)
    ]
    ck.ok(stripped == gold, f"{where}: reference_spans != gold_support_spans")
    roles = allowed_roles(c)
    for r in refs:
        if not isinstance(r, dict):
            continue
        ck.ok(r.get("credits_citation") is False, f"{where}: credits_citation must be false")
        ck.ok(r.get("role") in roles, f"{where}: role {r.get('role')!r} not allowed")
        rel = r.get("relation", "<absent>")
        ck.ok(rel in RELATIONS, f"{where}: relation {rel!r} unknown")
        if task == "claim_support":
            ck.ok(
                (rel == "contradicts") == (mutation in CONTRADICTION_MUTATIONS),
                f"{where}: relation 'contradicts' iff number/negation mutation",
            )
        else:
            ck.ok(rel is None, f"{where}: relation on a quote/locator case")
        if rel == "does_not_entail":
            ck.ok(
                label == "unsupported" and r.get("role") == "label_basis",
                f"{where}: does_not_entail needs unsupported label_basis",
            )
        if "note" in r:
            ck.ok(nonempty_str(r["note"]), f"{where}: reference note type")
    if label == "supported":
        ck.ok(gold == [], f"{where}: supported case with gold_support_spans")
    if label == "unsupported":
        ck.ok(len(refs) > 0, f"{where}: unsupported case without reference spans")

    occ = []
    if task == "claim_support":
        ck.ok("candidate_quote" not in c, f"{where}: candidate quote on a claim case")
        ck.ok(len(evidence) > 0, f"{where}: claim without evidence")
        if mutation == "wrong_span":
            ck.ok(
                any(isinstance(r, dict) and r.get("role") == "supported_elsewhere" for r in refs),
                f"{where}: wrong_span lacks a supported_elsewhere span",
            )
    else:
        q = c.get("candidate_quote")
        cited = c.get("cited_source_id")
        if ck.ok(isinstance(q, str) and cited in bodies, f"{where}: candidate quote"):
            ck.ok(c.get("candidate_quote_sha256") == sha(q), f"{where}: candidate quote hash")
            occ = occurrences(bodies[cited], q)
            ck.ok(c.get("candidate_quote_occurrences") == occ, f"{where}: occurrences differ")
            ck.ok(
                sorted(span_key(e) for e in evidence)
                == sorted((cited, o["start"], o["end"]) for o in occ),
                f"{where}: evidence != all occurrences in the cited source",
            )
            for r in refs:
                if isinstance(r, dict) and r.get("role") == "true_location_other_source":
                    ck.ok(
                        r.get("source_id") != cited
                        and r.get("text") == q
                        and q in bodies.get(r.get("source_id"), ""),
                        f"{where}: true_location_other_source",
                    )
            m = re.search(r"multi_occurrence=(\d+)", c.get("notes") or "")
            if m:
                ck.ok(int(m.group(1)) == len(occ), f"{where}: notes multi_occurrence")
    check_flags(ck, c, bodies, occ, where)

    ev_keys = {span_key(e) for e in evidence if isinstance(e, dict)}
    for e in ctx:
        if not isinstance(e, dict):
            continue
        ck.ok(e.get("rule") == "E3", f"{where}: context rule must be E3")
        ck.ok(e.get("source_id") == c.get("source_id"), f"{where}: context from another source")
        ck.ok(nonempty_str(e.get("use")), f"{where}: context use missing")
        ck.ok(span_key(e) not in ev_keys, f"{where}: context duplicates evidence")
    refs_g = c.get("guideline_refs")
    if ck.ok(
        isinstance(refs_g, list) and refs_g and all(isinstance(x, str) for x in refs_g),
        f"{where}: guideline_refs must be a non-empty list",
    ):
        ck.ok(len(set(refs_g)) == len(refs_g), f"{where}: duplicate guideline_refs")
        ck.ok(
            set(refs_g) <= rules, f"{where}: undefined guideline refs {sorted(set(refs_g) - rules)}"
        )
        ck.ok(("E3" in refs_g) == bool(ctx), f"{where}: E3 ref iff context_spans")


def check_cases(ck: Checker, base, root, bodies, ledger, rules):
    by_target = defaultdict(list)
    for ch in ledger:
        by_target[(ch.get("file"), ch.get("target"))].append(ch)
    v1_all, v2_all, label_changes = [], [], []
    for split in SPLITS:
        fn = f"cases.{split}.jsonl"
        raw1, c1 = read_jsonl(os.path.join(base, fn), ck, f"baseline {fn}")
        _, c2 = read_jsonl(os.path.join(root, fn), ck, f"v2 {fn}")
        ck.ok(len(c1) == len(c2), f"{fn}: case count differs from baseline")
        for line, a, b in zip(raw1, c1, c2, strict=False):
            cid = a.get("id")
            if not ck.ok(isinstance(b, dict) and b.get("id") == cid, f"{fn}: order/id {cid}"):
                continue
            ck.ok(b.get("v1_record_sha256") == sha(line), f"{cid}: v1_record_sha256")
            ck.ok(
                b.get("split") == split and str(cid).startswith(SPLIT_PREFIX[split]),
                f"{cid}: split/id prefix",
            )
            new = set(b) - set(a)
            ck.ok(
                set(b) >= NEW_REQUIRED, f"{cid}: missing v2 fields {sorted(NEW_REQUIRED - set(b))}"
            )
            ck.ok(
                new <= NEW_REQUIRED | NEW_OPTIONAL,
                f"{cid}: unexpected fields {sorted(new - NEW_REQUIRED - NEW_OPTIONAL)}",
            )
            entries = by_target.get((fn, cid), [])
            ledger_ids = {e.get("change_id") for e in entries}
            refs = b.get("change_refs") if isinstance(b.get("change_refs"), list) else []
            ck.ok(isinstance(b.get("change_refs"), list), f"{cid}: change_refs must be a list")
            ck.ok(
                set(refs) == ledger_ids and len(refs) == len(set(refs)),
                f"{cid}: change_refs != ledger entries for this case",
            )
            changed = False
            for k, v1v in a.items():
                if k not in b:
                    ck.ok(False, f"{cid}.{k}: v1 field dropped")
                    continue
                if b[k] == v1v:
                    ck.checks += 1
                    continue
                changed = True
                logged = [
                    e
                    for e in entries
                    if e.get("field") == k
                    and ledger_value_matches(k, e.get("original"), v1v)
                    and ledger_value_matches(k, e.get("new"), b[k])
                ]
                ck.ok(
                    k in CHANGEABLE_FIELDS and len(logged) == 1,
                    f"{cid}.{k}: v1 field changed without an allowed, exact ledger entry",
                )
                if k == "label":
                    label_changes.append((cid, v1v, b[k]))
                    ck.ok(
                        v1v == "uncertain",
                        f"{cid}: label flip {v1v}->{b[k]} forbidden (only uncertain may change)",
                    )
                    ck.ok(
                        all(nonempty_str(e.get("decided_by")) for e in logged),
                        f"{cid}: label change lacks decided_by",
                    )
            for e in entries:  # every ledger row for a case must describe a real change
                f = e.get("field")
                ck.ok(
                    f in a
                    and f in b
                    and a[f] != b[f]
                    and ledger_value_matches(f, e.get("original"), a[f])
                    and ledger_value_matches(f, e.get("new"), b[f]),
                    f"{cid}: ledger {e.get('change_id')} does not match data",
                )
            if changed:
                adj = b.get("adjudication_v2")
                ck.ok(
                    isinstance(adj, dict) and all(nonempty_str(adj.get(x)) for x in ADJ_V2_KEYS),
                    f"{cid}: changed case lacks adjudication_v2",
                )
            else:
                ck.ok("adjudication_v2" not in b, f"{cid}: adjudication_v2 on an unchanged case")
            v1_all.append(a)
            v2_all.append(b)
    known_case_ids = {c.get("id") for c in v2_all}
    for ch in ledger:
        if str(ch.get("file", "")).startswith("cases."):
            ck.ok(ch.get("target") in known_case_ids, f"ledger {ch.get('change_id')}: unknown case")
    ids = Counter(c.get("id") for c in v2_all)
    keys = Counter(c.get("key") for c in v2_all)
    ck.ok(all(n == 1 for n in ids.values()), "duplicate case ids")
    ck.ok(all(n == 1 for n in keys.values()), "duplicate case keys")
    for c in v2_all:
        check_case_semantics(ck, c, bodies, rules)
    pg1, pg2 = defaultdict(set), defaultdict(set)
    for c in v1_all:
        pg1[c.get("pair_group")].add(c.get("id"))
    for c in v2_all:
        pg2[c.get("pair_group")].add(c.get("id"))
    ck.ok(pg1 == pg2, "pair group membership changed")
    split_of = {c.get("id"): c.get("split") for c in v2_all}
    for g, mem in pg2.items():
        ck.ok(len({split_of[i] for i in mem}) == 1, f"pair group {g} crosses splits")
    ck.ok(
        not any(c.get("split") == "calibration" and c.get("label") == "uncertain" for c in v2_all),
        "uncertain case in calibration",
    )
    return v2_all, label_changes, pg2


# ------------------------------------------------------------------- coverage
def check_requirements(ck: Checker, it, sid, bodies, req_ids):
    iid = it["id"]
    reqs = it.get("requirements")
    if not ck.ok(isinstance(reqs, list) and reqs, f"{iid}: requirements empty"):
        return []
    item_keys = {span_key(e) for e in it.get("evidence", []) if isinstance(e, dict)}
    for r in reqs:
        rid = r.get("id") if isinstance(r, dict) else None
        if not ck.ok(isinstance(rid, str), f"{iid}: requirement without id"):
            continue
        ck.ok(
            set(r) <= {"id", "text", "essential", "evidence", "group"},
            f"{rid}: unexpected requirement fields",
        )
        ck.ok(rid not in req_ids, f"{rid}: duplicate requirement id")
        req_ids.add(rid)
        ck.ok(rid.startswith(iid + "."), f"{rid}: id prefix")
        ck.ok(nonempty_str(r.get("text")), f"{rid}: text")
        ck.ok(isinstance(r.get("essential"), bool), f"{rid}: essential must be a bool")
        ev = r.get("evidence")
        if ck.ok(isinstance(ev, list) and ev, f"{rid}: no evidence"):
            for e in ev:
                check_span(ck, bodies, e, f"{rid}.evidence", "cov_requirement_evidence", True)
                ck.ok(isinstance(e, dict) and e.get("source_id") == sid, f"{rid}: foreign source")
            ck.ok(
                {span_key(e) for e in ev if isinstance(e, dict)} <= item_keys,
                f"{rid}: requirement evidence not in item evidence",
            )
    return [r for r in reqs if isinstance(r, dict)]


def check_hit_rule(ck: Checker, it, reqs):
    iid, hr = it["id"], it.get("hit_rule")
    if not ck.ok(isinstance(hr, dict), f"{iid}: hit_rule missing"):
        return None
    essential = [r for r in reqs if r.get("essential") is True]
    if hr.get("type") == "all_essential":
        ck.ok(hr.get("rule") == "CV2", f"{iid}: all_essential rule tag")
        ck.ok(bool(essential), f"{iid}: no essential requirement")
        ck.ok(not any("group" in r for r in reqs), f"{iid}: group on all_essential item")
        return "all_essential"
    if hr.get("type") == "at_least_k_groups":
        groups, k = hr.get("groups"), hr.get("k")
        ok_groups = ck.ok(
            isinstance(groups, list)
            and groups
            and all(nonempty_str(g) for g in groups)
            and len(set(groups)) == len(groups),
            f"{iid}: bundle groups",
        )
        ck.ok(hr.get("rule") == "CV3", f"{iid}: bundle rule tag")
        if ok_groups:
            ck.ok(is_int(k) and 1 <= k <= len(groups), f"{iid}: bundle k out of range")
            ck.ok({r.get("group") for r in reqs} == set(groups), f"{iid}: groups != requirements")
            for g in groups:
                ck.ok(
                    any(r.get("essential") is True and r.get("group") == g for r in reqs),
                    f"{iid}: group {g} lacks an essential requirement",
                )
        return "bundle"
    ck.ok(False, f"{iid}: unknown hit_rule type {hr.get('type')!r}")
    return None


def check_coverage(ck: Checker, base, root, bodies, ledger):
    cov1 = read_json(os.path.join(base, "coverage_obligations.json"), ck, "baseline coverage")
    cov2 = read_json(os.path.join(root, "coverage_obligations.json"), ck, "v2 coverage")
    if not isinstance(cov1, dict) or not isinstance(cov2, dict):
        return {}
    ck.ok(
        nonempty_str(cov2.get("version")) and isinstance(cov2.get("scoring_rule"), dict),
        "coverage: version/scoring_rule",
    )
    s1s, s2s = cov1.get("sources", []), cov2.get("sources", [])
    ck.ok(
        [s.get("source_id") for s in s1s] == [s.get("source_id") for s in s2s],
        "coverage: source list changed",
    )
    logged = {(c.get("target"), c.get("field")) for c in ledger}
    req_ids: set = set()
    stats = {
        "sources": 0,
        "must_cover_items": 0,
        "atomic_requirements": 0,
        "essential_requirements": 0,
        "bundle_items": 0,
        "conditional_items": 0,
        "items_with_evidence_changed": 0,
        "qualifiers": 0,
        "must_not": 0,
    }
    for s1, s in zip(s1s, s2s, strict=False):
        sid = s.get("source_id")
        stats["sources"] += 1
        ck.ok(
            s.get("content_sha256") == s1.get("content_sha256")
            and sid in bodies
            and s.get("content_sha256") == sha(bodies[sid]),
            f"{sid}: coverage content hash",
        )
        for k in ("summary_language_required", "notes"):
            ck.ok(s.get(k) == s1.get(k), f"{sid}: {k} changed")
        items1, items = s1.get("must_cover", []), s.get("must_cover", [])
        ck.ok([i.get("id") for i in items1] == [i.get("id") for i in items], f"{sid}: item ids")
        for a, it in zip(items1, items, strict=False):
            iid = it.get("id")
            stats["must_cover_items"] += 1
            ck.ok(
                a.get("point") == it.get("point") and a.get("gloss_en") == it.get("gloss_en"),
                f"{iid}: point/gloss changed",
            )
            for e in it.get("evidence", []):
                check_span(ck, bodies, e, f"{iid}.evidence", "cov_item_evidence", True)
            reqs = check_requirements(ck, it, sid, bodies, req_ids)
            stats["atomic_requirements"] += len(reqs)
            stats["essential_requirements"] += sum(r.get("essential") is True for r in reqs)
            kind = check_hit_rule(ck, it, reqs)
            stats["bundle_items"] += kind == "bundle"
            if "conditional" in it:
                stats["conditional_items"] += 1
                cond = it["conditional"]
                ck.ok(
                    isinstance(cond, dict)
                    and nonempty_str(cond.get("applies_if"))
                    and nonempty_str(cond.get("if_not_applicable"))
                    and cond.get("rule") == "CV4",
                    f"{iid}: conditional block",
                )
            if "judge_notes" in it:
                ck.ok(isinstance(it["judge_notes"], list | str), f"{iid}: judge_notes type")
            changed = {span_key(e) for e in a.get("evidence", [])} != {
                span_key(e) for e in it.get("evidence", [])
            }
            audit = it.get("audit")
            ck.ok(
                isinstance(audit, dict)
                and isinstance(audit.get("reviewer_flagged"), bool)
                and audit.get("evidence_changed") is changed,
                f"{iid}: audit evidence flag",
            )
            stats["items_with_evidence_changed"] += changed
            if changed:
                ck.ok((iid, "must_cover.evidence") in logged, f"{iid}: unlogged evidence change")
        q1, q2 = s1.get("must_preserve_qualifiers", []), s.get("must_preserve_qualifiers", [])
        ck.ok(
            [(q.get("id"), q.get("requirement")) for q in q1]
            == [(q.get("id"), q.get("requirement")) for q in q2],
            f"{sid}: qualifiers changed",
        )
        for a, q in zip(q1, q2, strict=False):
            stats["qualifiers"] += 1
            for e in q.get("evidence", []):
                check_span(ck, bodies, e, f"{q.get('id')}.evidence", "cov_qualifier_evidence")
            if [span_key(e) for e in a.get("evidence", [])] != [
                span_key(e) for e in q.get("evidence", [])
            ]:
                ck.ok(
                    (q.get("id"), "must_preserve_qualifiers.evidence") in logged,
                    f"{q.get('id')}: unlogged qualifier evidence change",
                )
        n1, n2 = s1.get("must_not_assert", []), s.get("must_not_assert", [])
        ck.ok(
            [(n.get("id"), n.get("assertion")) for n in n1]
            == [(n.get("id"), n.get("assertion")) for n in n2],
            f"{sid}: must_not changed",
        )
        for a, n in zip(n1, n2, strict=False):
            stats["must_not"] += 1
            ck.ok(
                a.get("why") == n.get("why") and a.get("origin") == n.get("origin"),
                f"{n.get('id')}: why/origin changed",
            )
            ck.ok(
                [span_key(e) for e in a.get("contrary_evidence", [])]
                == [span_key(e) for e in n.get("contrary_evidence", [])],
                f"{n.get('id')}: contrary evidence changed",
            )
            for e in n.get("contrary_evidence", []):
                check_span(ck, bodies, e, f"{n.get('id')}.contrary", "cov_must_not_evidence")
        o1, o2 = s1.get("out_of_scope", []), s.get("out_of_scope", [])
        ck.ok(
            [(o.get("region"), o.get("policy"), span_key(o.get("span", {}))) for o in o1]
            == [(o.get("region"), o.get("policy"), span_key(o.get("span", {}))) for o in o2],
            f"{sid}: out_of_scope changed",
        )
        for o in o2:
            if "span" in o:
                check_span(ck, bodies, o["span"], f"{sid}.out_of_scope", "cov_out_of_scope")
    return stats


def check_ledger(ck: Checker, root, bodies):
    _, rows = read_jsonl(os.path.join(root, "CHANGES.jsonl"), ck, "CHANGES.jsonl")
    ids = Counter(r.get("change_id") for r in rows if isinstance(r, dict))
    ck.ok(all(n == 1 for n in ids.values()), "ledger: duplicate change ids")
    for r in rows:
        if not ck.ok(isinstance(r, dict) and set(r) >= LEDGER_KEYS, "ledger: malformed row"):
            continue
        ck.ok(
            nonempty_str(r["decided_by"]) and nonempty_str(r["rationale"]),
            f"ledger {r['change_id']}: decided_by/rationale",
        )
        ev = r["source_evidence"]
        if ck.ok(isinstance(ev, list), f"ledger {r['change_id']}: source_evidence list"):
            for e in ev:
                body = bodies.get(e.get("source_id")) if isinstance(e, dict) else None
                ck.ok(
                    body is not None
                    and is_int(e.get("start"))
                    and is_int(e.get("end"))
                    and 0 <= e["start"] < e["end"] <= len(body),
                    f"ledger {r['change_id']}: evidence offsets",
                )
    return [r for r in rows if isinstance(r, dict)]


# ----------------------------------------------------------------------- main
def verify(root, baseline, sources_dir, baseline_pin=None, expect_manifest_sha256=None):
    root, baseline, sources_dir = str(root), str(baseline), str(sources_dir)
    ck = Checker()
    base_entries = check_dir(baseline, V1_FILES, ck, "baseline")
    n_base_errors = len(ck.errors)
    check_dir(root, V2_FILES, ck, "v2")
    baseline_ok = n_base_errors == 0
    if baseline_pin is not None:
        pin = read_manifest(str(baseline_pin), ck, "baseline pin")
        baseline_ok &= ck.ok(pin == base_entries, "baseline MANIFEST differs from the baseline pin")
    if expect_manifest_sha256 is not None:
        mp = os.path.join(root, "MANIFEST.sha256")
        ck.ok(
            os.path.isfile(mp) and fsha(mp) == expect_manifest_sha256,
            "v2 MANIFEST.sha256 differs from the expected MANIFEST digest",
        )
    v1_unchanged = baseline_ok
    s1p, s2p = os.path.join(baseline, "sources.json"), os.path.join(root, "sources.json")
    ck.ok(
        os.path.isfile(s1p) and os.path.isfile(s2p) and fsha(s1p) == fsha(s2p),
        "v2 sources.json is not byte-identical to the baseline",
    )
    sources = read_json(s2p, ck, "sources.json") or {}
    if not ck.ok(os.path.isdir(sources_dir), f"sources dir missing: {sources_dir}"):
        return {"checks": ck.checks, "errors": ck.errors, "v1_unchanged": v1_unchanged}
    bodies, aliased = load_bodies(sources, sources_dir, ck)
    gpath = os.path.join(root, "GUIDELINES.md")
    rules = set()
    if ck.ok(os.path.isfile(gpath), "GUIDELINES.md missing"):
        with open(gpath, encoding="utf-8") as f:
            rules = set(RULE_RE.findall(f.read()))
    ledger = check_ledger(ck, root, bodies)
    cases, label_changes, pairs = check_cases(ck, baseline, root, bodies, ledger, rules)
    cov = check_coverage(ck, baseline, root, bodies, ledger)
    counts = {
        "cases": len(cases),
        "by_split": dict(Counter(c.get("split") for c in cases)),
        "by_label": dict(Counter(c.get("label") for c in cases)),
        "by_task": dict(Counter(c.get("task") for c in cases)),
        "primary_metric_cases": sum(
            isinstance(c.get("flags"), dict) and c["flags"].get("primary_metric") is True
            for c in cases
        ),
        "label_changes_vs_baseline": len(label_changes),
        "label_change_directions": dict(Counter(f"{a}->{b}" for _, a, b in label_changes)),
        "pair_groups": len(pairs),
        "context_span_cases": sum(bool(c.get("context_spans")) for c in cases),
        "supported_elsewhere_cases": sum(
            isinstance(c.get("flags"), dict) and c["flags"].get("supported_elsewhere") is True
            for c in cases
        ),
        "ledger_entries": len(ledger),
        "guideline_rules_defined": len(rules),
        "bodies_verified": len(bodies),
        "bodies_verified_via_identical_alias": sorted(aliased),
        "coverage": cov,
        "spans_verified": dict(sorted(ck.spans.items())),
        "spans_verified_total": sum(ck.spans.values()),
    }
    return {
        "checks": ck.checks,
        "errors": ck.errors,
        "counts": counts,
        "v1_unchanged": v1_unchanged,
    }


def main():
    ap = argparse.ArgumentParser(description="Verify grounding-reference-v2 portably.")
    ap.add_argument("--root", required=True, help="v2 reference directory")
    ap.add_argument("--baseline", required=True, help="frozen v1 reference directory")
    ap.add_argument("--sources", required=True, help="dir with <source_id>/source.json")
    ap.add_argument("--baseline-pin", help="committed v1 MANIFEST.sha256 to compare against")
    ap.add_argument("--expect-manifest-sha256", help="pinned SHA-256 of ROOT/MANIFEST.sha256")
    a = ap.parse_args()
    res = verify(a.root, a.baseline, a.sources, a.baseline_pin, a.expect_manifest_sha256)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    sys.exit(1 if res["errors"] else 0)


if __name__ == "__main__":
    main()
