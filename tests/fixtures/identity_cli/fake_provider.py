"""Independent local OpenAI-compatible HTTP double (stdlib only, real socket).

Answers three request shapes honestly from the request content:
* producer  : per-scenario scripted synthesis; claims cite the offered span that
              contains a phrase; optional copy of an offered candidate ID by slug.
* review    : honest token-containment reviewer (supported only if every content
              token of the unit appears in its evidence; claims judged vs cited_text).
* identity  : scripted HONEST assessor; rule = (subject_title, candidate_title,
              required_span_phrase) -> verdict, default "insufficient".
Every request/response is appended to $FAKE_LOG (JSONL) as a receipt.
Usage: python -I fake_provider.py PORT SCENARIO_JSON LOG_JSONL
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, SCENARIO, LOG = int(sys.argv[1]), sys.argv[2], sys.argv[3]
SPAN_RE = re.compile(r"⟦([0-9a-f]{32}-\d+)⟧ ")
CJK = re.compile(r"[\u3400-\u9fff]")
STOP = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "are",
    "was",
    "its",
    "has",
    "from",
    "into",
    "which",
    "also",
    "overview",
    "about",
    "same",
}


def scenario():
    with open(SCENARIO, encoding="utf-8") as fh:
        return json.load(fh)


def spans(text):
    marks = list(SPAN_RE.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1), text[m.end() : end]))
    return out


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    words = {w for w in re.findall(r"[a-z0-9][a-z0-9,']*", text) if len(w) >= 3 and w not in STOP}
    words = {w.strip(",'") for w in words}
    return words | set(CJK.findall(text))


def supported(unit_text, evidence):
    need = tokens(unit_text)
    have = tokens(evidence)
    return bool(need) and need <= have


# ---------------------------------------------------------------- producer
def produce(prompt, sc):
    body = prompt.split("--- SOURCE DOCUMENT", 1)[-1]
    offered = spans(body)
    cand_block = prompt.split("--- BOUNDED CONCEPT CANDIDATES", 1)[-1].split(
        "--- SOURCE DOCUMENT", 1
    )[0]
    for entry in sc["producer"]:
        if entry["marker"] not in body:
            continue
        concepts = []
        for c in entry["concepts"]:
            claims = []
            for cl in c["claims"]:
                ids = [sid for sid, txt in offered if cl["phrase"] in txt]
                assert len(ids) == 1, (cl["phrase"], ids)
                claims.append(
                    {
                        "text": cl["text"],
                        "source_ref": "",
                        "evidence_span_range": {"start_id": ids[0], "end_id": ids[0]},
                    }
                )
            cid = ""
            if c.get("copy_candidate_slug"):
                m = re.search(
                    rf"^  - (concept-[0-9a-f]+) \| {re.escape(c['copy_candidate_slug'])} \|",
                    cand_block,
                    re.M,
                )
                cid = m.group(1) if m else ""
            concepts.append(
                {
                    "title": c["title"],
                    "slug": c["slug"],
                    "canonical_id": cid,
                    "summary": c["summary"],
                    "tags": [],
                    "aliases": c.get("aliases", []),
                    "sections": [{"heading": "Overview", "points": [c["summary"]], "prose": ""}],
                    "claims": claims,
                    "related": c.get("related", []),
                    "confidence": 0.9,
                    "provenance": "extracted",
                    "is_new": not cid,
                }
            )
        return {
            "source_title": entry["source_title"],
            "source_summary": entry["source_summary"],
            "source_tags": [],
            "key_points": [],
            "open_questions": [],
            "language": entry.get("language", "en"),
            "concepts": concepts,
            "maps": [],
        }
    raise AssertionError("no producer entry for source: " + body[:200])


# ---------------------------------------------------------------- reviewer
def review(prompt, schema):
    head, units_json = prompt.split("--- UNITS ---", 1)
    window = spans(head.split("--- SOURCE SPANS ---", 1)[1])
    window_text = "\n".join(t for _i, t in window)
    units = json.loads(units_json)
    out = {}
    for u in units:
        if u["kind"] == "claim":
            ok = supported(u["text"], u.get("cited_text", ""))
            sup = list(u.get("cited_span_ids", [])) if ok else []
        elif u["kind"] == "section_heading" and u["text"] in ("Overview",):
            ok, sup = True, [window[0][0]]
        else:
            utext = (
                re.sub(r" [a-z]+(?:_[a-z]+)+ ", " ", u["text"])
                if u["kind"] == "relation"
                else u["text"]
            )
            ok = supported(utext, window_text)
            sup = [i for i, t in window if tokens(utext) & tokens(t)][:3] if ok else []
            ok = ok and bool(sup)
        out[u["unit_id"]] = {
            "text_sha256": u["text_sha256"],
            "verdict": "supported" if ok else "insufficient",
            "support_span_ids": sup,
            "rationale": "honest token-containment double",
            "model_context_uses": [],
        }
    assert set(schema["properties"]["reviews"]["required"]) == set(out)
    return {"reviews": out}


# ---------------------------------------------------------------- identity
def identity(prompt, schema, sc):
    head, listed_json = prompt.split("--- IDENTITY ASSESSMENTS ---", 1)
    offered = spans(head.split("--- SOURCE SPANS ---", 1)[1])
    text = "\n".join(t for _i, t in offered)
    listed = json.loads(listed_json)
    out = {}
    for item in listed:
        verdict, support = "insufficient", []
        for rule in sc.get("identity", []):
            if (
                rule["subject"] == item["subject"]["title"]
                and rule["candidate"] == item["candidate"]["title"]
                and rule.get("requires_phrase", "") in text
            ):
                verdict = rule["verdict"]
                if verdict == "same":
                    support = [i for i, t in offered if rule.get("requires_phrase", "") in t][:2]
                break
        out[item["assessment_id"]] = {
            "candidate_id": item["candidate"]["candidate_id"],
            "candidate_sha256": item["candidate"]["candidate_sha256"],
            "subject_sha256": item["subject"]["subject_sha256"],
            "verdict": verdict,
            "support_span_ids": support,
            "rationale": "scripted honest identity assessor",
        }
    assert set(schema["properties"]["assessments"]["required"]) == set(out)
    return {"assessments": out}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._log({"method": "GET", "path": self.path, "auth": self.headers.get("Authorization")})
        self._send({"data": [{"id": "gen-model"}]})

    def _log(self, rec):
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        sc = scenario()
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        prompt = body["messages"][-1]["content"]
        props = schema.get("properties", {})
        try:
            if "reviews" in props:
                kind, content = "review", review(prompt, schema)
            elif "assessments" in props:
                kind, content = "identity", identity(prompt, schema, sc)
            else:
                kind, content = "produce", produce(prompt, sc)
        except Exception as exc:  # surface loudly to the client
            self._log({"method": "POST", "error": repr(exc), "prompt": prompt})
            self.send_response(500)
            self.end_headers()
            return
        self._log(
            {
                "method": "POST",
                "kind": kind,
                "auth": self.headers.get("Authorization"),
                "prompt": prompt,
                "response": content,
            }
        )
        self._send(
            {
                "choices": [
                    {
                        "message": {"content": json.dumps(content, ensure_ascii=False)},
                        "finish_reason": "stop",
                    }
                ]
            }
        )


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
