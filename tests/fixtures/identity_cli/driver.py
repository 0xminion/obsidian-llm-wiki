"""Scenario driver: fresh-process public CLI builds against a local HTTP fake.

Usage: python -I driver.py SCENARIO_NAME
Everything (HOME, XDG_*, TMPDIR, vault, cwd, provider log) lives under
RUN_ROOT/<scenario>/. No credentials are present in the child environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
FROZEN = Path(__file__).resolve().parents[3]
PY = sys.executable
HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
from scenarios import SCENARIOS  # noqa: E402

LAUNCH = (
    "import sys, obsidian_llm_wiki as p; "
    f"assert p.__file__.startswith({str(FROZEN)!r}), p.__file__; "
    "sys.stderr.write('PKG=' + p.__file__ + '\\n'); "
    "from obsidian_llm_wiki.cli import app; sys.argv=['olw', *sys.argv[1:]]; app()"
)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def tree(wiki: Path):
    return {
        str(p.relative_to(wiki)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(wiki.rglob("*.md"))
        if p.name != "log.md"
    }


def concept_pages(wiki: Path):
    out = {}
    for p in sorted((wiki / "concepts").glob("*.md")):
        if p.name == "index.md":
            continue
        text = p.read_text(encoding="utf-8")
        m = re.search(r"^# (.+)$", text, re.M)
        out[p.stem] = {"title": m.group(1) if m else None, "text": text}
    return out


def registry_dump(wiki: Path):
    db = wiki / ".llmwiki"
    files = list(db.rglob("*.db")) + list(db.rglob("*.sqlite*"))
    if not files:
        return {"error": "no db", "files": [str(p) for p in db.rglob("*")]}
    con = sqlite3.connect(f"file:{files[0]}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT c.id, c.slug, c.title, c.summary, group_concat(s.filename, ',') "
        "FROM concept c LEFT JOIN concept_source cs ON cs.concept_id=c.id "
        "LEFT JOIN source s ON s.id=cs.source_id GROUP BY c.id ORDER BY c.slug"
    ).fetchall()
    receipts = {}
    for fn, payload in con.execute(
        "SELECT s.filename, sy.payload_json FROM synthesis sy JOIN source s ON s.id=sy.source_id"
    ):
        data = json.loads(payload or "{}")
        rr = data.get("review_receipt") or {}
        receipts[fn] = {
            "identity": rr.get("identity"),
            "concepts": [
                {k: c.get(k) for k in ("title", "slug", "canonical_id", "summary")}
                for c in data.get("concepts", [])
            ],
        }
    con.close()
    return {
        "db": str(files[0]),
        "concepts": [
            dict(zip(("id", "slug", "title", "summary", "owners"), r, strict=True)) for r in rows
        ],
        "receipts": receipts,
    }


def main(name: str):
    sc = SCENARIOS[name]
    base = ROOT / "runs" / name
    shutil.rmtree(base, ignore_errors=True)
    for d in ("home", "cache", "config", "data", "tmp", "cwd"):
        (base / d).mkdir(parents=True)
    vault = base / "vault"
    sources = vault / "04-Wiki" / "sources"
    sources.mkdir(parents=True)
    wiki = vault / "04-Wiki"
    port = free_port()
    host = f"http://127.0.0.1:{port}/v1"
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_PROVIDER=openai\nLLM_HOST={host}\nLLM_MODEL=gen-model\n"
        "EMBEDDINGS_ENABLED=false\nMIN_SOURCE_CHARS=1\nRETRY_COUNT=1\n",
        encoding="utf-8",
    )
    scen_file, log = base / "scenario.json", base / "provider.jsonl"
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(base / "home"),
        "USERPROFILE": str(base / "home"),
        "XDG_CACHE_HOME": str(base / "cache"),
        "XDG_CONFIG_HOME": str(base / "config"),
        "XDG_DATA_HOME": str(base / "data"),
        "TMPDIR": str(base / "tmp"),
        "TMP": str(base / "tmp"),
        "TEMP": str(base / "tmp"),
        "PYTHONPATH": str(FROZEN),
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
        "LANG": "C.UTF-8",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    stages = sc["stages"]
    (scen_file).write_text(json.dumps(stages[0]["provider"], ensure_ascii=False), encoding="utf-8")
    server = subprocess.Popen(
        [PY, "-I", str(HARNESS / "fake_provider.py"), str(port), str(scen_file), str(log)],
        env=env,
        cwd=base / "cwd",
    )
    time.sleep(0.8)
    results = []
    try:
        for i, stage in enumerate(stages):
            scen_file.write_text(
                json.dumps(stage["provider"], ensure_ascii=False), encoding="utf-8"
            )
            for fname, text in stage.get("write", {}).items():
                (sources / fname).write_text(text, encoding="utf-8")
            for fname in stage.get("delete", []):
                (sources / fname).unlink()
            before_lines = log.read_text(encoding="utf-8").count("\n") if log.exists() else 0
            pre = tree(wiki)
            proc = subprocess.run(
                [PY, "-s", "-c", LAUNCH, "build", str(vault)],
                env=env,
                cwd=base / "cwd",
                capture_output=True,
                text=True,
                timeout=600,
            )
            lines = (
                log.read_text(encoding="utf-8").splitlines()[before_lines:] if log.exists() else []
            )
            calls = [json.loads(line) for line in lines]
            post = tree(wiki)
            results.append(
                {
                    "stage": i,
                    "label": stage.get("label", ""),
                    "exit": proc.returncode,
                    "stdout_tail": proc.stdout[-3000:],
                    "stderr_tail": proc.stderr[-3000:],
                    "requests": [c.get("kind") or c.get("method") for c in calls],
                    "auth_headers": sorted({str(c.get("auth")) for c in calls}),
                    "changed_files": sorted(
                        k for k in set(pre) | set(post) if pre.get(k) != post.get(k)
                    ),
                    "tree": post,
                    "concepts": concept_pages(wiki),
                    "registry": registry_dump(wiki),
                    "identity_prompts": [c["prompt"] for c in calls if c.get("kind") == "identity"],
                    "identity_answers": [
                        c["response"] for c in calls if c.get("kind") == "identity"
                    ],
                    "review_insufficient": [
                        dict(r)
                        for c in calls
                        if c.get("kind") == "review"
                        for r in c["response"]["reviews"].values()
                        if r["verdict"] != "supported"
                    ],
                    "errors": [c for c in calls if "error" in c],
                }
            )
    finally:
        server.terminate()
        server.wait()
    out = base / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    # home must stay empty-ish: list anything written there
    home_writes = [
        str(p.relative_to(base)) for d in ("home", "config", "data") for p in (base / d).rglob("*")
    ]
    print(
        json.dumps(
            {
                "scenario": name,
                "home_xdg_writes": home_writes,
                "stages": [
                    {
                        k: r[k]
                        for k in (
                            "stage",
                            "label",
                            "exit",
                            "requests",
                            "auth_headers",
                            "changed_files",
                        )
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
