"""Offline persistence and prompt-version simulation, not a v3/v4 source upgrade."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("newline", ["lf", "crlf"])
def test_reviewer_version_only_upgrade_rebuilds_then_stable_noop_with_intact_history(
    tmp_path, newline
):
    script = Path(__file__).with_name("quotation_context_process.py")
    repo = Path(__file__).resolve().parents[2]
    results = {}
    for stage in ("old", "new", "noop"):
        env = dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1")
        cmd = [
            sys.executable,
            str(script),
            str(tmp_path),
            stage,
            newline,
            "reviewer-version-only-previous" if stage == "old" else "source-default",
        ]
        completed = subprocess.run(
            cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=60
        )
        (tmp_path / (stage + ".log")).write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results[stage] = json.loads((tmp_path / (stage + "-result.json")).read_text())
    old, new, noop = (results[stage] for stage in ("old", "new", "noop"))
    assert old["module_sha256"] == new["module_sha256"] == noop["module_sha256"]
    for key in ("receipt_version", "unit_version", "citation", "source_sha256"):
        assert old[key] == new[key] == noop[key]
    assert old["prompt_version"] != new["prompt_version"] == noop["prompt_version"]
    assert old["policy"] == new["policy"] == noop["policy"] == "review-context-v4"
    assert old["reviewer_fingerprint"] != new["reviewer_fingerprint"]
    assert new["receipt"] == noop["receipt"]
    assert noop["calls"] == 0 and noop["compiled"] == 0
