"""Offline integrity of the recorded corpus; this does not fetch public URLs."""
import json
from pathlib import Path


def test_live50_manifest_and_recorded_outcomes_cover_same_inputs():
    root = Path(__file__).parents[1] / "fixtures" / "live50"
    manifest = json.loads((root / "manifest.json").read_text())
    outcomes = json.loads((root / "outcomes.json").read_text())
    cases = manifest["cases"]
    assert len(cases) == len({case["url"] for case in cases}) == 50
    expected = {case["id"]: case["url"] for case in cases}
    assert len(outcomes["cases"]) == outcomes["total"] == 50
    assert {case["id"]: case["url"] for case in outcomes["cases"]} == expected
    assert outcomes["raw_returned"] == sum(
        case["retest_status"] == "returned" for case in outcomes["cases"]
    )
