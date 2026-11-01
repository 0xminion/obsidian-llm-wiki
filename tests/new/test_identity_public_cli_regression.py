"""Public CLI regression against a local honest reviewer/producer.

This fixture does not make external provider calls. The independent audit
harness is copied into the test fixtures and asserts the candidate import path.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.new.immutable_semantic_helpers import (
    published_meta,
    published_registry,
    published_root,
)

HARNESS = Path(__file__).resolve().parents[1] / "fixtures" / "identity_cli"


def _generation_readers(driver, monkeypatch):
    original_pages = driver.concept_pages
    monkeypatch.setattr(driver, "concept_pages", lambda wiki: original_pages(published_root(wiki)))

    def registry_dump(wiki):
        with published_registry(wiki) as registry:
            rows = registry.connection.execute(
                "SELECT c.id,c.slug,c.title,c.summary,group_concat(s.filename, ',') "
                "FROM concept c LEFT JOIN concept_source cs ON cs.concept_id=c.id "
                "LEFT JOIN source s ON s.id=cs.source_id GROUP BY c.id ORDER BY c.slug"
            ).fetchall()
            receipts = {}
            for filename, payload in registry.connection.execute(
                "SELECT s.filename,sy.payload_json FROM synthesis sy "
                "JOIN source s ON s.id=sy.source_id"
            ):
                data = json.loads(payload)
                receipts[filename] = {
                    "identity": (data.get("review_receipt") or {}).get("identity"),
                    "concepts": [
                        {k: c.get(k) for k in ("title", "slug", "canonical_id", "summary")}
                        for c in data.get("concepts", [])
                    ],
                }
        return {
            "db": str(published_meta(wiki) / "registry.db"),
            "concepts": [
                dict(zip(("id", "slug", "title", "summary", "owners"), r, strict=True))
                for r in rows
            ],
            "receipts": receipts,
        }

    monkeypatch.setattr(driver, "registry_dump", registry_dump)


@pytest.mark.parametrize("reverse", [False, True], ids=["il-first", "ma-first"])
def test_same_run_generic_homonyms_remain_separate_in_both_orders(tmp_path, monkeypatch, reverse):
    monkeypatch.syspath_prepend(str(HARNESS))
    spec = importlib.util.spec_from_file_location("identity_a_driver_order", HARNESS / "driver.py")
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    _generation_readers(driver, monkeypatch)
    driver.ROOT = tmp_path
    scenario = copy.deepcopy(driver.SCENARIOS["s1e_same_run_identical"])
    if reverse:
        writes = scenario["stages"][0]["write"]
        il = writes.pop("springfield-il.md")
        ma = writes.pop("springfield-ma.md")
        writes["a-ma.md"] = ma
        writes["z-il.md"] = il
    driver.SCENARIOS["order_test"] = scenario
    driver.main("order_test")
    stages = json.loads((tmp_path / "runs/order_test/result.json").read_text())
    assert [stage["exit"] for stage in stages] == [0, 0]
    pages = stages[0]["concepts"]
    il = {slug for slug, page in pages.items() if "capital of Illinois" in page["text"]}
    ma = {slug for slug, page in pages.items() if "Connecticut River" in page["text"]}
    assert il and ma
    assert il.isdisjoint(ma), "distinct Springfield entities were silently merged"
    assert stages[1]["requests"] == [] and not stages[1]["changed_files"]


def test_generic_identical_definitions_do_not_merge_distinct_source_claims(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HARNESS))
    spec = importlib.util.spec_from_file_location("identity_a_driver", HARNESS / "driver.py")
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    _generation_readers(driver, monkeypatch)
    driver.ROOT = tmp_path
    driver.main("s1_generic_identical_noproposal")
    stages = json.loads((tmp_path / "runs/s1_generic_identical_noproposal/result.json").read_text())
    assert [stage["exit"] for stage in stages] == [0, 0, 0, 0]
    pages = stages[1]["concepts"]
    il = {slug for slug, page in pages.items() if "capital of Illinois" in page["text"]}
    ma = {slug for slug, page in pages.items() if "Connecticut River" in page["text"]}
    assert il and ma, "both reviewed facts must be published"
    assert il.isdisjoint(ma), "distinct Springfield entities were silently merged"
    assert stages[2]["requests"] == [] and not stages[2]["changed_files"]
