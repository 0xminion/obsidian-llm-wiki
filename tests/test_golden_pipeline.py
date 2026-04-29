"""Golden path regression tests for the deterministic Stage 3 pipeline."""

import json
from pathlib import Path

from pipeline.config import Config
from pipeline.create import create_file_templates
from pipeline.lint import run_validate
from pipeline.models import Language, Plan, Template


def _cfg(tmp_path: Path) -> Config:
    vault = tmp_path / "vault"
    for rel in [
        "01-Raw",
        "04-Wiki/sources",
        "04-Wiki/entries",
        "04-Wiki/concepts",
        "04-Wiki/mocs",
        "06-Config",
        "08-Archive-Raw",
        "Meta/Scripts",
    ]:
        (vault / rel).mkdir(parents=True, exist_ok=True)
    cfg = Config(vault_path=vault, extract_dir=tmp_path / "extract")
    cfg.resolved_extract_dir.mkdir(parents=True, exist_ok=True)
    cfg.parallel = 1
    return cfg


def test_golden_template_create_pipeline_outputs_validate(tmp_path):
    cfg = _cfg(tmp_path)
    plan = Plan(
        hash="golden123",
        title="Golden Comparison",
        language=Language.EN,
        template=Template.COMPARISON,
        tags=["prediction-markets"],
        concept_new=["Market Resolution"],
        moc_targets=["Prediction Markets"],
    )
    (cfg.resolved_extract_dir / "golden123.json").write_text(
        json.dumps(
            {
                "url": "https://example.com/golden",
                "title": "Golden Comparison",
                "type": "web",
                "author": "Author",
                "content": "This source compares two approaches to prediction-market resolution. " * 20,
            }
        ),
        encoding="utf-8",
    )

    stats = create_file_templates([plan], cfg, use_agent_insights=False)

    assert stats["created"] == 1
    assert (cfg.sources_dir / "golden-comparison.md").exists()
    assert (cfg.entries_dir / "golden-comparison.md").exists()
    entry_content = (cfg.entries_dir / "golden-comparison.md").read_text(encoding="utf-8")
    assert "[[golden-comparison]]" in entry_content
    assert (cfg.concepts_dir / "market-resolution.md").exists()
    assert (cfg.mocs_dir / "prediction-markets.md").exists()
    assert run_validate(cfg.vault_path).total_issues == 0
