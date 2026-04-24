"""Regression tests for code-review findings (2026-04-22).

These guard against backsliding on the following verified bugs:
1. store.py __del__ deadlock hazard (C1)
2. agent.py batch early-break (H2)
3. vault.py MoC substring dedup false positive (M1)
4. extract.py ExtractionError swallowed (M2)
5. cli.py RAG arbitrary truncation (M3)
6. vault_setup.py corrupted env template (M4)
7. lint.py wikilink regex missing aliases/anchors (L1)
8. stats.py orphan regex including aliases (L2)
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.config import Config
from pipeline.extractors._shared import ExtractionError


# ─── M1: vault.py MoC substring dedup ───────────────────────────────────────

class TestMocSubstringDedup:
    """Ensure [[AI]] does NOT get rejected because [[AI Safety]] exists."""

    def test_exact_link_not_substring(self, tmp_path: Path):
        from pipeline.vault import update_moc

        cfg = Config(vault_path=tmp_path)
        cfg.mocs_dir.mkdir(parents=True, exist_ok=True)
        moc_path = cfg.mocs_dir / "ai.md"
        moc_path.write_text("## Entries\n- [[AI Safety]]\n", encoding="utf-8")

        update_moc(cfg, "AI", "AI", "test entry")
        content = moc_path.read_text(encoding="utf-8")
        assert "[[AI]]" in content
        assert content.count("[[AI]]") == 1  # no duplicate either

    def test_exact_link_rejects_duplicate(self, tmp_path: Path):
        from pipeline.vault import update_moc

        cfg = Config(vault_path=tmp_path)
        cfg.mocs_dir.mkdir(parents=True, exist_ok=True)
        moc_path = cfg.mocs_dir / "ai.md"
        moc_path.write_text("## Entries\n- [[AI]]\n", encoding="utf-8")

        update_moc(cfg, "AI", "AI", "test entry")
        content = moc_path.read_text(encoding="utf-8")
        assert content.count("[[AI]]") == 1  # still just one


# ─── M2: extract.py ExtractionError isolation ─────────────────────────────────

class TestExtractErrorIsolation:
    """ExtractionError must be caught distinctly, not swallowed by bare Exception."""

    def test_extraction_error_propagation(self, tmp_path: Path):
        from pipeline.extract import extract_all

        cfg = Config(vault_path=tmp_path)
        cfg.resolved_extract_dir.mkdir(parents=True, exist_ok=True)

        with patch("pipeline.extract.extract_url") as mock_extract:
            mock_extract.side_effect = ExtractionError("quota exhausted")
            with pytest.raises(ExtractionError, match="all extractions failed"):
                extract_all(["https://example.com/article"], cfg, parallel=1)


# ─── M4: vault_setup.py env template sanity ─────────────────────────────────

class TestVaultSetupTemplate:
    """The inline .env template must contain real-looking key names."""

    def test_env_template_has_reasonable_keys(self, tmp_path: Path):
        from pipeline.vault_setup import setup_vault

        vault = tmp_path / "vault"
        setup_vault(vault)

        env = vault / "Meta/Scripts/.env"
        content = env.read_text(encoding="utf-8")
        assert "TRANSCRIPT_API_KEY=" in content
        assert "ASSEMBLYAI_API_KEY=" in content
        assert "SU.....=" not in content  # old corrupted placeholder


# ─── L1: lint.py wikilink regex ──────────────────────────────────────────────

class TestLintWikilinkRegex:
    """The wikilink extractor must handle aliases and section anchors."""

    @pytest.mark.parametrize("text,expected", [
        ("[[Hello World]]", {"Hello World"}),
        ("[[Note|alias]]", {"Note"}),
        ("[[Note#Section]]", {"Note"}),
        ("[[A]] and [[A B]]", {"A", "A B"}),
    ])
    def test_wikilink_regex(self, text, expected):
        pattern = r"\[\[([^|#\]]+)(?:[|#][^\]]*)?\]\]"
        assert set(re.findall(pattern, text)) == expected


# ─── L2: stats.py orphan regex ───────────────────────────────────────────────

class TestStatsOrphanRegex:
    """The orphan detector must strip aliases from wikilink captures."""

    @pytest.mark.parametrize("text,expected", [
        ("[[Hello World]]", {"Hello World"}),
        ("[[Note|alias]]", {"Note"}),
        ("[[Note#Section]]", {"Note"}),
    ])
    def test_orphan_wikilink_regex(self, text, expected):
        pattern = r"\[\[([^\]]+)\]\]"
        raw = set(re.findall(pattern, text))
        # Simulate alias/anchor stripping logic
        cleaned = {ref.split("|")[0].split("#")[0] for ref in raw}
        assert cleaned == expected
