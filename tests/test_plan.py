"""Tests for pipeline.plan — Stage 2 planning module."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock


from pipeline.config import Config
from pipeline.models import ConceptMatch, ExtractedSource, Manifest, Plan, SourceType
from pipeline.qmd import run_qmd_query
from pipeline.plan import (
    _fingerprint,
    _jaccard_similarity,
    _parse_agent_output,
    build_plan_prompt,
    concept_search,
    dedup_check,
    generate_plans,
    plan_sources,
)
from pipeline.utils import extract_body as _extract_body


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path) -> Config:
    """Create a Config pointing to tmp_path as vault."""
    return Config(
        vault_path=tmp_path,
        extract_dir=tmp_path / "extract",
        qmd_cmd="qmd",
        qmd_collection="concepts",
        plan_timeout=5,
        agent_cmd="hermes",
    )


def _make_source(url: str = "https://example.com/article", title: str = "Test Article",
                 content: str = "Some content here.") -> ExtractedSource:
    return ExtractedSource(url=url, title=title, content=content, type=SourceType.WEB)


def _make_manifest(*sources) -> Manifest:
    if not sources:
        sources = [_make_source()]
    return Manifest(entries=list(sources))


# ─── Fingerprint helpers ──────────────────────────────────────────────────────

class TestFingerprint:
    def test_basic(self):
        assert _fingerprint("Hello World") == "hello world"

    def test_whitespace_collapsed(self):
        assert _fingerprint("Hello   World\n\nFoo") == "hello world foo"

    def test_truncation(self):
        long_text = "a" * 1000
        result = _fingerprint(long_text)
        assert len(result) == 800

    def test_empty(self):
        assert _fingerprint("") == ""
        assert _fingerprint("   ") == ""

    def test_case_insensitive(self):
        assert _fingerprint("Hello World") == _fingerprint("hello world")


class TestJaccardSimilarity:
    def test_identical(self):
        fp = _fingerprint("the quick brown fox jumps over the lazy dog")
        assert _jaccard_similarity(fp, fp) == 1.0

    def test_disjoint(self):
        assert _jaccard_similarity("abcdef", "xyzwvu") < 0.1

    def test_partial_overlap(self):
        a = _fingerprint("the quick brown fox jumps over the lazy dog")
        b = _fingerprint("the quick brown fox jumps over a lazy cat")
        sim = _jaccard_similarity(a, b)
        assert 0.5 < sim < 1.0

    def test_empty_strings(self):
        assert _jaccard_similarity("", "abc") == 0.0
        assert _jaccard_similarity("abc", "") == 0.0
        assert _jaccard_similarity("", "") == 0.0

    def test_short_strings(self):
        # Strings shorter than ngram size
        assert _jaccard_similarity("ab", "ab", ngram=3) == 0.0


class TestExtractBody:
    def test_with_frontmatter(self):
        md = "---\ntitle: Test\n---\nBody content"
        assert _extract_body(md) == "Body content"

    def test_without_frontmatter(self):
        assert _extract_body("Just body") == "Just body"

    def test_multiline_frontmatter(self):
        md = "---\ntitle: Test\nauthor: Me\n---\nLine 1\nLine 2"
        assert _extract_body(md) == "Line 1\nLine 2"


# ─── Dedup check ──────────────────────────────────────────────────────────────

class TestDedupCheck:
    def test_no_existing_sources(self, tmp_path):
        """No vault sources means no dedup — everything passes through."""
        cfg = _make_config(tmp_path)
        src = _make_source(content="Unique content " * 50)
        manifest = _make_manifest(src)
        result = dedup_check(manifest, cfg)
        assert len(result.entries) == 1
        assert result.entries[0].hash == src.hash

    def test_duplicate_filtered(self, tmp_path):
        """Sources with similar content are filtered out."""
        cfg = _make_config(tmp_path)
        content = "This is a detailed article about quantum computing and its applications " * 20

        # Create an existing source in vault
        sources_dir = tmp_path / "04-Wiki" / "sources"
        sources_dir.mkdir(parents=True)
        (sources_dir / "existing-source.md").write_text(
            "---\ntitle: Existing\n---\n" + content
        )

        # New source with same content
        src = _make_source(url="https://different.com/article", content=content)
        manifest = _make_manifest(src)
        result = dedup_check(manifest, cfg)
        assert len(result.entries) == 0  # filtered out

    def test_different_content_kept(self, tmp_path):
        """Sources with different content are kept."""
        cfg = _make_config(tmp_path)

        # Existing source
        sources_dir = tmp_path / "04-Wiki" / "sources"
        sources_dir.mkdir(parents=True)
        (sources_dir / "existing.md").write_text(
            "---\ntitle: Existing\n---\n" + "About cooking pasta and Italian cuisine " * 20
        )

        # New source about something different
        src = _make_source(content="About machine learning and neural networks " * 20)
        manifest = _make_manifest(src)
        result = dedup_check(manifest, cfg)
        assert len(result.entries) == 1

    def test_short_content_skipped(self, tmp_path):
        """Sources with short content (< 100 chars) are always kept."""
        cfg = _make_config(tmp_path)
        sources_dir = tmp_path / "04-Wiki" / "sources"
        sources_dir.mkdir(parents=True)
        (sources_dir / "existing.md").write_text("---\ntitle: E\n---\n" + "x" * 200)

        src = _make_source(content="Short")
        manifest = _make_manifest(src)
        result = dedup_check(manifest, cfg)
        assert len(result.entries) == 1  # kept because fingerprint < 100

    def test_stub_existing_skipped(self, tmp_path):
        """Existing sources with short content are not used for comparison."""
        cfg = _make_config(tmp_path)
        content = "Detailed content about quantum computing " * 20

        sources_dir = tmp_path / "04-Wiki" / "sources"
        sources_dir.mkdir(parents=True)
        (sources_dir / "stub.md").write_text("---\ntitle: Stub\n---\nShort")

        src = _make_source(content=content)
        manifest = _make_manifest(src)
        result = dedup_check(manifest, cfg)
        assert len(result.entries) == 1


# ─── QMD wrapper ──────────────────────────────────────────────────────────────

class TestRunQmd:
    def test_successful_query(self, monkeypatch):
        """Mock QMD client returning valid results."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.query.return_value = [
            MagicMock(file="concepts/ai-safety.md", score=0.85, collection="concepts"),
            MagicMock(file="concepts/alignment.md", score=0.72, collection="concepts"),
        ]
        monkeypatch.setattr(
            "pipeline.qmd._get_client", lambda base_url="": mock_client
        )
        matches = run_qmd_query("artificial intelligence", "qmd", "/tmp/coll", timeout=5)
        assert len(matches) == 2
        assert matches[0].concept == "ai-safety"
        assert matches[0].score > 0.5
        assert matches[1].concept == "alignment"

    def test_qmd_unavailable_returns_empty(self, monkeypatch):
        monkeypatch.setattr("pipeline.qmd._get_client", lambda base_url="": None)
        matches = run_qmd_query("query", "qmd", "/tmp/coll", timeout=5)
        assert matches == []

    def test_empty_query_returns_empty(self):
        assert run_qmd_query("", "qmd", "/tmp/coll") == []
        assert run_qmd_query("   ", "qmd", "/tmp/coll") == []

    def test_low_score_filtered(self, monkeypatch):
        """QMD server handles low-score filtering internally; pipeline trusts results."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        # Return only high-score results as QMD would
        mock_client.query.return_value = [
            MagicMock(file="concepts/good.md", score=0.85, collection="concepts"),
        ]
        monkeypatch.setattr(
            "pipeline.qmd._get_client", lambda base_url="": mock_client
        )
        matches = run_qmd_query("query", "qmd", "/tmp/coll", timeout=5)
        assert len(matches) == 1
        assert matches[0].concept == "good"

    def test_file_path_parsed(self, monkeypatch):
        """Concept name extracted from QMD result file path."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.query.return_value = [
            MagicMock(
                file="concepts/prediction-markets.md", score=0.9, collection="concepts"
            ),
        ]
        monkeypatch.setattr(
            "pipeline.qmd._get_client", lambda base_url="": mock_client
        )
        matches = run_qmd_query("query", "qmd", "/tmp/coll", timeout=5)
        assert len(matches) == 1
        assert matches[0].concept == "prediction-markets"


class TestQmdConceptSearch:
    """Tests for parallel qmd concept search."""

    def test_parallel_search_returns_all_hashes(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from pipeline.qmd import run_qmd_concept_search

        cfg = _make_config(tmp_path)
        queries = {"hash1": "query one", "hash2": "query two", "hash3": "query three"}

        mock_client = MagicMock()

        def mock_query(query_text, n_results, min_score):
            # Return a deterministic concept per query
            idx = hash(query_text) % 3
            concepts = ["c1", "c2", "c3"]
            return [
                MagicMock(
                    file=f"concepts/{concepts[idx]}.md",
                    score=0.8,
                    collection="concepts",
                )
            ]

        mock_client.query.side_effect = mock_query
        monkeypatch.setattr(
            "pipeline.qmd._get_client", lambda base_url="": mock_client
        )

        results = run_qmd_concept_search(queries, cfg)
        assert len(results) == 3
        assert all(h in results for h in queries)

    def test_empty_queries_return_empty(self, tmp_path):
        from pipeline.qmd import run_qmd_concept_search

        cfg = _make_config(tmp_path)
        queries = {"h1": "", "h2": "   "}
        results = run_qmd_concept_search(queries, cfg)
        assert results == {"h1": [], "h2": []}


class TestQmdConvergence:
    """Tests for qmd convergence wrapper used by creation stage."""

    def test_convergence_returns_dict_format(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from pipeline.qmd import run_qmd_convergence

        cfg = _make_config(tmp_path)
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Create extract files for plans
        for i, h in enumerate(["aaa111", "bbb222"]):
            ext = {"content": f"Content about topic {i}", "title": f"Topic {i}"}
            (extract_dir / f"{h}.json").write_text(json.dumps(ext))

        plans = [
            Plan(
                hash="aaa111", title="Topic 0", concept_new=["New Concept"], concept_updates=[]
            ),
            Plan(hash="bbb222", title="Topic 1", concept_new=[], concept_updates=["Existing"]),
        ]

        monkeypatch.setattr(
            "pipeline.qmd._get_client",
            lambda base_url="": MagicMock(
                query=lambda **kw: [
                    MagicMock(
                        file="concepts/test.md", score=0.9, collection="concepts"
                    )
                ]
            ),
        )
        convergence = run_qmd_convergence(plans, cfg)
        assert "aaa111" in convergence
        assert "bbb222" in convergence
        assert len(convergence["aaa111"]) == 1
        assert convergence["aaa111"][0]["concept"] == "test"
        assert convergence["aaa111"][0]["score"] == 0.9
        # Check dict format (not ConceptMatch objects)
        assert isinstance(convergence["aaa111"], list)
        if convergence["aaa111"]:
            assert "concept" in convergence["aaa111"][0]
            assert "score" in convergence["aaa111"][0]


# ─── Concept search ───────────────────────────────────────────────────────────

class TestConceptSearch:
    def test_returns_mapping(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        src1 = _make_source(url="https://a.com", title="Article A", content="Content A " * 50)
        src2 = _make_source(url="https://b.com", title="Article B", content="Content B " * 50)
        manifest = _make_manifest(src1, src2)

        call_count = 0

        def mock_qmd_search(queries, cfg, no_rerank=False):
            nonlocal call_count
            result = {}
            for h in queries:
                call_count += 1
                result[h] = [ConceptMatch(concept=f"concept-{call_count}", score=0.7)]
            return result

        monkeypatch.setattr("pipeline.qmd.run_qmd_concept_search", mock_qmd_search)
        result = concept_search(manifest, cfg)
        assert src1.hash in result
        assert src2.hash in result
        assert len(result[src1.hash]) == 1
        assert call_count == 2

    def test_empty_manifest(self, tmp_path):
        cfg = _make_config(tmp_path)
        manifest = Manifest(entries=[])
        result = concept_search(manifest, cfg)
        assert result == {}


# ─── Build plan prompt ────────────────────────────────────────────────────────

class TestBuildPlanPrompt:
    def test_basic_structure(self, tmp_path):
        cfg = _make_config(tmp_path)
        src = _make_source(title="My Article", content="Article content here.")
        manifest = _make_manifest(src)
        matches = {src.hash: [ConceptMatch(concept="ai", score=0.8)]}

        prompt = build_plan_prompt(manifest, matches, cfg)

        assert "My Article" in prompt
        assert src.hash in prompt
        assert "ai" in prompt
        assert "0.8" in prompt
        assert "JSON" in prompt
        assert "language" in prompt
        assert "template" in prompt

    def test_includes_common_instructions(self, tmp_path):
        cfg = _make_config(tmp_path)
        prompts_dir = cfg.prompts_dir
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "common-instructions.prompt").write_text("COMMON RULES HERE")

        manifest = _make_manifest()
        prompt = build_plan_prompt(manifest, {}, cfg)
        assert "COMMON RULES HERE" in prompt

    def test_concept_count(self, tmp_path):
        cfg = _make_config(tmp_path)
        concepts_dir = tmp_path / "04-Wiki" / "concepts"
        concepts_dir.mkdir(parents=True)
        (concepts_dir / "ai.md").write_text("# AI")
        (concepts_dir / "ml.md").write_text("# ML")

        manifest = _make_manifest()
        prompt = build_plan_prompt(manifest, {}, cfg)
        assert "2 existing concepts" in prompt

    def test_content_preview_truncated(self, tmp_path):
        cfg = _make_config(tmp_path)
        src = _make_source(content="x" * 500)
        manifest = _make_manifest(src)
        prompt = build_plan_prompt(manifest, {}, cfg)
        # Preview should be 300 chars max
        assert "x" * 400 not in prompt


# ─── Parse agent output ───────────────────────────────────────────────────────

class TestParseAgentOutput:
    def test_valid_json_array(self):
        plans = [
            {"hash": "abc123", "title": "Test", "language": "en"},
            {"hash": "def456", "title": "Test 2", "language": "zh"},
        ]
        raw = json.dumps(plans)
        result = _parse_agent_output(raw)
        assert len(result) == 2

    def test_with_ansi_codes(self):
        plans = [{"hash": "abc", "title": "Test"}]
        raw = "\x1b[32m" + json.dumps(plans) + "\x1b[0m"
        result = _parse_agent_output(raw)
        assert len(result) == 1

    def test_with_box_drawing(self):
        plans = [{"hash": "abc", "title": "Test"}]
        raw = "╭────╮\n│ " + json.dumps(plans) + " │\n╰────╯"
        result = _parse_agent_output(raw)
        assert len(result) == 1

    def test_object_by_object_fallback(self):
        # Partial JSON — objects separated outside array
        obj1 = json.dumps({"hash": "aaa", "title": "T1"})
        obj2 = json.dumps({"hash": "bbb", "title": "T2"})
        raw = f"Here is the output:\n{obj1}\n{obj2}\nDone!"
        result = _parse_agent_output(raw)
        assert len(result) == 2

    def test_invalid_object_skipped(self):
        obj1 = json.dumps({"hash": "aaa", "title": "T1"})
        raw = obj1 + "\n{invalid json}\n"
        result = _parse_agent_output(raw)
        assert len(result) == 1

    def test_missing_hash_skipped(self):
        plans = [
            {"hash": "valid", "title": "Good"},
            {"title": "No hash field"},
        ]
        result = _parse_agent_output(json.dumps(plans))
        assert len(result) == 1

    def test_empty_output(self):
        assert _parse_agent_output("") == []
        assert _parse_agent_output("no json here at all") == []


# ─── Generate plans ───────────────────────────────────────────────────────────

class TestGeneratePlans:
    def test_success(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        src = _make_source(title="Great Article")
        manifest = _make_manifest(src)
        matches = {src.hash: [ConceptMatch(concept="ai", score=0.8)]}

        agent_output = json.dumps([
            {
                "hash": src.hash,
                "title": "Great Article",
                "language": "en",
                "template": "standard",
                "tags": ["ai", "research"],
                "concept_updates": ["ai"],
                "concept_new": [],
                "moc_targets": ["AI Overview"],
            }
        ])

        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.stdout = agent_output
            result.returncode = 0
            return result

        monkeypatch.setattr("pipeline.plan.subprocess.run", mock_run)
        plans = generate_plans(manifest, matches, cfg)
        assert len(plans.plans) == 1
        assert plans.plans[0].hash == src.hash
        assert plans.plans[0].title == "Great Article"
        assert plans.plans[0].tags == ["ai", "research"]

    def test_agent_timeout_retries(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        manifest = _make_manifest()
        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise subprocess.TimeoutExpired(cmd="hermes", timeout=5)

        monkeypatch.setattr("pipeline.plan.subprocess.run", mock_run)
        plans = generate_plans(manifest, {}, cfg)
        assert plans.plans == []
        assert call_count == cfg.max_retries  # retried per config

    def test_agent_failure_returns_empty(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        manifest = _make_manifest()

        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.stdout = ""
            result.returncode = 1
            return result

        monkeypatch.setattr("pipeline.plan.subprocess.run", mock_run)
        plans = generate_plans(manifest, {}, cfg)
        assert plans.plans == []

    def test_invalid_plan_skipped(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        src = _make_source(title="Valid Source")
        manifest = _make_manifest(src)

        agent_output = json.dumps([
            {"hash": src.hash, "title": "Valid Plan"},
            {"hash": src.hash, "language": "invalid_lang"},  # bad language
            {"hash": "unknown_hash", "title": "Unknown"},  # unknown hash
        ])

        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.stdout = agent_output
            result.returncode = 0
            return result

        monkeypatch.setattr("pipeline.plan.subprocess.run", mock_run)
        plans = generate_plans(manifest, {}, cfg)
        # Only the valid plan should be accepted
        assert len(plans.plans) == 1

    def test_saves_plans_json(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        src = _make_source(title="Test")
        manifest = _make_manifest(src)
        agent_output = json.dumps([{
            "hash": src.hash, "title": "Test", "language": "en",
            "template": "standard", "tags": [],
        }])

        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.stdout = agent_output
            result.returncode = 0
            return result

        monkeypatch.setattr("pipeline.plan.subprocess.run", mock_run)
        generate_plans(manifest, {}, cfg)

        plans_file = cfg.resolved_extract_dir / "plans.json"
        assert plans_file.exists()
        data = json.loads(plans_file.read_text())
        assert len(data) == 1


# ─── Main entry point ─────────────────────────────────────────────────────────

class TestPlanSources:
    def test_full_flow(self, tmp_path, monkeypatch):
        """Integration test: dedup → concept search → plan generation."""
        cfg = _make_config(tmp_path)
        src1 = _make_source(url="https://a.com", title="Article A", content="Content A " * 50)
        src2 = _make_source(url="https://b.com", title="Article B", content="Content B " * 50)
        manifest = _make_manifest(src1, src2)

        agent_output = json.dumps([
            {"hash": src1.hash, "title": "Article A", "language": "en", "template": "standard", "tags": []},
            {"hash": src2.hash, "title": "Article B", "language": "en", "template": "standard", "tags": []},
        ])

        def mock_qmd_search(queries, cfg, no_rerank=False):
            return {h: [ConceptMatch(concept="test-concept", score=0.7)] for h in queries}

        def mock_run_agent(*args, **kwargs):
            result = MagicMock()
            result.stdout = agent_output
            result.returncode = 0
            return result

        monkeypatch.setattr("pipeline.qmd.run_qmd_concept_search", mock_qmd_search)
        monkeypatch.setattr("pipeline.plan.subprocess.run", mock_run_agent)

        plans = plan_sources(manifest, cfg)
        assert len(plans.plans) == 2

    def test_empty_manifest(self, tmp_path):
        cfg = _make_config(tmp_path)
        manifest = Manifest(entries=[])
        plans = plan_sources(manifest, cfg)
        assert plans.plans == []

    def test_all_duplicates(self, tmp_path, monkeypatch):
        """If all sources are duplicates, return empty plans."""
        cfg = _make_config(tmp_path)
        content = "Duplicate content about quantum physics " * 20

        # Create existing vault source
        sources_dir = tmp_path / "04-Wiki" / "sources"
        sources_dir.mkdir(parents=True)
        (sources_dir / "existing.md").write_text("---\ntitle: E\n---\n" + content)

        src = _make_source(url="https://different.com", content=content)
        manifest = _make_manifest(src)
        plans = plan_sources(manifest, cfg)
        assert plans.plans == []

    def test_concept_search_continues_on_qmd_failure(self, tmp_path, monkeypatch):
        """If qmd fails, concept search returns empty matches but planning continues."""
        cfg = _make_config(tmp_path)
        src = _make_source(title="Article", content="Content " * 50)
        manifest = _make_manifest(src)

        agent_output = json.dumps([
            {"hash": src.hash, "title": "Article", "language": "en", "template": "standard", "tags": []},
        ])

        call_count = [0]

        def mock_run_subprocess(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # qmd call fails
                raise subprocess.TimeoutExpired(cmd="qmd", timeout=5)
            # agent call succeeds
            result = MagicMock()
            result.stdout = agent_output
            result.returncode = 0
            return result

        monkeypatch.setattr("pipeline.plan.subprocess.run", mock_run_subprocess)
        plans = plan_sources(manifest, cfg)
        assert len(plans.plans) == 1


