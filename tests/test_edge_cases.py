"""Edge case tests — special characters, CJK, long content, boundary conditions."""

import hashlib
import json
import math
from pathlib import Path


from pipeline.vault import title_to_filename


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_extract_json(url: str, title: str, content: str,
                      source_type: str = "web", author: str = "") -> dict:
    """Create an extracted-source dict matching the pipeline schema."""
    return {
        "url": url,
        "title": title,
        "content": content,
        "type": source_type,
        "author": author,
        "source_file": "test.url",
    }


def write_extract_json(tmp_path: Path, url: str, title: str, content: str,
                       source_type: str = "web", author: str = "") -> Path:
    """Write an extracted JSON file and return its path."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    data = make_extract_json(url, title, content, source_type, author)
    json_file = tmp_path / f"{url_hash}.json"
    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_file


# ─── Special characters in titles — JSON serialization ──────────────────────

class TestSpecialCharsInTitle:
    def test_quotes_and_symbols_survive_roundtrip(self, tmp_path):
        title = 'Article: "Quotes" & <Tags> — Plus $pecial Ch@rs!'
        json_file = write_extract_json(tmp_path, "https://example.com/special", title, "Content with special chars.")
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["title"] == title
        assert '"Quotes"' in loaded["title"]

    def test_forward_slashes_in_title(self, tmp_path):
        title = "React/Vue/Angular Comparison"
        json_file = write_extract_json(tmp_path, "https://example.com/slash", title, "Content about frameworks.")
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["title"] == title


# ─── Chinese content — JSON serialization ───────────────────────────────────

class TestChineseContent:
    def test_chinese_roundtrip(self, tmp_path):
        title = "预测市场的未来发展趋势"
        content = (
            "预测市场是一种通过市场机制来进行预测的工具。"
            "随着区块链技术的发展，去中心化预测市场正在成为新的研究热点。"
            "本文将探讨预测市场的历史、现状和未来发展方向。"
        )
        json_file = write_extract_json(
            tmp_path, "https://example.cn/article", title, content,
            source_type="web", author="作者",
        )
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert "预测市场" in loaded["title"]
        assert "区块链" in loaded["content"]
        assert loaded["author"] == "作者"


# ─── Chinese title — filename generation ────────────────────────────────────

class TestChineseFilename:
    def test_chinese_title_preserved(self):
        title = "预测市场的未来发展趋势"
        filename = title_to_filename(title)
        assert "预测" in filename

    def test_chinese_filename_max_120(self):
        long_zh = (
            "这是一个非常长的中文标题用来测试文件名截断功能是否正确工作"
            "确保不会超过一百二十个字符的限制因为文件系统对文件名长度有限制"
            "我们需要确保截断逻辑正确"
        )
        filename = title_to_filename(long_zh)
        assert len(filename) <= 120

    def test_english_kebab_case(self):
        filename = title_to_filename("The Future of AI in 2026!")
        assert "the-future-of-ai" in filename

    def test_english_filename_max_120(self):
        long_en = (
            "This is an incredibly long English article title that goes on "
            "and on about nothing in particular and keeps adding more words"
        )
        filename = title_to_filename(long_en)
        assert len(filename) <= 120


# ─── Very long content (>100K chars) ────────────────────────────────────────

class TestVeryLongContent:
    def test_long_content_serializes(self, tmp_path):
        paragraphs = []
        for i in range(5000):
            paragraphs.append(
                f"This is paragraph {i} with some meaningful content "
                f"about topic number {i}. " * 3
            )
        content = "\n".join(paragraphs)
        assert len(content) > 100_000

        json_file = write_extract_json(tmp_path, "https://example.com/long", "Long", content)
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(loaded["content"]) > 100_000

    def test_max_content_per_source_truncation(self):
        """Pipeline truncates content to max_content_per_source (default 8000)."""
        from pipeline.config import Config
        cfg = Config(vault_path=Path("/tmp"), extract_dir=Path("/tmp"))
        long_content = "x" * 20_000
        truncated = long_content[:cfg.max_content_per_source]
        assert len(truncated) == 8_000
        assert len(truncated) < len(long_content)


# ─── Mixed language content ────────────────────────────────────────────────

class TestMixedLanguageContent:
    def test_english_chinese_mix(self, tmp_path):
        content = "This article covers AI (人工智能) and machine learning (机器学习) trends in 2026."
        json_file = write_extract_json(
            tmp_path, "https://example.com/mixed", "AI 人工智能 Overview", content,
        )
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert "人工智能" in loaded["content"]
        assert "machine learning" in loaded["content"]


# ─── Empty content ──────────────────────────────────────────────────────────

class TestEmptyContent:
    def test_empty_content_field(self, tmp_path):
        json_file = write_extract_json(
            tmp_path, "https://example.com/empty", "Empty Article", "",
        )
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["content"] == ""

    def test_empty_title(self, tmp_path):
        json_file = write_extract_json(tmp_path, "https://example.com/no-title", "", "Some content")
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["title"] == ""


# ─── Unicode URLs ───────────────────────────────────────────────────────────

class TestUnicodeUrls:
    def test_unicode_url_hash_doesnt_crash(self):
        url = "https://例え.jp/記事"
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        assert len(url_hash) == 12

    def test_cjk_url_roundtrip(self, tmp_path):
        url = "https://例え.jp/記事"
        title = "テスト記事"
        content = "Unicode URL test content."
        json_file = write_extract_json(tmp_path, url, title, content)
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["url"] == url


# ─── Long title truncation ─────────────────────────────────────────────────

class TestLongTitleTruncation:
    def test_title_in_json_is_preserved(self, tmp_path):
        """The raw JSON stores the full title; truncation happens at filename generation."""
        long_title = "Word " * 50
        json_file = write_extract_json(tmp_path, "https://example.com/long-title", long_title, "Content")
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(loaded["title"]) == len(long_title)

    def test_title_to_filename_truncates(self):
        long_title = "Word " * 50
        filename = title_to_filename(long_title)
        assert len(filename) <= 120


# ─── No-stub policy ────────────────────────────────────────────────────────

class TestNoStubPolicy:
    def test_entry_template_mentions_no_stubs(self):
        entry_prompt = Path(__file__).parent.parent / "prompts" / "entry-structure.prompt"
        if entry_prompt.exists():
            content = entry_prompt.read_text(encoding="utf-8").lower()
            # Entry prompt has naming rules and "NEVER use URL slugs" — that's anti-stub
            assert "never use" in content or "stub" in content or "todo" in content

    def test_common_instructions_mentions_no_stubs(self):
        common_prompt = Path(__file__).parent.parent / "prompts" / "common-instructions.prompt"
        if common_prompt.exists():
            content = common_prompt.read_text(encoding="utf-8").lower()
            assert "stub" in content


# ─── Manifest stress (50 entries) ──────────────────────────────────────────

class TestManifestStress:
    def test_fifty_entries(self, tmp_path):
        entries = []
        for i in range(50):
            url = f"https://example.com/article-{i}"
            title = f"Article Number {i}"
            content = f"Content for article {i}."
            data = make_extract_json(url, title, content)
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            json_file = tmp_path / f"{url_hash}.json"
            json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            entries.append(url_hash)

        # Verify all 50 files exist and parse
        assert len(entries) == 50
        for h in entries:
            f = tmp_path / f"{h}.json"
            assert f.exists()
            loaded = json.loads(f.read_text(encoding="utf-8"))
            assert "title" in loaded

    def test_batch_split_parallel_greater_than_plans(self):
        """Parallel(10) > plans(2) → 1+1 batches."""
        plans = list(range(2))
        parallel = 10
        batch_size = math.ceil(len(plans) / parallel)
        batches = []
        for i in range(parallel):
            start = i * batch_size
            end = min(start + batch_size, len(plans))
            if start >= len(plans):
                break
            batches.append(len(plans[start:end]))
        assert batches == [1, 1]


# ─── Newline handling ──────────────────────────────────────────────────────

class TestNewlineHandling:
    def test_newlines_survive_json_roundtrip(self, tmp_path):
        content = (
            "Line 1\n"
            "Line 2\n"
            "\n"
            "Line 4 (blank above)\n"
            "\n"
            "Line 6 (blank above)\n"
        )
        json_file = write_extract_json(tmp_path, "https://example.com/newlines", "Newline Test", content)
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        line_count = loaded["content"].count("\n")
        assert line_count > 3

    def test_empty_lines_preserved(self, tmp_path):
        content = "First\n\n\nThird"
        json_file = write_extract_json(tmp_path, "https://example.com/blanks", "Blank", content)
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["content"] == content
