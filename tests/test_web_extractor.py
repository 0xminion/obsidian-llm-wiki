"""Tests for pipeline.extractors.web — HTML entity decoding, tag stripping, extract_web."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.extractors.web import _decode_html_entities, _strip_tags, extract_web


class TestDecodeHtmlEntities:
    """_decode_html_entities edge cases."""

    def test_standard_entities(self) -> None:
        """Standard named entities (&amp; &lt; &gt;) should decode correctly."""
        assert _decode_html_entities("a &amp; b") == "a & b"
        assert _decode_html_entities("x &lt; y") == "x < y"
        assert _decode_html_entities("x &gt; y") == "x > y"

    def test_quot_and_apos(self) -> None:
        """&quot; and &apos; / &#39; should decode."""
        assert _decode_html_entities("&quot;hello&quot;") == '"hello"'
        assert _decode_html_entities("it&apos;s") == "it's"
        assert _decode_html_entities("it&#39;s") == "it's"

    def test_nbsp(self) -> None:
        """&nbsp; should decode to a space."""
        assert _decode_html_entities("a&nbsp;b") == "a b"
        assert _decode_html_entities("a&#160;b") == "a b"

    def test_numeric_decimal_entities(self) -> None:
        """Numeric decimal entities (&#65;) should decode to characters."""
        assert _decode_html_entities("&#65;") == "A"
        assert _decode_html_entities("&#66;&#67;") == "BC"

    def test_numeric_hex_entities(self) -> None:
        """Numeric hex entities (&#x41;) should decode to characters."""
        assert _decode_html_entities("&#x41;") == "A"
        assert _decode_html_entities("&#x42;&#x43;") == "BC"

    def test_mixed_entities(self) -> None:
        """Mix of named and numeric entities should all decode."""
        text = "&lt;tag&gt; &#65; &amp; &#x42;"
        assert _decode_html_entities(text) == "<tag> A & B"

    def test_special_chars(self) -> None:
        """Special characters: ndash, mdash, hellipsis, etc."""
        assert _decode_html_entities("a&ndash;b") == "a–b"
        assert _decode_html_entities("a&mdash;b") == "a—b"
        assert _decode_html_entities("a&hellip;b") == "a…b"
        assert _decode_html_entities("&copy; 2024") == "© 2024"
        assert _decode_html_entities("&reg;") == "®"
        assert _decode_html_entities("&trade;") == "™"
        assert _decode_html_entities("&deg;C") == "°C"
        assert _decode_html_entities("&euro;100") == "€100"

    def test_no_entities(self) -> None:
        """Plain text with no entities should be returned unchanged."""
        assert _decode_html_entities("plain text") == "plain text"

    def test_empty_string(self) -> None:
        """Empty string should return empty."""
        assert _decode_html_entities("") == ""
    def test_invalid_hex_entity_no_crash(self) -> None:
        """Invalid hex entity with overflow value should not crash.

        &#x999999999; is far beyond the Unicode range (> 0x10FFFF).
        Previously chr() would raise ValueError; now the function catches
        the error and returns the original entity text unchanged.
        """
        result = _decode_html_entities("&#x999999999;")
        assert result == "&#x999999999;"

    def test_invalid_decimal_entity_no_crash(self) -> None:
        """Large decimal entity that's valid for regex but too large for chr().

        &#99999999999; exceeds 0x10FFFF. The function catches the ValueError
        and returns the original entity text unchanged.
        """
        result = _decode_html_entities("&#99999999999;")
        assert result == "&#99999999999;"


class TestStripTags:
    """_strip_tags basic HTML stripping."""

    def test_basic_html_stripping(self) -> None:
        """Simple HTML tags should be stripped to text."""
        html = "<p>Hello <b>world</b></p>"
        result = _strip_tags(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result
        assert ">" not in result

    def test_strips_script_tags(self) -> None:
        """Script content should be removed entirely."""
        html = "<script>alert('xss')</script><p>content</p>"
        result = _strip_tags(html)
        assert "alert" not in result
        assert "xss" not in result
        assert "content" in result

    def test_strips_style_tags(self) -> None:
        """Style content should be removed entirely."""
        html = "<style>body { color: red; }</style><p>visible</p>"
        result = _strip_tags(html)
        assert "color" not in result
        assert "red" not in result
        assert "visible" in result

    def test_strips_comments(self) -> None:
        """HTML comments should be removed."""
        html = "<!-- a comment --><p>text</p>"
        result = _strip_tags(html)
        assert "a comment" not in result
        assert "text" in result

    def test_block_tags_add_newlines(self) -> None:
        """Block-level tags should produce newlines."""
        html = "<div>line1</div><div>line2</div>"
        result = _strip_tags(html)
        assert "line1" in result
        assert "line2" in result
        # Should have a newline between them
        assert "\n" in result

    def test_decodes_entities_in_stripped(self) -> None:
        """_strip_tags should decode HTML entities in the output."""
        html = "<p>&amp; &lt; &gt;</p>"
        result = _strip_tags(html)
        assert "&" in result
        assert "<" in result
        assert ">" in result

    def test_empty_html(self) -> None:
        """Empty string should return empty."""
        assert _strip_tags("") == ""

    def test_nested_tags(self) -> None:
        """Nested tags should be fully stripped."""
        html = "<div><p><span>nested</span></p></div>"
        result = _strip_tags(html)
        assert "nested" in result
        assert "<" not in result


class TestExtractWeb:
    """extract_web with mocked subprocess."""

    def test_extract_defuddle_success(self) -> None:
        """extract_web should return IngestedSource when defuddle succeeds."""
        fake_json = json.dumps({
            "title": "Test Article",
            "contentMarkdown": "# Test Article\n\nThis is the content.",
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_json
        mock_result.stderr = ""

        with patch("pipeline.extractors.web.subprocess.run", return_value=mock_result):
            result = extract_web("https://example.com/article")

        assert result.title == "Test Article"
        assert "This is the content." in result.content

    def test_extract_defuddle_fallback_content_key(self) -> None:
        """When contentMarkdown is empty, should fall back to 'content' key."""
        fake_json = json.dumps({
            "title": "Fallback Title",
            "contentMarkdown": "",
            "content": "<p>Fallback content</p>",
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_json
        mock_result.stderr = ""

        with patch("pipeline.extractors.web.subprocess.run", return_value=mock_result):
            result = extract_web("https://example.com/article")

        assert result.title == "Fallback Title"
        assert "Fallback content" in result.content

    def test_extract_defuddle_empty_content_raises(self) -> None:
        """defuddle returning empty content should trigger fallback strategies,
        and if all fail, raise RuntimeError."""
        fake_json = json.dumps({
            "title": "Empty",
            "contentMarkdown": "",
            "content": "",
        })
        # All three strategies fail: defuddle empty, curl empty, wayback empty
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_json
        mock_result.stderr = ""

        # For curl fallbacks: returncode != 0 and no stdout
        mock_curl_fail = MagicMock()
        mock_curl_fail.returncode = 1
        mock_curl_fail.stdout = ""
        mock_curl_fail.stderr = "Connection failed"

        with patch(
            "pipeline.extractors.web.subprocess.run",
            side_effect=[mock_result, mock_curl_fail, mock_curl_fail],
        ), pytest.raises(RuntimeError, match="All extraction strategies failed"):
            extract_web("https://example.com/empty")

    def test_extract_defuddle_nonzero_returncode_falls_through(self) -> None:
        """defuddle exiting non-zero should trigger curl fallback."""
        # defuddle fails
        mock_defuddle_fail = MagicMock()
        mock_defuddle_fail.returncode = 1
        mock_defuddle_fail.stdout = ""
        mock_defuddle_fail.stderr = "defuddle not found"

        # curl succeeds with HTML
        mock_curl_ok = MagicMock()
        mock_curl_ok.returncode = 0
        mock_curl_ok.stdout = "<html><head><title>Curl Title</title></head><body><p>Curl content here</p></body></html>"
        mock_curl_ok.stderr = ""

        with patch(
            "pipeline.extractors.web.subprocess.run",
            side_effect=[mock_defuddle_fail, mock_curl_ok],
        ):
            result = extract_web("https://example.com/article")

        assert result.title == "Curl Title"
        assert "Curl content here" in result.content

    def test_extract_all_strategies_fail(self) -> None:
        """When all three strategies fail, RuntimeError should be raised."""
        mock_fail = MagicMock()
        mock_fail.returncode = 1
        mock_fail.stdout = ""
        mock_fail.stderr = "failed"

        with patch(
            "pipeline.extractors.web.subprocess.run",
            side_effect=[mock_fail, mock_fail, mock_fail],
        ), pytest.raises(RuntimeError, match="All extraction strategies failed"):
            extract_web("https://example.com/broken")
