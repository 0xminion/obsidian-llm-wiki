"""Tests for proxy fallback and GitHub blob URL conversion."""

import pytest

from obsidian_llm_wiki.ingest.proxy import (
    get_proxy_url,
    make_client_kwargs,
    make_proxy_client_kwargs,
)


class TestProxyFallback:
    """Test the direct-first, proxy-fallback client kwargs."""

    def test_make_client_kwargs_no_proxy_by_default(self, monkeypatch):
        """make_client_kwargs does NOT include proxy by default."""
        monkeypatch.setenv("RESIDENTIAL_PROXY_URL", "socks5h://127.0.0.1:1080")
        kw = make_client_kwargs(timeout=30)
        assert "proxy" not in kw
        assert kw["timeout"] == 30

    def test_make_proxy_client_kwargs_includes_proxy(self, monkeypatch):
        """make_proxy_client_kwargs includes the proxy URL."""
        monkeypatch.setenv("RESIDENTIAL_PROXY_URL", "socks5h://127.0.0.1:1080")
        kw = make_proxy_client_kwargs(timeout=30)
        assert "proxy" in kw

    def test_make_proxy_client_kwargs_no_proxy(self, monkeypatch):
        """make_proxy_client_kwargs without env var has no proxy."""
        monkeypatch.delenv("RESIDENTIAL_PROXY_URL", raising=False)
        kw = make_proxy_client_kwargs(timeout=30)
        assert "proxy" not in kw

    def test_get_proxy_url_returns_configured(self, monkeypatch):
        monkeypatch.setenv("RESIDENTIAL_PROXY_URL", "http://proxy:8080")
        assert get_proxy_url() == "http://proxy:8080"

    def test_get_proxy_url_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("RESIDENTIAL_PROXY_URL", raising=False)
        assert get_proxy_url() is None


class TestGithubBlobToRaw:
    """Test GitHub blob URL → raw URL conversion."""

    @pytest.mark.parametrize(
        "blob_url,expected_raw",
        [
            (
                "https://github.com/owner/repo/blob/main/path/to/file.pdf",
                "https://raw.githubusercontent.com/owner/repo/main/path/to/file.pdf",
            ),
            (
                "https://github.com/TrepaOrg/trepa-research/blob/main/Trepa_Orthogonal_Precision_20260513_Blanco%26Chung%26Meka.pdf",
                "https://raw.githubusercontent.com/TrepaOrg/trepa-research/main/Trepa_Orthogonal_Precision_20260513_Blanco&Chung&Meka.pdf",
            ),
            (
                "https://github.com/user/project/blob/v1.0/docs/report.pdf",
                "https://raw.githubusercontent.com/user/project/v1.0/docs/report.pdf",
            ),
        ],
    )
    def test_blob_to_raw_conversion(self, blob_url: str, expected_raw: str):
        from obsidian_llm_wiki.ingest.extractors.pdf import _github_blob_to_raw

        assert _github_blob_to_raw(blob_url) == expected_raw

    @pytest.mark.parametrize(
        "url",
        [
            "https://raw.githubusercontent.com/owner/repo/main/file.pdf",
            "https://example.com/file.pdf",
            "https://github.com/owner/repo",
            "https://arxiv.org/pdf/2606.16852",
        ],
    )
    def test_non_blob_urls_unchanged(self, url: str):
        from obsidian_llm_wiki.ingest.extractors.pdf import _github_blob_to_raw

        assert _github_blob_to_raw(url) == url
