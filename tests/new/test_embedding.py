"""Regression tests for embedding module gating.

EMBEDEDDINGS_ENABLED=false (the default in CI) must make all
embedding functions short-circuit without any network I/O.
"""
from __future__ import annotations

from unittest import mock

from obsidian_llm_wiki.synth.embedding import embed_text


class TestEmbeddingGating:
    """Embedding calls must be no-ops when _EMBEDDINGS_ENABLED is False."""

    def test_embed_text_returns_none_when_disabled(self):
        """embed_text must return None immediately when _EMBEDDINGS_ENABLED is False."""
        with mock.patch(
            "obsidian_llm_wiki.synth.embedding._EMBEDDINGS_ENABLED", False
        ):
            result = embed_text("some text to embed")
        assert result is None

    def test_embed_text_returns_none_on_ollama_connection_error(self):
        """Connection errors from Ollama must return None, not raise."""
        with mock.patch(
            "obsidian_llm_wiki.synth.embedding._EMBEDDINGS_ENABLED", True
        ), mock.patch(
            "obsidian_llm_wiki.synth.embedding.httpx.Client"
        ) as mock_client_cls:
            mock_client_cls.return_value.__enter__ = mock.Mock(
                side_effect=OSError("connection refused")
            )
            mock_client_cls.return_value.__exit__ = mock.Mock(return_value=False)
            result = embed_text("test")
        assert result is None

    def test_embed_text_returns_none_on_ollama_500(self):
        """Ollama returning 500 must return None."""
        with mock.patch(
            "obsidian_llm_wiki.synth.embedding._EMBEDDINGS_ENABLED", True
        ), mock.patch(
            "obsidian_llm_wiki.synth.embedding.httpx.Client"
        ) as mock_client_cls:
            err_resp = mock.Mock(status_code=500)
            err_resp.raise_for_status.side_effect = Exception("server error")
            mock_client_cls.return_value.__enter__ = mock.Mock(
                return_value=mock.Mock(get=mock.Mock(return_value=err_resp))
            )
            mock_client_cls.return_value.__exit__ = mock.Mock(return_value=False)
            result = embed_text("test")
        assert result is None
