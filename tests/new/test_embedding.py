"""Regression tests for embedding module gating.

EMBEDDINGS_ENABLED=false (the default in CI) must make all
embedding functions short-circuit without any network I/O.
"""
from __future__ import annotations

from unittest import mock

from obsidian_llm_wiki.synth.embedding import embed_text


class TestEmbeddingGating:
    """Embedding calls must be no-ops when EMBEDDINGS_ENABLED is False."""

    def test_embed_text_returns_none_when_disabled(self):
        """embed_text must return None immediately when EMBEDDINGS_ENABLED is False."""
        with mock.patch.dict(
            "os.environ", {"EMBEDDINGS_ENABLED": "false"}, clear=False
        ):
            result = embed_text("some text to embed")
        assert result is None

    def test_embed_text_returns_none_on_ollama_connection_error(self):
        """Connection errors from Ollama must return None, not raise."""
        with mock.patch.dict(
            "os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False
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
        with mock.patch.dict(
            "os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False
        ), mock.patch(
            "obsidian_llm_wiki.synth.embedding.httpx.Client"
        ) as mock_client_cls:
            err_resp = mock.Mock(status_code=500)
            err_resp.raise_for_status.side_effect = Exception("server error")
            mock_client_cls.return_value.__enter__ = mock.Mock(
                return_value=mock.Mock(get=mock.Mock(return_value=err_resp))
            )
            mock_client_cls.return_value.__exit__ = mock.Mock(return_value=False)
            mock_client_cls.return_value.post = mock.Mock(return_value=err_resp)
            result = embed_text("test")
        assert result is None

    def test_embed_text_uses_vault_environment_loaded_after_import(self):
        """Configured model and host must not be frozen at module import."""
        response = mock.Mock(status_code=200)
        response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        client = mock.Mock()
        client.post.return_value = response
        client.__enter__ = mock.Mock(return_value=client)
        client.__exit__ = mock.Mock(return_value=False)
        with (
            mock.patch.dict(
                "os.environ",
                {
                    "EMBEDDINGS_ENABLED": "true",
                    "EMBEDDING_MODEL": "qwen3-embedding:0.6b",
                    "LLM_HOST": "http://localhost:11435/",
                },
                clear=False,
            ),
            mock.patch("obsidian_llm_wiki.synth.embedding.httpx.Client", return_value=client),
        ):
            assert embed_text("bilingual concept") == [0.1, 0.2]

        client.post.assert_called_once_with(
            "http://localhost:11435/api/embed",
            json={"model": "qwen3-embedding:0.6b", "input": "bilingual concept"},
        )
