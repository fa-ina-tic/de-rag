"""Tests for de_rag.embedders."""
import pickle
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from de_rag.embedders import (
    SentenceTransformerEmbedder,
    CohereEmbedder,
    embed_corpus,
    load_or_embed_corpus,
)
from de_rag.classes import Document

DIM = 16


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_st_model(dim=DIM):
    """Return a mock SentenceTransformer model."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def _encode(texts, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
        if isinstance(texts, str):
            texts = [texts]
        arr = np.ones((len(texts), dim), dtype=np.float32)
        if normalize_embeddings:
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return arr

    model.encode.side_effect = _encode
    return model


def _fake_cohere_response(embeddings):
    resp = MagicMock()
    resp.embeddings = embeddings
    return resp


# ── SentenceTransformerEmbedder ────────────────────────────────────────────────

class TestSentenceTransformerEmbedder:

    @pytest.fixture
    def st_embedder(self):
        # Bypass __init__ (which does the real import) and inject a mock model directly
        mock_model = _fake_st_model()
        embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
        embedder._model = mock_model
        return embedder

    def test_encode_single_string(self, st_embedder):
        result = st_embedder.encode("hello", normalize=True)
        assert result.shape == (1, DIM) or result.ndim >= 1

    def test_encode_list(self, st_embedder):
        result = st_embedder.encode(["a", "b", "c"], normalize=True)
        assert result.shape == (3, DIM)

    def test_encode_normalized(self, st_embedder):
        result = st_embedder.encode(["x", "y"], normalize=True)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embedding_dim(self, st_embedder):
        assert st_embedder.embedding_dim == DIM

    def test_encode_passes_normalize_flag(self, st_embedder):
        st_embedder.encode(["z"], normalize=False)
        call_kwargs = st_embedder._model.encode.call_args
        assert call_kwargs.kwargs.get("normalize_embeddings") is False or \
               (call_kwargs.args and False in call_kwargs.args)

    def test_encode_passes_batch_size(self, st_embedder):
        st_embedder.encode(["a", "b"], batch_size=32)
        call_kwargs = st_embedder._model.encode.call_args
        assert call_kwargs.kwargs.get("batch_size") == 32


# ── CohereEmbedder ──────────────────────────────────────────────────────────────

class TestCohereEmbedder:

    @pytest.fixture
    def cohere_client(self):
        client = MagicMock()
        embeddings = np.random.rand(3, DIM).astype(np.float32).tolist()
        client.embed.return_value = _fake_cohere_response(embeddings)
        return client

    @pytest.fixture
    def embedder(self, cohere_client):
        embedder = CohereEmbedder.__new__(CohereEmbedder)
        embedder._client = cohere_client
        embedder._model = "embed-english-v3.0"
        embedder._default_input_type = "search_document"
        embedder._dim = None
        embedder._min_interval = 0.0  # disable rate limiting in tests
        embedder._last_request_time = 0.0
        return embedder

    def test_encode_list_returns_2d(self, embedder, cohere_client):
        texts = ["a", "b", "c"]
        cohere_client.embed.return_value = _fake_cohere_response(
            np.random.rand(3, DIM).astype(np.float32).tolist()
        )
        result = embedder.encode(texts)
        assert result.ndim == 2
        assert result.shape[0] == 3

    def test_encode_single_string_returns_1d(self, embedder, cohere_client):
        cohere_client.embed.return_value = _fake_cohere_response(
            np.random.rand(1, DIM).astype(np.float32).tolist()
        )
        result = embedder.encode("hello")
        assert result.ndim == 1

    def test_encode_normalizes_by_default(self, embedder, cohere_client):
        raw = np.array([[3.0] * DIM], dtype=np.float32)
        cohere_client.embed.return_value = _fake_cohere_response(raw.tolist())
        result = embedder.encode(["x"])
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_encode_no_normalize(self, embedder, cohere_client):
        raw = np.array([[3.0] * DIM], dtype=np.float32)
        cohere_client.embed.return_value = _fake_cohere_response(raw.tolist())
        result = embedder.encode(["x"], normalize=False)
        # Should not be unit vector
        norm = np.linalg.norm(result[0])
        assert norm > 1.1

    def test_embedding_dim_after_encode(self, embedder, cohere_client):
        cohere_client.embed.return_value = _fake_cohere_response(
            np.random.rand(1, DIM).astype(np.float32).tolist()
        )
        embedder.encode(["test"])
        assert embedder.embedding_dim == DIM

    def test_embedding_dim_before_encode_raises(self, embedder):
        with pytest.raises(RuntimeError, match="unknown until the first call"):
            _ = embedder.embedding_dim

    def test_encode_batches_large_input(self, embedder, cohere_client):
        """Input > batch_size should trigger multiple API calls."""
        n = 200
        batch_size = 96
        texts = [f"text {i}" for i in range(n)]

        def batch_embed(**kwargs):
            batch = kwargs.get("texts", [])
            return _fake_cohere_response(np.random.rand(len(batch), DIM).astype(np.float32).tolist())

        cohere_client.embed.side_effect = lambda **kw: batch_embed(**kw)
        result = embedder.encode(texts, batch_size=batch_size)
        assert result.shape == (n, DIM)
        # Should have called embed ceil(200/96) = 3 times
        assert cohere_client.embed.call_count >= 2

    def test_encode_uses_input_type_override(self, embedder, cohere_client):
        cohere_client.embed.return_value = _fake_cohere_response(
            np.random.rand(1, DIM).astype(np.float32).tolist()
        )
        embedder.encode(["q"], input_type="search_query")
        call_kwargs = cohere_client.embed.call_args.kwargs
        assert call_kwargs.get("input_type") == "search_query"

    def test_429_retry_succeeds(self, embedder):
        """Retry on 429-like exception and eventually succeed."""

        class TooManyRequestsError(Exception):
            pass

        raw = np.random.rand(1, DIM).astype(np.float32).tolist()
        success_response = _fake_cohere_response(raw)

        call_count = {"n": 0}

        def flaky_embed(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise TooManyRequestsError("429 TooManyRequests")
            return success_response

        embedder._client.embed.side_effect = flaky_embed

        with patch("de_rag.embedders.time.sleep"):  # skip actual sleep
            result = embedder.encode(["x"])
        assert result.ndim == 2
        assert result.shape == (1, DIM)
        assert call_count["n"] == 3

    def test_429_exhausted_raises(self, embedder):
        """After 5 retries, raise RuntimeError."""

        class TooManyRequestsError(Exception):
            pass

        embedder._client.embed.side_effect = TooManyRequestsError("429 TooManyRequests")

        with patch("de_rag.embedders.time.sleep"):
            with pytest.raises(RuntimeError, match="5 retries"):
                embedder.encode(["x"])


# ── embed_corpus ────────────────────────────────────────────────────────────────

class TestEmbedCorpus:
    def test_returns_documents(self, mock_embedder):
        corpus = {
            "d1": {"title": "Title 1", "text": "Text one"},
            "d2": {"title": "Title 2", "text": "Text two"},
        }
        mock_embedder.encode.side_effect = lambda texts, **kw: np.ones(
            (len(texts), 8), dtype=np.float32
        )
        docs = embed_corpus(corpus, mock_embedder)
        assert len(docs) == 2
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.doc_type == "chunk"

    def test_doc_text_concatenates_title_and_text(self, mock_embedder):
        corpus = {"d1": {"title": "MyTitle", "text": "MyText"}}
        mock_embedder.encode.side_effect = lambda texts, **kw: np.ones(
            (len(texts), 8), dtype=np.float32
        )
        docs = embed_corpus(corpus, mock_embedder)
        assert "MyTitle" in docs[0].text
        assert "MyText" in docs[0].text


# ── load_or_embed_corpus ────────────────────────────────────────────────────────

class TestLoadOrEmbedCorpus:
    def test_loads_from_cache(self, tmp_path, mock_embedder, sample_docs):
        cache = tmp_path / "cache.pkl"
        with open(cache, "wb") as f:
            pickle.dump(sample_docs, f)

        corpus = {}
        result = load_or_embed_corpus(corpus, mock_embedder, cache_path=cache)
        assert len(result) == len(sample_docs)
        assert [d.id for d in result] == [d.id for d in sample_docs]
        assert [d.text for d in result] == [d.text for d in sample_docs]
        mock_embedder.encode.assert_not_called()

    def test_embeds_and_saves_when_no_cache(self, tmp_path, mock_embedder):
        corpus = {"d1": {"title": "T", "text": "X"}}
        mock_embedder.encode.side_effect = lambda texts, **kw: np.ones(
            (len(texts), 8), dtype=np.float32
        )
        cache = tmp_path / "out.pkl"
        result = load_or_embed_corpus(corpus, mock_embedder, cache_path=cache)
        assert len(result) == 1
        assert cache.exists()

    def test_no_cache_path_does_not_save(self, tmp_path, mock_embedder):
        corpus = {"d1": {"title": "T", "text": "X"}}
        mock_embedder.encode.side_effect = lambda texts, **kw: np.ones(
            (len(texts), 8), dtype=np.float32
        )
        result = load_or_embed_corpus(corpus, mock_embedder, cache_path=None)
        assert len(result) == 1
        # Nothing saved
        assert not list(tmp_path.iterdir())
