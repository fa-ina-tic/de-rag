"""Tests for de_rag.retriever (SimpleRetriever, NonNegativeRetriever, NERRetriever)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from de_rag.classes import Document, RetrievalResult
from de_rag.index import FaissIndex
from de_rag.retriever import SimpleRetriever, NonNegativeRetriever

DIM = 8


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_index(n=5, dim=DIM):
    rng = np.random.default_rng(7)
    vecs = rng.random((n, dim)).astype(np.float32)
    docs = [
        Document(id=str(i), text=f"doc {i}", embedding=vecs[i], doc_type="chunk")
        for i in range(n)
    ]
    idx = FaissIndex(dim=dim, factory_string="Flat", metric="l2")
    idx.add_documents(docs)
    return idx


def _make_embedder(dim=DIM, value=None):
    rng = np.random.default_rng(99)
    embedder = MagicMock()

    def _encode(texts, **kwargs):
        if isinstance(texts, str):
            v = rng.random(dim).astype(np.float32) if value is None else np.full(dim, value, dtype=np.float32)
            # normalize
            v = v / (np.linalg.norm(v) + 1e-9)
            return v
        n = len(texts)
        arr = rng.random((n, dim)).astype(np.float32) if value is None else np.full((n, dim), value, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-9)

    embedder.encode.side_effect = _encode
    embedder.embedding_dim = dim
    return embedder


# ── SimpleRetriever ────────────────────────────────────────────────────────────

class TestSimpleRetriever:

    @pytest.fixture
    def retriever(self):
        idx = _make_index()
        emb = _make_embedder()
        return SimpleRetriever(embedder=emb, index=idx)

    def test_retrieve_returns_list_of_lists(self, retriever):
        results = retriever.retrieve("what is python?", top_k=3)
        assert isinstance(results, list)
        assert isinstance(results[0], list)

    def test_retrieve_top_k(self, retriever):
        results = retriever.retrieve("query", top_k=3)
        assert len(results[0]) == 3

    def test_retrieve_results_are_retrieval_result(self, retriever):
        results = retriever.retrieve("query", top_k=2)
        for rr in results[0]:
            assert isinstance(rr, RetrievalResult)

    def test_retrieve_source_label(self, retriever):
        results = retriever.retrieve("query", top_k=2, source_label="my_retriever")
        for rr in results[0]:
            assert rr.source == "my_retriever"

    def test_retrieve_scores_are_float(self, retriever):
        results = retriever.retrieve("query", top_k=2)
        for rr in results[0]:
            assert isinstance(rr.score, float)

    def test_retrieve_no_index_raises(self):
        emb = _make_embedder()
        r = SimpleRetriever(embedder=emb, index=None)
        with pytest.raises(ValueError, match="No Index provided"):
            r.retrieve("query")

    def test_retrieve_with_index_override(self):
        emb = _make_embedder()
        r = SimpleRetriever(embedder=emb, index=None)
        idx = _make_index()
        results = r.retrieve("query", top_k=2, index=idx)
        assert len(results[0]) == 2

    def test_batch_query_alias(self, retriever):
        results = retriever.batch_query(["q1", "q2"], top_k=2)
        assert len(results) == 2

    def test_retrieve_list_of_queries(self, retriever):
        results = retriever.retrieve(["q1", "q2", "q3"], top_k=2)
        assert len(results) == 3

    def test_preprocess_returns_ndarray(self, retriever):
        result = retriever._preprocess("hello world")
        assert isinstance(result, np.ndarray)


# ── NonNegativeRetriever ───────────────────────────────────────────────────────

class TestNonNegativeRetriever:

    @pytest.fixture
    def retriever(self):
        idx = _make_index()
        # Use an embedder that returns mixed positive/negative values
        emb = MagicMock()
        rng = np.random.default_rng(13)

        def _encode(texts, **kwargs):
            if isinstance(texts, str):
                v = rng.random(DIM).astype(np.float32) - 0.5  # [-0.5, 0.5]
                return v
            n = len(texts)
            return rng.random((n, DIM)).astype(np.float32) - 0.5

        emb.encode.side_effect = _encode
        emb.embedding_dim = DIM
        return NonNegativeRetriever(embedder=emb, index=idx)

    def test_preprocess_has_no_negatives(self, retriever):
        result = retriever._preprocess("test query")
        assert np.all(result >= 0), "NonNegativeRetriever should zero out negatives"

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("query", top_k=2)
        assert len(results[0]) > 0

    def test_negatives_zeroed_not_flipped(self, retriever):
        """Negative values should become 0, not |value|."""
        fixed_emb = MagicMock()
        fixed_emb.encode.return_value = np.array([-1.0, 0.5, -0.3, 1.0, -0.2, 0.1, 0.8, -0.4], dtype=np.float32)
        fixed_emb.embedding_dim = DIM
        retriever.embedder = fixed_emb
        result = retriever._preprocess("q")
        expected_pos = np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.1, 0.8, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected_pos)


# ── NERRetriever ───────────────────────────────────────────────────────────────

class TestNERRetriever:
    """Test NERRetriever with a mocked GLiNER model."""

    @pytest.fixture
    def ner_retriever(self):
        from de_rag.retriever import NERRetriever

        idx = _make_index()
        emb = _make_embedder()

        mock_ner = MagicMock()
        mock_ner.predict_entities.return_value = [
            {"text": "Python", "label": "organization"},
            {"text": "2023", "label": "date"},
        ]

        with patch.dict("sys.modules", {"gliner": MagicMock(GLiNER=MagicMock(from_pretrained=MagicMock(return_value=mock_ner)))}):
            retriever = NERRetriever.__new__(NERRetriever)
            retriever.embedder = emb
            retriever.index = idx
            retriever.ner_model = mock_ner
        return retriever

    def test_mask_query_replaces_entities(self, ner_retriever):
        query = "What did Python do in 2023?"
        masked = ner_retriever._mask_query(query)
        # Should have variants where entities are replaced with space
        assert any("Python" not in q for q in masked[:-1])  # all but last are masked variants

    def test_mask_query_appends_entity_only_query(self, ner_retriever):
        query = "Python released in 2023"
        masked = ner_retriever._mask_query(query)
        # Last element should be entity-only query
        last = masked[-1]
        assert "Python" in last or "2023" in last

    def test_mask_query_no_entities_returns_original(self):
        from de_rag.retriever import NERRetriever

        idx = _make_index()
        emb = _make_embedder()

        mock_ner = MagicMock()
        mock_ner.predict_entities.return_value = []  # no entities found

        with patch.dict("sys.modules", {"gliner": MagicMock(GLiNER=MagicMock(from_pretrained=MagicMock(return_value=mock_ner)))}):
            retriever = NERRetriever.__new__(NERRetriever)
            retriever.embedder = emb
            retriever.index = idx
            retriever.ner_model = mock_ner

        query = "what is the meaning of life?"
        masked = retriever._mask_query(query)
        # With no entities: queries = [original] then append empty entity string
        assert query in masked

    def test_preprocess_returns_stacked_embeddings(self, ner_retriever):
        result = ner_retriever._preprocess("Python released in 2023")
        assert result.ndim == 2  # stacked: (n_queries, dim)

    def test_ner_missing_raises(self):
        from de_rag.retriever import NERRetriever
        import sys

        with patch.dict("sys.modules", {"gliner": None}):
            # Force ImportError by removing gliner
            with pytest.raises((AssertionError, ImportError, Exception)):
                NERRetriever(embedder=_make_embedder(), index=_make_index())
