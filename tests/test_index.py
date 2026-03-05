"""Tests for de_rag.index (FaissIndex, chunk_texts, HNSWIndex, build)."""
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from de_rag.index import FaissIndex, HNSWIndex, chunk_texts, available_index_classes, build
from de_rag.classes import Document

DIM = 8


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rand_docs(n, dim=DIM, rng=None):
    rng = rng or np.random.default_rng(1)
    vecs = rng.random((n, dim)).astype(np.float32)
    return [
        Document(id=str(i), text=f"doc {i}", embedding=vecs[i], doc_type="chunk")
        for i in range(n)
    ]


# ── FaissIndex ─────────────────────────────────────────────────────────────────

class TestFaissIndex:

    # Construction
    def test_create_flat_l2(self):
        idx = FaissIndex(dim=DIM, factory_string="Flat", metric="l2")
        assert idx.dim == DIM
        assert len(idx) == 0

    def test_create_flat_ip(self):
        idx = FaissIndex(dim=DIM, factory_string="Flat", metric="ip")
        assert idx.dim == DIM

    def test_create_flat_cosine(self):
        # cosine aliases to inner product
        idx = FaissIndex(dim=DIM, factory_string="Flat", metric="cosine")
        assert idx.dim == DIM

    def test_create_hnsw(self):
        idx = FaissIndex(dim=DIM, factory_string="HNSW32", metric="l2")
        assert idx.dim == DIM

    def test_docs_initially_empty(self):
        idx = FaissIndex(dim=DIM, factory_string="Flat")
        assert idx.docs == []

    # Adding documents
    def test_add_documents(self):
        idx = FaissIndex(dim=DIM, factory_string="Flat")
        docs = _rand_docs(5)
        idx.add_documents(docs)
        assert len(idx) == 5
        assert len(idx.docs) == 5

    def test_add_documents_dimension_mismatch_raises(self):
        idx = FaissIndex(dim=DIM, factory_string="Flat")
        bad_docs = [Document(id="x", text="t", embedding=np.zeros(DIM + 1, dtype=np.float32), doc_type="c")]
        with pytest.raises(AssertionError):
            idx.add_documents(bad_docs)

    def test_add_raw_vectors(self):
        idx = FaissIndex(dim=DIM, factory_string="Flat")
        vecs = np.random.rand(4, DIM).astype(np.float32)
        idx.add(vecs)
        assert len(idx) == 4

    def test_add_wrong_dim_raises(self):
        idx = FaissIndex(dim=DIM, factory_string="Flat")
        bad = np.zeros((2, DIM + 1), dtype=np.float32)
        with pytest.raises(AssertionError):
            idx.add(bad)

    def test_add_1d_raises(self):
        idx = FaissIndex(dim=DIM, factory_string="Flat")
        with pytest.raises(AssertionError):
            idx.add(np.zeros(DIM, dtype=np.float32))

    # Searching
    def test_search_single_query(self, flat_index):
        query = np.random.rand(DIM).astype(np.float32)
        distances, indices = flat_index.search(query, k=2)
        assert distances.shape == (1, 2)
        assert indices.shape == (1, 2)

    def test_search_batch_queries(self, flat_index):
        queries = np.random.rand(4, DIM).astype(np.float32)
        distances, indices = flat_index.search(queries, k=2)
        assert distances.shape == (4, 2)
        assert indices.shape == (4, 2)

    def test_search_returns_valid_indices(self, flat_index):
        query = np.random.rand(DIM).astype(np.float32)
        _, indices = flat_index.search(query, k=3)
        # All indices should be within range [0, n_docs)
        assert all(0 <= i < 3 for i in indices[0])

    def test_search_k_larger_than_index(self, flat_index):
        query = np.random.rand(DIM).astype(np.float32)
        # flat_index has 3 docs, requesting k=10 should still work
        distances, indices = flat_index.search(query, k=10)
        assert distances.shape[0] == 1

    # Save and load
    def test_save_and_load(self, tmp_path, flat_index):
        path = str(tmp_path / "index.pkl")
        flat_index.save(path)
        loaded = FaissIndex.load(path)
        assert len(loaded) == len(flat_index)
        assert loaded.dim == flat_index.dim
        assert len(loaded.docs) == len(flat_index.docs)

    def test_load_preserves_search_results(self, tmp_path, flat_index):
        path = str(tmp_path / "index.pkl")
        flat_index.save(path)
        loaded = FaissIndex.load(path)
        query = np.random.rand(DIM).astype(np.float32)
        d1, i1 = flat_index.search(query, k=2)
        d2, i2 = loaded.search(query, k=2)
        np.testing.assert_array_equal(i1, i2)

    # __len__ and __repr__
    def test_len(self, flat_index):
        assert len(flat_index) == 3

    def test_repr_contains_info(self, flat_index):
        r = repr(flat_index)
        assert "FaissIndex" in r
        assert "Flat" in r
        assert str(DIM) in r


# ── HNSWIndex ──────────────────────────────────────────────────────────────────

class TestHNSWIndex:
    def test_returns_faiss_index(self):
        idx = HNSWIndex(dim=DIM, M=16)
        assert isinstance(idx, FaissIndex)
        assert idx.dim == DIM

    def test_ef_construction_set(self):
        idx = HNSWIndex(dim=DIM, M=16, ef_construction=100)
        assert idx.index.hnsw.efConstruction == 100

    def test_ef_search_set(self):
        idx = HNSWIndex(dim=DIM, M=16, ef_search=32)
        assert idx.index.hnsw.efSearch == 32


# ── available_index_classes ────────────────────────────────────────────────────

class TestAvailableIndexClasses:
    def test_returns_list(self):
        result = available_index_classes()
        assert isinstance(result, list)

    def test_contains_flat(self):
        result = available_index_classes()
        assert any("Flat" in name for name in result)

    def test_is_sorted(self):
        result = available_index_classes()
        assert result == sorted(result)


# ── chunk_texts ────────────────────────────────────────────────────────────────

class TestChunkTexts:
    def test_basic_chunking(self):
        texts = ["one two three four five"]
        chunks = chunk_texts(texts, chunk_size=3, overlap=0)
        assert chunks == ["one two three", "four five"]

    def test_with_overlap(self):
        texts = ["a b c d e"]
        # chunk_size=3, overlap=1, step=2
        chunks = chunk_texts(texts, chunk_size=3, overlap=1)
        assert chunks[0] == "a b c"
        assert chunks[1] == "c d e"

    def test_empty_text_skipped(self):
        texts = ["", "   ", "hello world"]
        chunks = chunk_texts(texts, chunk_size=5, overlap=0)
        # Empty texts produce no tokens, so skipped
        assert all("hello" in c or "world" in c for c in chunks)

    def test_multiple_texts(self):
        texts = ["a b c", "x y z"]
        chunks = chunk_texts(texts, chunk_size=2, overlap=0)
        assert len(chunks) == 4  # [a b], [c], [x y], [z]

    def test_overlap_equals_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            chunk_texts(["hello world"], chunk_size=3, overlap=3)

    def test_overlap_exceeds_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_texts(["hello world"], chunk_size=3, overlap=5)

    def test_single_token_text(self):
        chunks = chunk_texts(["hello"], chunk_size=3, overlap=0)
        assert chunks == ["hello"]

    def test_exact_chunk_size(self):
        chunks = chunk_texts(["a b c"], chunk_size=3, overlap=0)
        assert chunks == ["a b c"]

    def test_returns_flat_list(self):
        texts = ["a b", "c d"]
        chunks = chunk_texts(texts, chunk_size=1, overlap=0)
        assert all(isinstance(c, str) for c in chunks)


# ── build (high-level) ─────────────────────────────────────────────────────────

class TestBuild:
    def test_build_creates_index_and_saves(self, tmp_path, mock_embedder):
        output = str(tmp_path / "index.pkl")
        texts = ["hello world foo bar baz", "another document here"]
        mock_embedder.encode.side_effect = lambda texts, **kw: np.ones(
            (len(texts), DIM), dtype=np.float32
        )
        mock_embedder.embedding_dim = DIM

        idx = build(
            dataloader=iter(texts),
            embedder=mock_embedder,
            output_path=output,
            factory_string="Flat",
            chunk_size=None,
        )

        assert isinstance(idx, FaissIndex)
        assert len(idx) == len(texts)
        import os
        assert os.path.exists(output)

    def test_build_with_chunking(self, tmp_path, mock_embedder):
        output = str(tmp_path / "index.pkl")
        long_text = " ".join([f"word{i}" for i in range(20)])
        texts = [long_text]

        call_counts = {}

        def _encode(texts_arg, **kw):
            n = len(texts_arg)
            call_counts["n"] = n
            return np.ones((n, DIM), dtype=np.float32)

        mock_embedder.encode.side_effect = _encode

        idx = build(
            dataloader=iter(texts),
            embedder=mock_embedder,
            output_path=output,
            factory_string="Flat",
            chunk_size=5,
            overlap=0,
        )
        # 20 words / 5 = 4 chunks
        assert len(idx) == 4

    def test_build_accepts_string_embedder(self, tmp_path):
        output = str(tmp_path / "index.pkl")
        texts = ["hello world"]
        fake_embedder = MagicMock()
        fake_embedder.encode.return_value = np.ones((1, DIM), dtype=np.float32)

        with patch("de_rag.index.SentenceTransformerEmbedder", return_value=fake_embedder):
            idx = build(
                dataloader=iter(texts),
                embedder="some-model-name",
                output_path=output,
                factory_string="Flat",
                chunk_size=None,
            )
        assert isinstance(idx, FaissIndex)
