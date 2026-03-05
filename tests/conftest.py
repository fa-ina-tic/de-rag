"""Shared fixtures for de_rag unit tests."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from de_rag.classes import Document, RetrievalResult
from de_rag.index import FaissIndex

DIM = 8  # Small dimension for fast tests


@pytest.fixture
def dim():
    return DIM


@pytest.fixture
def mock_embedder():
    """A mock BaseEmbedder that returns deterministic random vectors."""
    rng = np.random.default_rng(42)
    embedder = MagicMock()
    embedder.embedding_dim = DIM

    def _encode(texts, **kwargs):
        if isinstance(texts, str):
            return rng.random(DIM).astype(np.float32)
        return rng.random((len(texts), DIM)).astype(np.float32)

    embedder.encode.side_effect = _encode
    return embedder


@pytest.fixture
def sample_docs():
    """Three Document objects with small random embeddings."""
    rng = np.random.default_rng(0)
    vecs = rng.random((3, DIM)).astype(np.float32)
    return [
        Document(id=f"doc{i}", text=f"sample text {i}", embedding=vecs[i], doc_type="chunk")
        for i in range(3)
    ]


@pytest.fixture
def flat_index(sample_docs):
    """A FaissIndex (Flat/exact search) pre-loaded with sample_docs."""
    idx = FaissIndex(dim=DIM, factory_string="Flat", metric="l2")
    idx.add_documents(sample_docs)
    return idx


@pytest.fixture
def retrieval_results(sample_docs):
    """Three RetrievalResult objects wrapping sample_docs."""
    return [
        RetrievalResult(doc=doc, score=1.0 - i * 0.1, source="test")
        for i, doc in enumerate(sample_docs)
    ]
