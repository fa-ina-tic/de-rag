"""Tests for de_rag.classes (Document, RetrievalResult)."""
import dataclasses

import numpy as np
import pytest
import torch

from de_rag.classes import Document, RetrievalResult


def _make_doc(id="d1", text="hello", doc_type="chunk"):
    embedding = torch.tensor([0.1, 0.2, 0.3])
    return Document(id=id, text=text, embedding=embedding, doc_type=doc_type)


class TestDocument:
    def test_creation(self):
        emb = torch.zeros(4)
        doc = Document(id="1", text="foo", embedding=emb, doc_type="chunk")
        assert doc.id == "1"
        assert doc.text == "foo"
        assert doc.doc_type == "chunk"
        assert torch.equal(doc.embedding, emb)

    def test_accepts_numpy_embedding(self):
        emb = np.array([1.0, 2.0], dtype=np.float32)
        doc = Document(id="x", text="bar", embedding=emb, doc_type="text")
        np.testing.assert_array_equal(doc.embedding, emb)

    def test_equality(self):
        emb = torch.tensor([1.0])
        doc1 = Document(id="a", text="t", embedding=emb, doc_type="c")
        doc2 = Document(id="a", text="t", embedding=emb, doc_type="c")
        assert doc1 == doc2

    def test_inequality_different_id(self):
        emb = torch.tensor([1.0])
        doc1 = Document(id="a", text="t", embedding=emb, doc_type="c")
        doc2 = Document(id="b", text="t", embedding=emb, doc_type="c")
        assert doc1 != doc2

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(Document)

    def test_fields(self):
        names = {f.name for f in dataclasses.fields(Document)}
        assert names == {"id", "text", "embedding", "doc_type"}


class TestRetrievalResult:
    def test_creation(self):
        doc = _make_doc()
        rr = RetrievalResult(doc=doc, score=0.95, source="simple")
        assert rr.doc is doc
        assert rr.score == 0.95
        assert rr.source == "simple"

    def test_equality(self):
        doc = _make_doc()
        rr1 = RetrievalResult(doc=doc, score=0.5, source="s")
        rr2 = RetrievalResult(doc=doc, score=0.5, source="s")
        assert rr1 == rr2

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(RetrievalResult)

    def test_fields(self):
        names = {f.name for f in dataclasses.fields(RetrievalResult)}
        assert names == {"doc", "score", "source"}

    def test_score_accepts_float(self):
        doc = _make_doc()
        rr = RetrievalResult(doc=doc, score=1 / 3, source="x")
        assert isinstance(rr.score, float)
