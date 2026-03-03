import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
from de_rag.classes import Document, RetrievalResult
from de_rag.index import HNSWIndex


class BaseRetriever(ABC):
    """Base retriever backed by an HNSWIndex.

    The index can be supplied at construction time or overridden per-call.
    Document management (add_documents) is handled entirely by HNSWIndex.
    """

    def __init__(self, index: Optional[HNSWIndex] = None):
        self.index = index

    def _resolve(self, index: Optional[HNSWIndex]):
        idx = index or self.index
        if not self.index:
            self.index = index
        if idx is None:
            raise ValueError("No Index provided — pass one to __init__ or retrieve().")

    @abstractmethod
    def _preprocess(self, query: np.ndarray) -> np.ndarray:
        """Apply any query-side transformation before searching."""

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
        source_label: str = "",
        index: Optional[HNSWIndex] = None,
        **kwargs,
    ) -> List[List[RetrievalResult]]:
        """Search for one or more queries.

        Parameters
        ----------
        query : np.ndarray
            Shape (dim,) for a single query or (N, dim) for a batch.
        index : HNSWIndex, optional
            Overrides the instance-level index for this call.

        Returns
        -------
        List[List[RetrievalResult]] — one inner list per query.
        """
        self._resolve(index)
        assert self.index is not None
        query = self._preprocess(np.atleast_2d(query).astype(np.float32))
        distances, raw_indices = self.index.search(query, k=top_k)
        docs = self.index.docs
        return [
            [
                RetrievalResult(
                    doc=docs[i] if i < len(docs)
                    else Document(id=str(i), text="", embedding=torch.empty(0), doc_type=""),
                    score=float(-d),
                    source=source_label,
                )
                for d, i in zip(dists, idxs)
                if i != -1
            ]
            for dists, idxs in zip(distances, raw_indices)
        ]

    def batch_query(
        self, queries: np.ndarray, **kwargs
    ) -> List[List[RetrievalResult]]:
        """Alias for retrieve."""
        return self.retrieve(queries, **kwargs)


class CosineRetriever(BaseRetriever):
    """Retriever with no query-side transformation."""

    def _preprocess(self, query: np.ndarray) -> np.ndarray:
        return query


class NonNegativeRetriever(BaseRetriever):
    """Retriever that zeros out negative query dimensions before searching."""

    def _preprocess(self, query: np.ndarray) -> np.ndarray:
        q = query.copy()
        q[q < 0] = 0
        return q
