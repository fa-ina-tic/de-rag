from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
from de_rag.classes import Document, RetrievalResult
from de_rag.embedders import BaseEmbedder
from de_rag.index import FaissIndex
from de_rag.logger import get_logger

logger = get_logger(__name__)


class BaseRetriever(ABC):
    """Base retriever backed by an FaissIndex.

    The index can be supplied at construction time or overridden per-call.
    Document management (add_documents) is handled entirely by FaissIndex.
    """

    def __init__(self, embedder: BaseEmbedder, index: Optional[FaissIndex] = None, ):
        self.index = index
        self.embedder = embedder

    def _resolve(self, index: Optional[FaissIndex]):
        idx = index or self.index
        if not self.index:
            self.index = index
        if idx is None:
            raise ValueError("No Index provided — pass one to __init__ or retrieve().")

    @abstractmethod
    def _preprocess(self, query: str|List[str], *args) -> np.ndarray:
        """Apply any query-side transformation before searching."""

    def retrieve(
        self,
        query: str|List[str],
        top_k: int = 5,
        source_label: str = "",
        index: Optional[FaissIndex] = None,
        **kwargs,
    ) -> List[List[RetrievalResult]]:
        """Search for one or more queries.

        Parameters
        ----------
        query : np.ndarray
            Shape (dim,) for a single query or (N, dim) for a batch.
        index : FaissIndex, optional
            Overrides the instance-level index for this call.

        Returns
        -------
        List[List[RetrievalResult]] — one inner list per query.
        """
        self._resolve(index)
        assert self.index is not None
        query_embeds = self._preprocess(query)
        logger.debug("Searching index with top_k=%d, retriever=%s", top_k, source_label or type(self).__name__)
        distances, raw_indices = self.index.search(query_embeds, k=top_k)
        docs = self.index.docs
        return [
            [
                RetrievalResult(
                    doc=docs[i],
                    score=float(d),
                    source=source_label,
                )
                for d, i in zip(dists, idxs)
                if i != -1
            ]
            for dists, idxs in zip(distances, raw_indices)
        ]

    def batch_query(
        self, queries: str|List[str], **kwargs
    ) -> List[List[RetrievalResult]]:
        """Alias for retrieve."""
        return self.retrieve(queries, **kwargs)


class CosineRetriever(BaseRetriever):
    """Retriever with no query-side transformation."""

    def _preprocess(self, query, *args) -> np.ndarray:
        return self.embedder.encode(query, show_progress=False, normalize=True)


class NonNegativeRetriever(BaseRetriever):
    """Retriever that zeros out negative query dimensions before searching."""

    def _preprocess(self, query, *args) -> np.ndarray:
        embedded_query: np.ndarray = self.embedder.encode(query, show_progress=False, normalize=True)
        embedded_query[embedded_query < 0] = 0
        return embedded_query

class NERRetriever(BaseRetriever):

    def __init__(self,
                embedder: BaseEmbedder,
                index: Optional[FaissIndex] = None,
                ner_model_name_or_path: str = "urchade/gliner_multi-v2.1"):
        super().__init__(embedder, index)
        ## Check dependencies and arguments
        try:
            from gliner import GLiNER
        except ImportError:
            raise AssertionError(
                "Required library 'gliner' is not installed\n",
                "Run 'pip install gliner' to use NERRetriever.\n"
            )
        
        self.ner_model = GLiNER.from_pretrained(ner_model_name_or_path)
        self.embedder = embedder

    def _mask_query(self, query: str, labels: List[str] = ["subject", "object"]) -> List[str]:
        entities = self.ner_model.predict_entities(query, labels)
        logger.debug("NER predict_entities found %d entities for query: '%s'", len(entities), query)
        queries: List[str] = []
        for entity in entities:
            logger.info("NER entity detected: '%s'", entity['text'])
            logger.debug("NER masking: '%s' -> '%s'", query, query.replace(entity['text'], ' '))
            queries.append(query.replace(entity["text"], " "))
        if not queries:
            logger.debug("No entities detected — using original query")
            queries = [query]
        return queries

    def _preprocess(self, query, *args) -> np.ndarray:
        queries = self._mask_query(query)
        embeddings = [
            self.embedder.encode(q, show_progress=False, normalize=True)
            for q in queries
        ]
        return np.stack(embeddings)