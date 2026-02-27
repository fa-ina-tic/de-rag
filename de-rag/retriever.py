# ---------------------------------------------------------------------------
# Retrieval engine (cosine similarity over pre-embedded index)
# ---------------------------------------------------------------------------

import numpy as np
from typing import List, Optional, Dict
from decomposer import HadamardQueryDecomposer
from classes import Document, RetrievalResult

class FrequencyAwareRetriever:
    """
    Plug-in RAG retriever that uses Hadamard frequency decomposition.

    Two separate document indices are maintained:
        - global_index: document summaries / GraphRAG community reports
        - local_index:  fine-grained text chunks

    At query time:
        - q_global → retrieves from global_index
        - q_local  → retrieves from local_index
        - Results are merged and deduplicated
    """

    def __init__(
        self,
        dim: int,
        low_ratio: float = 0.5,
        normalize_output: bool = True,
    ):
        self.decomposer = HadamardQueryDecomposer(dim, low_ratio, normalize_output)
        self.global_docs: List[Document] = []
        self.local_docs:  List[Document] = []

        # Pre-built embedding matrices for fast batch cosine similarity
        self._global_matrix: Optional[np.ndarray] = None
        self._local_matrix:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def add_documents(self, docs: List[Document]) -> None:
        """Add pre-embedded documents to the appropriate index."""
        for doc in docs:
            if doc.doc_type == "summary":
                self.global_docs.append(doc)
            elif doc.doc_type == "chunk":
                self.local_docs.append(doc)
            else:
                raise ValueError(f"doc_type must be 'summary' or 'chunk', got '{doc.doc_type}'")
        self._rebuild_matrices()

    def _rebuild_matrices(self) -> None:
        """Stack embeddings into matrices for efficient batch similarity."""
        if self.global_docs:
            self._global_matrix = np.vstack([d.embedding for d in self.global_docs])
            self._global_matrix = self._row_normalize(self._global_matrix)
        if self.local_docs:
            self._local_matrix = np.vstack([d.embedding for d in self.local_docs])
            self._local_matrix = self._row_normalize(self._local_matrix)

    @staticmethod
    def _row_normalize(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return M / (norms + eps)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: np.ndarray,
        top_k_global: int = 3,
        top_k_local:  int = 5,
        deduplicate: bool = True,
    ) -> List[RetrievalResult]:
        """
        Main retrieval method.

        Args:
            query: raw query embedding, shape (dim,)
            top_k_global: number of summary docs to retrieve
            top_k_local:  number of chunk docs to retrieve
            deduplicate:  remove duplicates by doc id

        Returns:
            Merged list of RetrievalResult, global results first
        """
        q_global, q_local = self.decomposer.decompose(query)

        global_results = self._cosine_search(
            q_global, self._global_matrix, self.global_docs, top_k_global, "global"
        )
        local_results = self._cosine_search(
            q_local, self._local_matrix, self.local_docs, top_k_local, "local"
        )

        merged = global_results + local_results

        if deduplicate:
            seen = set()
            deduped = []
            for r in merged:
                if r.doc.id not in seen:
                    seen.add(r.doc.id)
                    deduped.append(r)
            merged = deduped

        return merged

    def retrieve_multi_band(
        self,
        query: np.ndarray,
        n_bands: int = 4,
        top_k_per_band: int = 3,
    ) -> List[RetrievalResult]:
        """
        Generalized retrieval using n_bands frequency decomposition.
        All bands retrieve from local_index; useful when you only have one index
        and want diverse sub-query coverage.
        """
        sub_vectors = self.decomposer.decompose_multi_band(query, n_bands)
        all_results = []
        seen = set()

        band_labels = [f"band_{i}" for i in range(n_bands)]
        for label, q_sub in zip(band_labels, sub_vectors):
            results = self._cosine_search(
                q_sub, self._local_matrix, self.local_docs, top_k_per_band, label
            )
            for r in results:
                if r.doc.id not in seen:
                    seen.add(r.doc.id)
                    all_results.append(r)

        return all_results

    @staticmethod
    def _cosine_search(
        query_vec: np.ndarray,
        doc_matrix: Optional[np.ndarray],
        docs: List[Document],
        top_k: int,
        source_label: str,
    ) -> List[RetrievalResult]:
        """Batch cosine similarity search against a pre-normalized doc matrix."""
        if doc_matrix is None or len(docs) == 0:
            return []

        # query_vec already normalized by decomposer
        scores = doc_matrix @ query_vec          # shape: (num_docs,)
        top_k  = min(top_k, len(docs))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            RetrievalResult(doc=docs[i], score=float(scores[i]), source=source_label)
            for i in top_indices
        ]

    def diagnostics(self, query: np.ndarray) -> Dict:
        """Return energy split and sub-vector stats for a query."""
        energy = self.decomposer.energy_split(query)
        q_global, q_local = self.decomposer.decompose(query)
        return {
            "energy_split": energy,
            "q_global_norm": float(np.linalg.norm(q_global)),
            "q_local_norm":  float(np.linalg.norm(q_local)),
            "cosine_between_subvectors": float(q_global @ q_local),
        }
