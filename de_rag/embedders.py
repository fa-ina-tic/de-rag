"""embedders.py – unified embedder interface for de-rag.

Provides a common ``BaseEmbedder`` ABC so that retrieval and indexing code
is decoupled from the underlying embedding backend (local model vs. API).

Supported backends
------------------
- SentenceTransformerEmbedder  – wraps ``sentence_transformers.SentenceTransformer``
- CohereEmbedder               – wraps the Cohere Embed API (requires ``cohere``)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from de_rag.logger import get_logger

logger = get_logger(__name__)

Texts = Union[str, List[str]]


class BaseEmbedder(ABC):
    """Abstract base class for all embedding backends.

    Subclasses must implement :meth:`encode`.
    """

    @abstractmethod
    def encode(
        self,
        texts: Texts,
        *,
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed one or more texts.

        Parameters
        ----------
        texts:
            A single string or a list of strings to embed.
        batch_size:
            Number of texts to process per batch (ignored by API backends).
        show_progress:
            Display a progress bar while encoding (local models only).
        normalize:
            L2-normalise the output vectors.

        Returns
        -------
        np.ndarray
            Shape ``(dim,)`` for a single string, ``(N, dim)`` for a list.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output vectors."""


# ── SentenceTransformer backend ───────────────────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder backed by a local ``SentenceTransformer`` model.

    Parameters
    ----------
    model_name_or_path:
        Any model identifier accepted by ``SentenceTransformer()``.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).  ``None`` lets
        sentence-transformers choose automatically.
    """

    def __init__(self, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name_or_path, device=device)
        logger.info("Loaded SentenceTransformer model '%s'", model_name_or_path)

    def encode(
        self,
        texts: Texts,
        *,
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

    @property
    def embedding_dim(self) -> int | None:
        return self._model.get_sentence_embedding_dimension()


# ── Cohere backend ────────────────────────────────────────────────────────────

class CohereEmbedder(BaseEmbedder):
    """Embedder backed by the Cohere Embed API.

    Parameters
    ----------
    api_key:
        Cohere API key.  If ``None``, the ``COHERE_API_KEY`` environment
        variable is used automatically by the ``cohere`` client.
    model:
        Cohere embed model name (e.g. ``"embed-english-v3.0"``).
    input_type:
        ``"search_document"`` when indexing corpus texts;
        ``"search_query"`` when embedding queries.
        The default ``"search_document"`` works for both when fine-grained
        control is not needed, but callers can pass ``input_type`` per call
        via keyword arguments (forwarded to :meth:`encode`).

    Notes
    -----
    Requires ``pip install cohere``.
    Cohere v3 models return float32 embeddings; earlier models return lists
    that are converted here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
    ) -> None:
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Package 'cohere' is not installed. "
                "Run 'pip install cohere' to use CohereEmbedder."
            )

        self._client = cohere.Client(api_key) if api_key else cohere.Client()
        self._model = model
        self._default_input_type = input_type
        self._dim: int | None = None
        logger.info("Initialized CohereEmbedder with model '%s'", model)

    def encode(
        self,
        texts: Texts,
        *,
        batch_size: int = 96,
        show_progress: bool = False,
        normalize: bool = True,
        input_type: str | None = None,
    ) -> np.ndarray:
        """Embed texts via the Cohere API.

        Parameters
        ----------
        input_type:
            Overrides the instance-level default for this call.
            Use ``"search_query"`` when encoding queries at retrieval time.
        """
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        effective_input_type = input_type or self._default_input_type

        # Cohere recommends ≤96 texts per request
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            if show_progress:
                logger.info(
                    "Cohere embed batch %d-%d / %d",
                    start + 1,
                    start + len(batch),
                    len(text_list),
                )
            response = self._client.embed(
                texts=batch,  # ty:ignore[invalid-argument-type]
                model=self._model,
                input_type=effective_input_type,
            )
            batch_arr = np.array(response.embeddings, dtype=np.float32)
            all_embeddings.append(batch_arr)

        embeddings = np.concatenate(all_embeddings, axis=0)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        if self._dim is None:
            self._dim = embeddings.shape[1]

        return embeddings[0] if single else embeddings

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            raise RuntimeError(
                "Embedding dimension is unknown until the first call to encode()."
            )
        return self._dim
