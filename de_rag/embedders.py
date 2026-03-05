"""embedders.py – unified embedder interface for de-rag.

Provides a common ``BaseEmbedder`` ABC so that retrieval and indexing code
is decoupled from the underlying embedding backend (local model vs. API).

Supported backends
------------------
- SentenceTransformerEmbedder  – wraps ``sentence_transformers.SentenceTransformer``
- CohereEmbedder               – wraps the Cohere Embed API (requires ``cohere``)
"""

from __future__ import annotations

import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

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
        requests_per_minute: int = 40,
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
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._last_request_time: float = 0.0
        logger.info("Initialized CohereEmbedder with model '%s' (%.1f req/min limit)", model, requests_per_minute)

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
            # Rate-limit: enforce minimum interval between requests
            elapsed = time.monotonic() - self._last_request_time
            wait = self._min_interval - elapsed
            if wait > 0:
                logger.debug("Rate limiting: sleeping %.2fs before Cohere request", wait)
                time.sleep(wait)

            # Retry with exponential backoff on 429
            for attempt in range(5):
                try:
                    self._last_request_time = time.monotonic()
                    response = self._client.embed(
                        texts=batch,  # ty:ignore[invalid-argument-type]
                        model=self._model,
                        input_type=effective_input_type,
                    )
                    break
                except Exception as exc:
                    if "429" in str(type(exc)) or "TooManyRequests" in type(exc).__name__:
                        backoff = 2 ** attempt * 5
                        logger.warning("Cohere 429 on attempt %d; retrying in %ds", attempt + 1, backoff)
                        time.sleep(backoff)
                    else:
                        raise
            else:
                raise RuntimeError("Cohere embed failed after 5 retries due to rate limiting")

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


# ── Corpus embedding helpers ───────────────────────────────────────────────────

def embed_corpus(
    corpus: Dict,
    embedder: BaseEmbedder,
    batch_size: int = 64,
) -> List:
    """Encode all corpus documents and wrap them in Document objects."""
    from de_rag.classes import Document

    doc_ids = list(corpus.keys())
    texts = [
        (corpus[d]["title"] + " " + corpus[d]["text"]).strip()
        for d in doc_ids
    ]

    logger.info("Encoding %d corpus documents ...", len(texts))
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress=True,
        normalize=True,
    )

    return [
        Document(id=doc_ids[i], text=texts[i], embedding=embeddings[i], doc_type="chunk")
        for i in range(len(doc_ids))
    ]


def load_or_embed_corpus(
    corpus: Dict,
    embedder: BaseEmbedder,
    cache_path: Optional[Path],
    batch_size: int = 64,
) -> List:
    """Return embedded corpus Documents, loading from cache if available.

    If cache_path is provided and the file exists, documents are loaded from
    it and embedding is skipped. Otherwise documents are embedded and, if
    cache_path is provided, saved for future runs.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading embeddings from '%s' ...", cache_path)
        with open(cache_path, "rb") as f:
            docs = pickle.load(f)
        logger.info("Loaded %d documents from cache.", len(docs))
        return docs

    docs = embed_corpus(corpus, embedder, batch_size)

    if cache_path is not None:
        logger.info("Saving embeddings to '%s' ...", cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(docs, f)
        logger.info("Saved %d documents to cache.", len(docs))

    return docs
