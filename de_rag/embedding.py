import pickle
from pathlib import Path
from typing import Dict, List, Optional

from de_rag.classes import Document
from de_rag.embedders import BaseEmbedder
from de_rag.logger import get_logger

logger = get_logger(__name__)

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_corpus(
    corpus: Dict,
    embedder: BaseEmbedder,
    batch_size: int = 64,
) -> List[Document]:
    """Encode all corpus documents and wrap them in Document objects."""
    doc_ids = list(corpus.keys())
    texts   = [
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
) -> List[Document]:
    """
    Return embedded corpus Documents, loading from cache if available.

    If cache_path is provided and the file exists, documents are loaded from
    it and embedding is skipped. Otherwise documents are embedded and, if
    cache_path is provided, saved for future runs.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading embeddings from '%s' ...", cache_path)
        with open(cache_path, "rb") as f:
            docs: List[Document] = pickle.load(f)
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
