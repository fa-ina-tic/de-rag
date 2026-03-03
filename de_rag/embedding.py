import pickle
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from de_rag.classes import Document

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_corpus(
    corpus: Dict,
    embedder: SentenceTransformer,
    batch_size: int = 64,
) -> List[Document]:
    """Encode all corpus documents and wrap them in Document objects."""
    doc_ids = list(corpus.keys())
    texts   = [
        (corpus[d]["title"] + " " + corpus[d]["text"]).strip()
        for d in doc_ids
    ]

    print(f"[Embed] Encoding {len(texts):,} corpus documents ...")
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return [
        Document(id=doc_ids[i], text=texts[i], embedding=embeddings[i], doc_type="chunk")
        for i in range(len(doc_ids))
    ]


def load_or_embed_corpus(
    corpus: Dict,
    embedder: SentenceTransformer,
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
        print(f"[Cache] Loading embeddings from '{cache_path}' ...")
        with open(cache_path, "rb") as f:
            docs: List[Document] = pickle.load(f)
        print(f"[Cache] Loaded {len(docs):,} documents.")
        return docs

    docs = embed_corpus(corpus, embedder, batch_size)

    if cache_path is not None:
        print(f"[Cache] Saving embeddings to '{cache_path}' ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(docs, f)
        print(f"[Cache] Saved {len(docs):,} documents.")

    return docs
