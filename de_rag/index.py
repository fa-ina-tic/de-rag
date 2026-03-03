import pickle

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from de_rag.classes import Document

class HNSWIndex:
    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200, ef_search: int = 50):
        self.dim = dim
        self.docs: List["Document"] = []
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def add_documents(self, docs: List["Document"]) -> None:
        self.docs.extend(docs)
        vectors = np.vstack([d.embedding for d in docs]).astype(np.float32)
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)  # type: ignore[arg-type]

    def add(self, vectors: np.ndarray) -> None:
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        self.index.add(vectors.astype(np.float32))  # type: ignore[arg-type]

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for a single query vector (1-D) or a batch (2-D).
        Returns (distances, indices) with shape (n_queries, k).
        """
        query = np.atleast_2d(query).astype(np.float32)
        assert query.shape[1] == self.dim
        return self.index.search(query, k)  # type: ignore[arg-type]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return self.index.ntotal

def chunk_texts(texts: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Split each text into overlapping token-based chunks.

    Parameters
    ----------
    texts : list of str
        Raw texts to chunk.
    chunk_size : int
        Maximum number of whitespace-separated tokens per chunk.
    overlap : int
        Number of tokens shared between consecutive chunks.

    Returns
    -------
    Flat list of chunk strings.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    chunks: List[str] = []
    step = chunk_size - overlap
    for text in texts:
        tokens = text.split()
        if not tokens:
            continue
        for start in range(0, len(tokens), step):
            chunk = " ".join(tokens[start : start + chunk_size])
            chunks.append(chunk)
    return chunks


def build(
    dataloader,
    model_name_or_path: str,
    output_path: str,
    chunk_size: int | None = 256,
    overlap: int = 0,
) -> HNSWIndex:
    """Embed all texts from dataloader, build an HNSWIndex, and save it.

    Parameters
    ----------
    dataloader : BaseDataLoader
        Any iterable that yields raw text strings.
    model_name_or_path : str
        SentenceTransformer model name or local path.
    output_path : str
        File path where the FAISS index will be written (e.g. "index.faiss").
    chunk_size : int, optional
        If provided, each document is split into chunks of this many
        whitespace-separated tokens before embedding.
    overlap : int
        Number of tokens shared between consecutive chunks (default 0).
        Only used when chunk_size is set.

    Returns
    -------
    HNSWIndex with all documents added.
    """
    from de_rag.classes import Document

    embedder = SentenceTransformer(model_name_or_path)
    dim = embedder.get_sentence_embedding_dimension()
    if not dim:
        raise ValueError("Could not determine embedding dimension from model.")

    texts = list(dataloader)
    if chunk_size is not None:
        texts = chunk_texts(texts, chunk_size=chunk_size, overlap=overlap)

    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    docs = [
        Document(id=str(i), text=text, embedding=emb, doc_type="text")
        for i, (text, emb) in enumerate(zip(texts, embeddings))
    ]

    index = HNSWIndex(dim=dim)
    index.add_documents(docs)
    index.save(output_path)
    return index


if __name__ == "__main__":
    import argparse
    from de_rag.dataloader import WikitextDataLoader

    parser = argparse.ArgumentParser(
        description="Build a FAISS HNSW index from Wikitext.",
    )
    parser.add_argument("--model", required=True, help="SentenceTransformer model name or path.")
    parser.add_argument("--output", required=True, metavar="PATH", help="Output .faiss file path.")
    parser.add_argument("--dataset", default="wikitext-2-raw-v1", metavar="NAME", help="Wikitext config name. (default: wikitext-2-raw-v1)")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="Dataset split. (default: train)")
    parser.add_argument("--chunk-size", type=int, default=256, metavar="N", help="Split documents into chunks of N tokens before embedding.")
    parser.add_argument("--overlap", type=int, default=0, metavar="N", help="Token overlap between consecutive chunks. (default: 0)")
    args = parser.parse_args()

    loader = WikitextDataLoader(name=args.dataset, split=args.split)
    idx = build(loader, model_name_or_path=args.model, output_path=args.output, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Done — {len(idx):,} vectors saved to '{args.output}'")