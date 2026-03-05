import pickle
from typing import List, TYPE_CHECKING

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from de_rag.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from de_rag.classes import Document

# Resolve FAISS metric constants via getattr — faiss ships no type stubs so
# direct attribute access produces spurious Pylance warnings.
METRIC_L2: int = getattr(faiss, "METRIC_L2")
METRIC_INNER_PRODUCT: int = getattr(faiss, "METRIC_INNER_PRODUCT")
_METRIC_L1: int = getattr(faiss, "METRIC_L1")
_METRIC_Linf: int = getattr(faiss, "METRIC_Linf")
_IndexBase: type = getattr(faiss, "Index")
_index_factory = getattr(faiss, "index_factory")

_METRIC_ALIASES: dict[str, int] = {
    "l2": METRIC_L2,
    "ip": METRIC_INNER_PRODUCT,
    "inner_product": METRIC_INNER_PRODUCT,
    "cosine": METRIC_INNER_PRODUCT,  # cosine == IP on normalised vectors
    "l1": _METRIC_L1,
    "linf": _METRIC_Linf,
}


def available_index_classes() -> list[str]:
    """Return the names of every FAISS Index subclass in the current installation."""
    return sorted(
        name for name, obj in vars(faiss).items()
        if isinstance(obj, type) and issubclass(obj, _IndexBase) and name != "Index"
    )


class FaissIndex:
    """Generic FAISS index built from any FAISS factory string.

    Uses ``faiss.index_factory`` under the hood, so every index that FAISS
    supports is available — no hardcoded list required.

    Parameters
    ----------
    dim : int
        Vector dimension.
    factory_string : str
        FAISS index-factory descriptor.  Examples::

            "Flat"            # IndexFlatL2 (exact)
            "HNSW32"          # IndexHNSWFlat, M=32
            "IVF100,Flat"     # IndexIVFFlat,  nlist=100  (needs training)
            "IVF100,PQ8"      # IndexIVFPQ,    nlist=100  (needs training)
            "IVF100,SQ8"      # IndexIVFSQ,    nlist=100  (needs training)
            "PQ8"             # IndexPQ                   (needs training)
            "LSH"             # IndexLSH

        See ``faiss.index_factory`` docs or ``available_index_classes()`` for
        the full list supported by your installation.
    metric : int or str
        Distance metric.  Pass a ``faiss.METRIC_*`` constant or one of the
        string aliases ``"l2"``, ``"ip"`` / ``"cosine"``, ``"l1"``, ``"linf"``.
        Default: ``faiss.METRIC_L2``.
    """

    def __init__(
        self,
        dim: int,
        factory_string: str = "HNSW32",
        metric: int | str = METRIC_L2,
    ):
        self.dim = dim
        self.factory_string = factory_string
        if isinstance(metric, str):
            metric = _METRIC_ALIASES.get(metric.lower(), METRIC_L2)
        self.metric: int = metric
        self.docs: List["Document"] = []
        self.index = _index_factory(dim, factory_string, metric)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_if_needed(self, vectors: np.ndarray) -> None:
        if not self.index.is_trained:
            logger.info(
                "Training index %r on %d vectors...", self.factory_string, len(vectors)
            )
            self.index.train(vectors)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_documents(self, docs: List["Document"]) -> None:
        self.docs.extend(docs)
        vectors = np.vstack([d.embedding for d in docs]).astype(np.float32)
        assert vectors.shape[1] == self.dim
        self._train_if_needed(vectors)
        self.index.add(vectors)

    def add(self, vectors: np.ndarray) -> None:
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        vectors = vectors.astype(np.float32)
        self._train_if_needed(vectors)
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for a single query vector (1-D) or a batch (2-D).
        Returns (distances, indices) with shape (n_queries, k).
        """
        query = np.atleast_2d(query).astype(np.float32)
        assert query.shape[1] == self.dim
        return self.index.search(query, k)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"FaissIndex(factory={self.factory_string!r}, "
            f"dim={self.dim}, ntotal={self.index.ntotal})"
        )


# Backward-compatible factory so existing code using HNSWIndex(...) still works.
def HNSWIndex(
    dim: int, M: int = 32, ef_construction: int = 200, ef_search: int = 50
) -> FaissIndex:
    """Return an HNSW-backed FaissIndex (backward-compatible helper)."""
    fi = FaissIndex(dim, f"HNSW{M}")
    fi.index.hnsw.efConstruction = ef_construction
    fi.index.hnsw.efSearch = ef_search
    return fi


# ------------------------------------------------------------------
# Text chunking
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# High-level build helper
# ------------------------------------------------------------------

def build(
    dataloader,
    model_name_or_path: str,
    output_path: str,
    factory_string: str = "HNSW32",
    metric: int | str = faiss.METRIC_L2,
    chunk_size: int | None = 256,
    overlap: int = 0,
) -> FaissIndex:
    """Embed all texts from *dataloader*, build a :class:`FaissIndex`, and save it.

    Parameters
    ----------
    dataloader : BaseDataLoader
        Any iterable that yields raw text strings.
    model_name_or_path : str
        SentenceTransformer model name or local path.
    output_path : str
        File path where the index will be saved.
    factory_string : str
        FAISS factory descriptor (default ``"HNSW32"``).
        Passed verbatim to ``faiss.index_factory``.
    metric : int or str
        Distance metric (default ``faiss.METRIC_L2``).
    chunk_size : int, optional
        If provided, each document is split into chunks of this many
        whitespace-separated tokens before embedding.
    overlap : int
        Number of tokens shared between consecutive chunks (default 0).
        Only used when *chunk_size* is set.

    Returns
    -------
    FaissIndex with all documents added.
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

    index = FaissIndex(dim=dim, factory_string=factory_string, metric=metric)
    index.add_documents(docs)
    index.save(output_path)
    return index


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from de_rag.dataloader import WikitextDataLoader
    from de_rag.logger import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Build a FAISS index from Wikitext.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Factory string examples:\n"
            "  Flat            exact L2 (or IP with --metric ip)\n"
            "  HNSW32          HNSW graph, M=32\n"
            "  IVF100,Flat     IVF with 100 clusters (auto-trained)\n"
            "  IVF100,PQ8      IVF + product quantisation\n"
            "  IVF100,SQ8      IVF + scalar quantisation\n"
            "  PQ8             pure product quantisation\n"
            "  LSH             locality-sensitive hashing\n"
            "\nRun 'python -c \"from de_rag.index import available_index_classes; "
            "print(*available_index_classes(), sep=chr(10))\"' for the full list."
        ),
    )
    parser.add_argument("--model", required=True, help="SentenceTransformer model name or path.")
    parser.add_argument("--output", required=True, metavar="PATH", help="Output file path.")
    parser.add_argument(
        "--factory",
        default="HNSW32",
        metavar="DESC",
        help="FAISS index-factory string. (default: HNSW32)",
    )
    parser.add_argument(
        "--metric",
        default="l2",
        choices=list(_METRIC_ALIASES),
        help="Distance metric. (default: l2)",
    )
    parser.add_argument("--dataset", default="wikitext-2-raw-v1", metavar="NAME", help="Wikitext config name. (default: wikitext-2-raw-v1)")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="Dataset split. (default: train)")
    parser.add_argument("--chunk-size", type=int, default=256, metavar="N", help="Split documents into chunks of N tokens before embedding.")
    parser.add_argument("--overlap", type=int, default=0, metavar="N", help="Token overlap between consecutive chunks. (default: 0)")
    args = parser.parse_args()

    loader = WikitextDataLoader(name=args.dataset, split=args.split)
    idx = build(
        loader,
        model_name_or_path=args.model,
        output_path=args.output,
        factory_string=args.factory,
        metric=args.metric,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    logger.info(
        "Done — %d vectors saved to '%s' (factory=%s)", len(idx), args.output, args.factory
    )
