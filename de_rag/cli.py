"""cli.py – interactive retrieval tool for de-rag.

Loads a pre-built HNSW index (produced by de_rag.index), initialises an
embedding model, and lets you search with one or more retrievers
either in a single-shot or interactive REPL mode.

Index files are produced by running:
    python -m de_rag.index --model <model> --output <name>.faiss ...

The index file bundles both the FAISS vectors and document metadata.

Usage examples
--------------
# one-shot query with local SentenceTransformer (default)
python -m de_rag.cli "CRISPR mechanism" \\
    --index wikitext.faiss \\
    --retriever cosine --retriever nonneg

# use Cohere embeddings
python -m de_rag.cli "CRISPR mechanism" \\
    --index wikitext.faiss \\
    --embedder cohere --cohere-model embed-english-v3.0

# interactive REPL (omit the query argument)
python -m de_rag.cli \\
    --index wikitext.faiss \\
    --retriever cosine --retriever nonneg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

from de_rag.classes import RetrievalResult
from de_rag.embedders import BaseEmbedder, CohereEmbedder, SentenceTransformerEmbedder
from de_rag.index import HNSWIndex
from de_rag.logger import get_logger, setup_logging
from de_rag.retriever import BaseRetriever, CosineRetriever, NonNegativeRetriever, NERRetriever

logger = get_logger("de_rag.cli")

# ── Retriever registry ────────────────────────────────────────────────────────

RETRIEVERS: Dict[str, type[BaseRetriever]] = {
    "cosine": CosineRetriever,
    "nonneg": NonNegativeRetriever,
    "ner": NERRetriever
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_index(index_path: Path) -> HNSWIndex:
    """Load a pre-built HNSWIndex (vectors + docs bundled in one file)."""
    if not index_path.exists():
        logger.error("Index file '%s' not found.", index_path)
        sys.exit(1)

    logger.info("Loading index '%s' ...", index_path)
    index = HNSWIndex.load(str(index_path))
    logger.info("Index ready — %d vectors, %d documents.", len(index), len(index.docs))
    return index


def _build_embedder(args: argparse.Namespace) -> BaseEmbedder:
    """Instantiate the embedder selected by --embedder."""
    if args.embedder == "cohere":
        return CohereEmbedder(
            api_key=args.cohere_api_key or None,
            model=args.cohere_model,
            input_type="search_query",
        )
    # default: sentence-transformers
    return SentenceTransformerEmbedder(args.model)


def _build_retrievers(names: List[str], index: HNSWIndex, embedder: BaseEmbedder) -> Dict[str, BaseRetriever]:
    """Instantiate each requested retriever with the shared index."""
    return {name: RETRIEVERS[name](embedder=embedder, index=index) for name in names}


def _print_results(name: str, results: List[RetrievalResult]) -> None:
    width = 72
    logger.info("  Retriever : %s", name)
    logger.info("%s", "─" * width)
    if not results:
        logger.info("  (no results)")
    for rank, r in enumerate(results, start=1):
        score_str = f"{r.score:.4f}"
        logger.info("  [%d] score=%s  id=%s", rank, score_str, r.doc.id)
        snippet = r.doc.text.replace("\n", " ")[:200]
        if len(r.doc.text) > 200:
            snippet += " …"
        logger.info("      %s", snippet)
    logger.info("%s", "=" * width)


def _run_query(
    query: str,
    retrievers: Dict[str, BaseRetriever],
    top_k: int,
) -> None:
    for name, retriever in retrievers.items():
        # retrieve() returns List[List[RetrievalResult]]; index [0] for single query
        batch_results = retriever.retrieve(query, top_k=top_k, source_label=name)
        _print_results(name, batch_results[0])


def _interactive_loop(
    retrievers: Dict[str, BaseRetriever],
    top_k: int,
) -> None:
    logger.info("Interactive mode — type a query and press Enter.  Ctrl-C or 'q' to quit.")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            break
        _run_query(query, retrievers, top_k)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query documents with one or more retrievers.  "
            "Omit QUERY to enter interactive mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query string to search for.  Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a pre-built FAISS index file (.faiss).",
    )
    parser.add_argument(
        "--retriever",
        action="append",
        dest="retrievers",
        choices=list(RETRIEVERS),
        metavar=f"{{{','.join(RETRIEVERS)}}}",
        default=None,
        help=(
            "Retriever to use.  Repeatable to compare multiple at once.  "
            f"Choices: {', '.join(RETRIEVERS)}.  (default: cosine)"
        ),
    )

    # ── Embedder selection ────────────────────────────────────────────────────
    emb_group = parser.add_argument_group("embedder")
    emb_group.add_argument(
        "--embedder",
        choices=["sentence-transformers", "cohere"],
        default="sentence-transformers",
        help="Embedding backend to use.  (default: sentence-transformers)",
    )
    emb_group.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name/path.  (default: all-MiniLM-L6-v2)",
    )
    emb_group.add_argument(
        "--cohere-model",
        default="embed-english-v3.0",
        dest="cohere_model",
        help="Cohere embed model name.  (default: embed-english-v3.0)",
    )
    emb_group.add_argument(
        "--cohere-api-key",
        default=None,
        dest="cohere_api_key",
        help="Cohere API key.  Falls back to COHERE_API_KEY env var if omitted.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results per retriever.  (default: 5)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging (shows NER masking details, etc.).",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    retriever_names: List[str] = args.retrievers or ["cosine"]

    # ── Load index & documents ────────────────────────────────────────────────
    index = _load_index(args.index)

    # ── Load embedding model ──────────────────────────────────────────────────
    logger.info("Initializing embedder '%s' ...", args.embedder)
    embedder = _build_embedder(args)

    # ── Build retrievers ──────────────────────────────────────────────────────
    retrievers = _build_retrievers(retriever_names, index, embedder)

    # ── One-shot or interactive ───────────────────────────────────────────────
    if args.query:
        logger.info("Query: \"%s\"", args.query)
        _run_query(args.query, retrievers, args.top_k)
    else:
        _interactive_loop(retrievers, args.top_k)


if __name__ == "__main__":
    main()
