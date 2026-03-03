"""cli.py – interactive retrieval tool for de-rag.

Loads a pre-built HNSW index (produced by de_rag.index), initialises an
embedding model, and lets you search with one or more retrievers
either in a single-shot or interactive REPL mode.

Index files are produced by running:
    python -m de_rag.index --model <model> --output <name>.faiss ...

The index file bundles both the FAISS vectors and document metadata.

Usage examples
--------------
# one-shot query
python -m de_rag.cli "CRISPR mechanism" \\
    --index wikitext.faiss \\
    --retriever cosine --retriever nonneg

# interactive REPL (omit the query argument)
python -m de_rag.cli \\
    --index wikitext.faiss \\
    --retriever cosine --retriever nonneg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

from de_rag.classes import RetrievalResult
from de_rag.index import HNSWIndex
from de_rag.retriever import BaseRetriever, CosineRetriever, NonNegativeRetriever

# ── Retriever registry ────────────────────────────────────────────────────────

RETRIEVERS: Dict[str, type[BaseRetriever]] = {
    "cosine": CosineRetriever,
    "nonneg": NonNegativeRetriever,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_index(index_path: Path) -> HNSWIndex:
    """Load a pre-built HNSWIndex (vectors + docs bundled in one file)."""
    if not index_path.exists():
        print(f"error: index file '{index_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"[Index] Loading '{index_path}' ...")
    index = HNSWIndex.load(str(index_path))
    print(f"[Index] Ready — {len(index):,} vectors, {len(index.docs):,} documents.")
    return index


def _build_retrievers(names: List[str], index: HNSWIndex) -> Dict[str, BaseRetriever]:
    """Instantiate each requested retriever with the shared index."""
    return {name: RETRIEVERS[name](index=index) for name in names}


def _print_results(name: str, results: List[RetrievalResult]) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  Retriever : {name}")
    print(f"{'─' * width}")
    if not results:
        print("  (no results)")
    for rank, r in enumerate(results, start=1):
        score_str = f"{r.score:.4f}"
        header = f"  [{rank}] score={score_str}  id={r.doc.id}"
        print(header)
        snippet = r.doc.text.replace("\n", " ")[:200]
        if len(r.doc.text) > 200:
            snippet += " …"
        print(f"      {snippet}")
    print(f"{'=' * width}")


def _run_query(
    query: str,
    retrievers: Dict[str, BaseRetriever],
    embedder: SentenceTransformer,
    top_k: int,
) -> None:
    query_emb = embedder.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    for name, retriever in retrievers.items():
        # retrieve() returns List[List[RetrievalResult]]; index [0] for single query
        batch_results = retriever.retrieve(query_emb, top_k=top_k, source_label=name)
        _print_results(name, batch_results[0])


def _interactive_loop(
    retrievers: Dict[str, BaseRetriever],
    embedder: SentenceTransformer,
    top_k: int,
) -> None:
    print("\nInteractive mode — type a query and press Enter.  Ctrl-C or 'q' to quit.")
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
        _run_query(query, retrievers, embedder, top_k)


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
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name/path.  (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results per retriever.  (default: 5)",
    )
    args = parser.parse_args()

    retriever_names: List[str] = args.retrievers or ["cosine"]

    # ── Load index & documents ────────────────────────────────────────────────
    index = _load_index(args.index)

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"[Model] Loading '{args.model}' ...")
    embedder = SentenceTransformer(args.model)

    # ── Build retrievers ──────────────────────────────────────────────────────
    retrievers = _build_retrievers(retriever_names, index)

    # ── One-shot or interactive ───────────────────────────────────────────────
    if args.query:
        print(f'\n[Query] "{args.query}"')
        _run_query(args.query, retrievers, embedder, args.top_k)
    else:
        _interactive_loop(retrievers, embedder, args.top_k)


if __name__ == "__main__":
    main()
