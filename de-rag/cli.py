import argparse
from typing import List, Optional
import textwrap
import sys
from classes import RetrievalResult, Document
from retriever import FrequencyAwareRetriever
from dataloader import PDFLoader
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Embedding  (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model(model_name: str, device: Optional[str] = None):
    """Load a SentenceTransformer model. Returns (model, dim)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("[ERROR] sentence-transformers not installed.  "
                 "Run: pip install sentence-transformers")

    print(f"[Embedder] Loading '{model_name}' ...")
    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    print(f"[Embedder] dim={dim}")
    return model, dim


def embed_documents(
    model,
    docs:           List[Document],
    batch_size:     int  = 32,
    show_progress:  bool = True,
) -> None:
    """Embed all documents in-place (fills doc.embedding)."""
    texts      = [d.text for d in docs]
    embeddings = model.encode(
        texts,
        batch_size         = batch_size,
        normalize_embeddings = True,
        show_progress_bar  = show_progress,
        convert_to_numpy   = True,
    )
    for doc, emb in zip(docs, embeddings):
        doc.embedding = emb


def embed_query(model, query_text: str) -> np.ndarray:
    """Embed a single query string, L2-normalised."""
    return model.encode(
        [query_text],
        normalize_embeddings = True,
        convert_to_numpy     = True,
    )[0]

# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_results(results: List[RetrievalResult], max_chars: int = 300) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        snippet = r.doc.text[:max_chars].replace("\n", " ")
        elipsis = "..." if len(r.doc.text) > max_chars else ""
        lines.append(
            f"[{i}] {r.source.upper()}  score={r.score:.4f}  id={r.doc.id}\n"
            f"    {snippet}{elipsis}"
        )
    return "\n\n".join(lines)


def _separator(char: str = "─", width: int = 60) -> str:
    return char * width

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hadamard_rag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Hadamard Frequency-Band RAG
            ───────────────────────────
            Indexes a PDF, decomposes query embeddings via WHT,
            and retrieves global (page-level) + local (chunk-level) context.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python de-rag/cli.py --pdf paper.pdf --query "What is the main idea?"
              python de-rag/cli.py --pdf paper.pdf --query "Methods used?" --model all-mpnet-base-v2 --show-diagnostics
              python de-rag/cli.py --pdf paper.pdf --interactive
        """),
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--pdf", required=True, metavar="PATH",
        help="Path to the input PDF file.",
    )

    # ── Query mode (mutually exclusive) ───────────────────────────────────────
    qmode = p.add_mutually_exclusive_group(required=True)
    qmode.add_argument(
        "--query", "-q", metavar="TEXT",
        help="Single query string.",
    )
    qmode.add_argument(
        "--interactive", "-i", action="store_true",
        help="Enter an interactive query loop (type 'quit' to exit).",
    )

    # ── Embedding model ───────────────────────────────────────────────────────
    p.add_argument(
        "--model", "-m",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help=(
            "SentenceTransformer model name. "
            "Default: all-MiniLM-L6-v2 (384-dim, fast). "
            "Other options: all-mpnet-base-v2 (768-dim), "
            "BAAI/bge-small-en-v1.5, intfloat/e5-base-v2"
        ),
    )
    p.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Torch device: cpu | cuda | mps  (default: auto-detect).",
    )
    p.add_argument(
        "--batch-size", type=int, default=32, metavar="N",
        help="Embedding batch size (default: 32).",
    )

    # ── Retrieval parameters ──────────────────────────────────────────────────
    p.add_argument(
        "--top-k-global", type=int, default=3, metavar="K",
        help="Number of page-level summaries to retrieve (default: 3).",
    )
    p.add_argument(
        "--top-k-local", type=int, default=5, metavar="K",
        help="Number of fine-grained chunks to retrieve (default: 5).",
    )
    p.add_argument(
        "--low-ratio", type=float, default=0.5, metavar="R",
        help=(
            "Fraction of WHT coefficients assigned to the global (low-freq) band. "
            "Range: 0.1 – 0.9  (default: 0.5)."
        ),
    )

    # ── Chunking parameters ───────────────────────────────────────────────────
    p.add_argument(
        "--chunk-size", type=int, default=150, metavar="N",
        help="Target words per chunk (default: 150).",
    )
    p.add_argument(
        "--chunk-overlap", type=int, default=30, metavar="N",
        help="Word overlap between consecutive chunks (default: 30).",
    )

    # ── Output control ────────────────────────────────────────────────────────
    p.add_argument(
        "--max-chars", type=int, default=300, metavar="N",
        help="Max characters shown per retrieved passage (default: 300).",
    )
    p.add_argument(
        "--show-diagnostics", action="store_true",
        help="Print WHT energy split and sub-vector cosine for each query.",
    )

    return p


def run_query(
    query_text:      str,
    retriever:       FrequencyAwareRetriever,
    model,
    args:            argparse.Namespace,
) -> None:
    """Embed a query, retrieve, and print results."""
    q_emb = embed_query(model, query_text)

    if args.show_diagnostics:
        d = retriever.diagnostics(q_emb)
        print(f"  [WHT] low_energy={d['low_ratio']:.3f}  "
              f"high_energy={d['high_ratio']:.3f}  "
              f"cosine(q_g,q_l)={d['cosine_g_l']:.4f}")

    results = retriever.retrieve(
        q_emb,
        top_k_global = args.top_k_global,
        top_k_local  = args.top_k_local,
    )

    if not results:
        print("  (no results)")
        return

    print(_format_results(results, max_chars=args.max_chars))


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if not (0.05 <= args.low_ratio <= 0.95):
        parser.error("--low-ratio must be between 0.05 and 0.95")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be less than --chunk-size")

    print(_separator("═"))
    print("  Hadamard Frequency-Band RAG")
    print(_separator("═"))

    # ── Load PDF ──────────────────────────────────────────────────────────────
    loader = PDFLoader(
        chunk_size    = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
    )
    chunk_docs = loader.load(args.pdf)

    if not chunk_docs:
        sys.exit("[ERROR] No usable text extracted from PDF.")

    all_docs = chunk_docs

    # ── Load embedding model ──────────────────────────────────────────────────
    model, dim = load_embedding_model(args.model)

    # ── Embed documents ───────────────────────────────────────────────────────
    print(f"[Embedder] Embedding {len(all_docs)} documents ...")
    embed_documents(model, all_docs, batch_size=args.batch_size, show_progress=True)

    # ── Build retriever ───────────────────────────────────────────────────────
    retriever = FrequencyAwareRetriever(dim=dim, low_ratio=args.low_ratio)
    retriever.add_documents(all_docs)

    print(_separator())
    print(f"  Index ready:  {len(retriever.summary_docs)} summaries  |  "
          f"{len(retriever.chunk_docs)} chunks")
    print(f"  WHT dim: {retriever.decomposer.n}  |  "
          f"low-band: {len(retriever.decomposer.low_idx)}  |  "
          f"high-band: {len(retriever.decomposer.high_idx)}")
    print(_separator())

    # ── Query ─────────────────────────────────────────────────────────────────
    if args.query:
        print(f"Query: {args.query}")
        print(_separator())
        run_query(args.query, retriever, model, args)

    elif args.interactive:
        print("Interactive mode — type your query and press Enter.")
        print("Commands: 'quit' or 'exit' to stop, 'help' to show options.\n")
        while True:
            try:
                raw = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not raw:
                continue
            if raw.lower() in {"quit", "exit", "q"}:
                print("Bye.")
                break
            if raw.lower() == "help":
                print(
                    "  quit / exit  — stop\n"
                    "  diag on/off  — toggle diagnostics\n"
                    "  <any text>   — run as query\n"
                )
                continue
            if raw.lower() in {"diag on", "diag off"}:
                args.show_diagnostics = raw.lower().endswith("on")
                print(f"  Diagnostics {'enabled' if args.show_diagnostics else 'disabled'}.")
                continue

            print(_separator())
            run_query(raw, retriever, model, args)
            print(_separator())


if __name__ == "__main__":
    main()