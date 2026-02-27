from typing import Union, Tuple, List
from classes import Document
from pathlib import Path
import re

import numpy as np

class PDFLoader:
    """
    Loads a PDF and produces two levels of text granularity:
        - summaries : one entry per page  (global-level)
        - chunks    : fixed-size sliding window over sentences (local-level)

    Args:
        chunk_size:    approximate number of words per chunk
        chunk_overlap: number of words to overlap between consecutive chunks
        min_chunk_len: discard chunks shorter than this (words)
    """

    def __init__(
        self,
        chunk_size: int = 150,
        chunk_overlap: int = 30,
        min_chunk_len: int = 20,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    def load(self, pdf_path: Union[str, Path]) -> List[Document]:
        """
        Parse a PDF file and return (summary_docs, chunk_docs).

        summary_docs : one Document per page — used for the global index
        chunk_docs   : sliding-window word chunks — used for the local index

        Embeddings are left as empty arrays (filled later by EmbeddingModel).
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required: pip install pypdf")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        chunk_docs:   List[Document] = []

        for page_num, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            clean    = self._clean_text(raw_text)

            if len(clean.split()) < self.min_chunk_len:
                continue  # skip near-empty pages

            # --- chunks: sliding window over words ---
            words  = clean.split()
            stride = max(1, self.chunk_size - self.chunk_overlap)
            for i, start in enumerate(range(0, len(words), stride)):
                chunk_words = words[start : start + self.chunk_size]
                if len(chunk_words) < self.min_chunk_len:
                    continue
                chunk_docs.append(Document(
                    id=f"{pdf_path.stem}_page{page_num}_chunk{i}",
                    text=" ".join(chunk_words),
                    embedding=np.array([]),
                    doc_type="chunk",
                ))

        print(f"[PDFLoader] '{pdf_path.name}': "
              f"{len(reader.pages)} pages → "
              f"{len(chunk_docs)} chunks")

        return chunk_docs

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic cleanup: collapse whitespace, strip control chars."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)  # control chars
        text = re.sub(r"\s+", " ", text)                              # collapse whitespace
        return text.strip()
