from dataclasses import dataclass, field
import numpy as np

@dataclass
class Document:
    id: str
    text: str
    embedding: np.ndarray
    doc_type: str

@dataclass
class RetrievalResult:
    doc: Document
    score: float
    source: str
