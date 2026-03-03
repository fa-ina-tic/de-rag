from dataclasses import dataclass
from torch import Tensor

@dataclass
class Document:
    id: str
    text: str
    embedding: Tensor
    doc_type: str

@dataclass
class RetrievalResult:
    doc: Document
    score: float
    source: str
