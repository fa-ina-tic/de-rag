from abc import ABC, abstractmethod
from typing import Iterator
from datasets import load_dataset, Dataset


class BaseDataLoader(ABC):
    """Iterable data loader that yields raw text strings."""

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Yield one text string per item."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items."""


class WikitextDataLoader(BaseDataLoader):
    """Streams text from the Wikitext dataset via HuggingFace datasets.

    Parameters
    ----------
    name : str
        Dataset config name, e.g. "wikitext-2-raw-v1" or "wikitext-103-raw-v1".
    split : str
        Dataset split — "train", "validation", or "test".
    skip_empty : bool
        Skip blank / whitespace-only lines (default True).
    """

    def __init__(
        self,
        name: str = "wikitext-2-raw-v1",
        split: str = "train",
        skip_empty: bool = True,
    ):
        self._dataset: Dataset = load_dataset("wikitext", name, split=split)  # type: ignore[assignment]
        self.skip_empty = skip_empty

    def __iter__(self) -> Iterator[str]:
        for row in self._dataset:
            text = row["text"]
            if self.skip_empty and not text.strip():
                continue
            yield text

    def __len__(self) -> int:
        if self.skip_empty:
            return sum(1 for row in self._dataset if row["text"].strip())
        return len(self._dataset)
