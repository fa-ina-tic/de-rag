"""Tests for de_rag.dataloader."""
from unittest.mock import MagicMock, patch

import pytest

from de_rag.dataloader import WikitextDataLoader


def _make_dataset(rows):
    """Return a mock HuggingFace Dataset-like object."""
    mock_ds = MagicMock()
    # Use side_effect on the class-level magic method so each for-loop
    # gets a fresh iterator (setting __iter__ as an instance attribute
    # is ignored by Python's special-method lookup).
    mock_ds.__iter__.side_effect = lambda: iter(rows)
    mock_ds.__len__.return_value = len(rows)
    return mock_ds


ROWS = [
    {"text": "Hello world"},
    {"text": ""},
    {"text": "  "},
    {"text": "Python is great"},
    {"text": "Another sentence"},
]


@pytest.fixture
def loader_with_mock(monkeypatch):
    """WikitextDataLoader backed by a fake dataset (no network call)."""
    mock_ds = _make_dataset(ROWS)
    # Patch load_dataset to return the mock dataset
    with patch("de_rag.dataloader.load_dataset", return_value=mock_ds):
        loader = WikitextDataLoader(name="wikitext-2-raw-v1", split="train", skip_empty=True)
    # Re-attach the mock dataset (constructor already ran, just ensure attribute)
    loader._dataset = mock_ds
    return loader


@pytest.fixture
def loader_no_skip(monkeypatch):
    mock_ds = _make_dataset(ROWS)
    with patch("de_rag.dataloader.load_dataset", return_value=mock_ds):
        loader = WikitextDataLoader(name="wikitext-2-raw-v1", split="train", skip_empty=False)
    loader._dataset = mock_ds
    return loader


class TestWikitextDataLoader:
    def test_init_calls_load_dataset(self):
        with patch("de_rag.dataloader.load_dataset") as mock_ld:
            mock_ld.return_value = _make_dataset([])
            WikitextDataLoader(name="wikitext-2-raw-v1", split="validation")
            mock_ld.assert_called_once_with("wikitext", "wikitext-2-raw-v1", split="validation")

    def test_iter_skips_empty(self, loader_with_mock):
        texts = list(loader_with_mock)
        assert "" not in texts
        assert "  " not in texts
        assert "Hello world" in texts
        assert "Python is great" in texts
        assert "Another sentence" in texts

    def test_iter_no_skip(self, loader_no_skip):
        texts = list(loader_no_skip)
        assert len(texts) == len(ROWS)
        assert "" in texts

    def test_len_skip_empty(self, loader_with_mock):
        non_empty = sum(1 for r in ROWS if r["text"].strip())
        assert len(loader_with_mock) == non_empty

    def test_len_no_skip(self, loader_no_skip):
        assert len(loader_no_skip) == len(ROWS)

    def test_iter_yields_strings(self, loader_with_mock):
        for text in loader_with_mock:
            assert isinstance(text, str)

    def test_skip_empty_default_is_true(self):
        with patch("de_rag.dataloader.load_dataset") as mock_ld:
            mock_ld.return_value = _make_dataset([])
            loader = WikitextDataLoader()
            assert loader.skip_empty is True

    def test_multiple_iterations(self, loader_with_mock):
        """DataLoader can be iterated multiple times (relies on dataset re-iteration)."""
        first = list(loader_with_mock)
        second = list(loader_with_mock)
        assert first == second
