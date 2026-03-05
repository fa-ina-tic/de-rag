"""Tests for de_rag.logger."""
import logging

import pytest

from de_rag.logger import get_logger, setup_logging


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("de_rag.test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = get_logger("de_rag.mymodule")
        assert logger.name == "de_rag.mymodule"

    def test_different_names_give_different_loggers(self):
        a = get_logger("de_rag.a")
        b = get_logger("de_rag.b")
        assert a is not b

    def test_same_name_gives_same_logger(self):
        a = get_logger("de_rag.same")
        b = get_logger("de_rag.same")
        assert a is b


class TestSetupLogging:
    def _get_root(self):
        return logging.getLogger("de_rag")

    def test_sets_level(self):
        setup_logging(logging.DEBUG)
        assert self._get_root().level == logging.DEBUG
        # Reset
        setup_logging(logging.INFO)

    def test_adds_stream_handler(self):
        root = self._get_root()
        # Remove all handlers first
        root.handlers.clear()
        setup_logging()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_removes_null_handlers(self):
        root = self._get_root()
        root.handlers.clear()
        root.addHandler(logging.NullHandler())
        setup_logging()
        null_handlers = [h for h in root.handlers if isinstance(h, logging.NullHandler)]
        assert len(null_handlers) == 0

    def test_idempotent_handler_count(self):
        """Calling setup_logging twice should not double-add handlers."""
        root = self._get_root()
        root.handlers.clear()
        setup_logging()
        count_after_first = len(root.handlers)
        setup_logging()
        assert len(root.handlers) == count_after_first

    def test_propagation_disabled(self):
        setup_logging()
        assert self._get_root().propagate is False

    def test_default_level_is_info(self):
        root = self._get_root()
        root.handlers.clear()
        setup_logging()
        assert root.level == logging.INFO
