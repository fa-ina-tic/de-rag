from __future__ import annotations

import logging
import sys

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the de_rag root logger. Safe to call multiple times."""
    root = logging.getLogger("de_rag")
    # Drop NullHandlers that external libraries may have silently attached
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.NullHandler)]
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_FORMATTER)
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
