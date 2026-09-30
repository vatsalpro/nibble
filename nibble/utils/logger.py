"""
Structured logging module for Nibble.
Configures file and stream loggers with consistent timestamps and formatting.
Writes logs to logs/nibble.log.
"""

import os
import sys
import logging
from pathlib import Path


_LOG_FILE = Path("logs") / "nibble.log"
_LOGGER_INITIALIZED = False


def setup_logger(name: str = "nibble") -> logging.Logger:
    """Get or configure structured logger routing to logs/nibble.log and stderr."""
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)

    if not _LOGGER_INITIALIZED:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler
        try:
            file_handler = logging.FileHandler(str(_LOG_FILE), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        except Exception:
            pass

        # Console Handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

        logging.getLogger().setLevel(logging.INFO)
        _LOGGER_INITIALIZED = True

    return logger


def get_logger(name: str = "nibble") -> logging.Logger:
    """Retrieve logger instance."""
    return setup_logger(name)
