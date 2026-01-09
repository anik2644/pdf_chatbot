"""
Logging utilities.
"""

import logging
import sys
from typing import Optional

# Store configured loggers
_loggers = {}


def setup_logging(level: str = "INFO") -> None:
    """Setup application-wide logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logger_name = name or "pdf_qa_system"

    if logger_name in _loggers:
        return _loggers[logger_name]

    logger = logging.getLogger(logger_name)

    # Only add handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    _loggers[logger_name] = logger
    return logger