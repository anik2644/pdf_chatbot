"""
General helper functions.
"""

import os
from pathlib import Path
from typing import Optional


def validate_file_path(file_path: str) -> Path:
    """Validate that a file exists and return Path object."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    return path


def get_file_extension(file_path: str) -> str:
    """Get the file extension from a path."""
    return Path(file_path).suffix.lower()


def ensure_directory(directory: str) -> Path:
    """Ensure a directory exists, create if it doesn't."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."