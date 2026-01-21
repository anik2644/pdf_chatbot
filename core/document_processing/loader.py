"""
Document loading utilities.
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

from core.utils.logger import get_logger
from core.utils.helpers import validate_file_path, get_file_extension

logger = get_logger(__name__)


class DocumentLoader:
    """Handles loading documents from various sources."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

    def __init__(self):
        self._loaders = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".md": self._load_text,
            ".docx": self._load_docx,
        }

    def load(self, file_path: str) -> List[Document]:
        """Load a document from the given file path."""
        path = validate_file_path(file_path)
        extension = get_file_extension(file_path)

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Loading document: {file_path}")
        loader_func = self._loaders[extension]
        documents = loader_func(path)

        logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
        return documents

    def _load_pdf(self, path: Path) -> List[Document]:
        """Load a PDF document."""
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path))
        return loader.load()

    def _load_text(self, path: Path) -> List[Document]:
        """Load a text or markdown document."""
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    def _load_docx(self, path: Path) -> List[Document]:
        """Load a Word document."""
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(str(path))
        return loader.load()

    def load_directory(
            self,
            directory_path: str,
            glob_pattern: str = "**/*"
    ) -> List[Document]:
        """Load all supported documents from a directory."""
        from langchain_community.document_loaders import DirectoryLoader

        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        all_documents = []
        for extension in self.SUPPORTED_EXTENSIONS:
            pattern = f"{glob_pattern}{extension}"
            try:
                loader = DirectoryLoader(
                    str(path),
                    glob=pattern,
                    show_progress=True
                )
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Error loading {extension} files: {e}")

        logger.info(f"Loaded {len(all_documents)} document(s) from {directory_path}")
        return all_documents