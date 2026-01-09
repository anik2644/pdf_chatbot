"""
Document processor orchestrating loading and chunking.
"""

from typing import List, Optional
from langchain_core.documents import Document

from pdf_qa_system.document_processing.loader import DocumentLoader
from pdf_qa_system.document_processing.chunker import TextChunker, ChunkingStrategy
from pdf_qa_system.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Orchestrates document loading and processing."""

    def __init__(
            self,
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None,
            chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy
        )

    def process_file(self, file_path: str) -> List[Document]:
        """Load and process a single file."""
        documents = self.loader.load(file_path)
        chunks = self.chunker.chunk_documents(documents)

        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source_file"] = file_path

        return chunks

    def process_directory(
            self,
            directory_path: str,
            glob_pattern: str = "**/*"
    ) -> List[Document]:
        """Load and process all documents in a directory."""
        documents = self.loader.load_directory(directory_path, glob_pattern)
        chunks = self.chunker.chunk_documents(documents)
        return chunks

    def process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """Load and process multiple files."""
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        return all_chunks