"""
Text chunking strategies.
"""

from enum import Enum
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

from core.config.settings import get_settings
from core.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"


class TextChunker:
    """Handles text chunking with various strategies."""

    def __init__(
            self,
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None,
            strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.strategy = strategy
        self._splitter = self._create_splitter()

    def _create_splitter(self):
        """Create the appropriate text splitter."""
        if self.strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        elif self.strategy == ChunkingStrategy.CHARACTER:
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
        elif self.strategy == ChunkingStrategy.TOKEN:
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        logger.info(
            f"Chunking {len(documents)} document(s) with strategy={self.strategy.value}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

        chunks = self._splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """Split a single text string into chunks."""
        return self._splitter.split_text(text)