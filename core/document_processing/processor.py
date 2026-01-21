"""
Document processor orchestrating loading and chunking.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.document_processing.loader import DocumentLoader
from core.document_processing.chunker import TextChunker, ChunkingStrategy
from core.utils.logger import get_logger
import re
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
        text = self.loader.extract_text_from_pdf(file_path)
        print(f"file type: {type(text)}")

        segments = self.split_by_tags(text)

        print(f"📘 Loaded PDF with {len(segments)} segments")

        if not segments:
            # Fallback to text splitting if no tags found
            print("📘 No tags found, using text splitter...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            segments = splitter.split_text(text)

        # Convert strings to Document objects
        chunks = [
            Document(
                page_content=segment,
                metadata={"source": file_path}
            )
            for segment in segments
        ]

        return chunks


    def split_by_tags(self, text):
        pattern = r'\(START#\)(.*?)\(#END\)'
        segments = re.findall(pattern, text, re.DOTALL)
        cleaned_segments = []
        for segment in segments:
            cleaned_segment = re.sub(r'\s+', ' ', segment.strip())
            cleaned_segments.append(cleaned_segment)
        return cleaned_segments

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