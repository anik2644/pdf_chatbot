"""
PDF search tool for document retrieval.
"""

from typing import Optional, List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from core.agents.tools.base_tool import BaseTool
from core.utils.logger import get_logger
from core.utils.helpers import truncate_text

logger = get_logger(__name__)


class PDFSearchTool(BaseTool):
    """Tool for searching through PDF documents."""

    def __init__(
            self,
            retriever: BaseRetriever,
            k: int = 4,
            include_metadata: bool = True
    ):
        self.retriever = retriever
        self.k = k
        self.include_metadata = include_metadata

    @property
    def name(self) -> str:
        return "pdf_search"

    @property
    def description(self) -> str:
        return (
            "Search through PDF documents to find relevant information. "
            "Input should be a search query or question. "
            "Returns relevant passages from the documents."
        )

    def _run(self, query: str, **kwargs) -> str:
        """Search the PDF documents."""
        logger.info(f"Searching PDFs for: {truncate_text(query, 100)}")

        try:
            documents = self.retriever.invoke(query)

            if not documents:
                return "No relevant information found in the documents."

            results = []
            for i, doc in enumerate(documents[:self.k], 1):
                result = f"[Result {i}]\n{doc.page_content}"

                if self.include_metadata:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    result += f"\n(Source: {source}, Page: {page})"

                results.append(result)

            return "\n\n".join(results)

        except Exception as e:
            logger.error(f"Error searching PDFs: {e}")
            return f"Error searching documents: {str(e)}"

    def search_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Search and return documents with relevance scores."""
        # Note: This requires the underlying vector store to support similarity_search_with_score
        try:
            return self.retriever.vectorstore.similarity_search_with_score(query, k=self.k)
        except AttributeError:
            logger.warning("Vector store doesn't support similarity_search_with_score")
            return [(doc, 0.0) for doc in self.retriever.invoke(query)]