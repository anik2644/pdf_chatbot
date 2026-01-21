"""
Retriever implementations and factory.
"""

from typing import Optional, List
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from core.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverFactory:
    """Factory for creating retrievers from vector stores."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def create_basic_retriever(
            self,
            search_type: str = "similarity",
            k: int = 1,
            **kwargs
    ) -> BaseRetriever:
        """Create a basic retriever."""
        logger.info(f"Creating basic retriever: search_type={search_type}, k={k}")

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, **kwargs}
        )

    def create_mmr_retriever(
            self,
            k: int = 1,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs
    ) -> BaseRetriever:
        """Create a Maximum Marginal Relevance retriever."""
        logger.info(
            f"Creating MMR retriever: k={k}, fetch_k={fetch_k}, lambda={lambda_mult}"
        )

        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
                **kwargs
            }
        )

    def create_threshold_retriever(
            self,
            score_threshold: float = 0.5,
            k: int = 1,
            **kwargs
    ) -> BaseRetriever:
        """Create a retriever with score threshold."""
        logger.info(
            f"Creating threshold retriever: threshold={score_threshold}, k={k}"
        )

        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                "k": k,
                **kwargs
            }
        )