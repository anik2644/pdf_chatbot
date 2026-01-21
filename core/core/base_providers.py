"""
Abstract base classes for LLM and Embeddings providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_llm(
            self,
            model_name: Optional[str] = None,
            temperature: float = 0.7,
            **kwargs
    ) -> BaseLanguageModel:
        """Get an LLM instance."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Return list of supported models."""
        pass


class BaseEmbeddingsProvider(ABC):
    """Abstract base class for Embeddings providers."""

    @abstractmethod
    def get_embeddings(
            self,
            model_name: Optional[str] = None,
            **kwargs
    ) -> Embeddings:
        """Get an embeddings instance."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass