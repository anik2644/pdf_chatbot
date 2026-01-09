"""
Factory for creating Embeddings instances from various providers.
"""

from typing import Optional, Dict, Type
from langchain_core.embeddings import Embeddings

from pdf_qa_system.config.settings import get_settings
from pdf_qa_system.core.base_providers import BaseEmbeddingsProvider
from pdf_qa_system.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiEmbeddingsProvider(BaseEmbeddingsProvider):
    """Google Gemini embeddings provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "gemini"

    def get_embeddings(
            self,
            model_name: Optional[str] = None,
            **kwargs
    ) -> Embeddings:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = model_name or "models/embedding-001"
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=self.api_key,
            **kwargs
        )


class HuggingFaceEmbeddingsProvider(BaseEmbeddingsProvider):
    """HuggingFace embeddings provider (local)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def get_embeddings(
            self,
            model_name: Optional[str] = None,
            **kwargs
    ) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(
            model_name=model,
            **kwargs
        )


class OpenAIEmbeddingsProvider(BaseEmbeddingsProvider):
    """OpenAI embeddings provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_embeddings(
            self,
            model_name: Optional[str] = None,
            **kwargs
    ) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        model = model_name or "text-embedding-3-small"
        return OpenAIEmbeddings(
            model=model,
            api_key=self.api_key,
            **kwargs
        )


class EmbeddingsFactory:
    """Factory for creating Embeddings instances."""

    _providers: Dict[str, Type[BaseEmbeddingsProvider]] = {
        "gemini": GeminiEmbeddingsProvider,
        "huggingface": HuggingFaceEmbeddingsProvider,
        "openai": OpenAIEmbeddingsProvider,
    }

    def __init__(self):
        self.settings = get_settings()

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseEmbeddingsProvider]) -> None:
        """Register a new embeddings provider."""
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered embeddings provider: {name}")

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers."""
        return list(cls._providers.keys())

    def create(
            self,
            provider: Optional[str] = None,
            model_name: Optional[str] = None,
            **kwargs
    ) -> Embeddings:
        """Create an embeddings instance."""
        provider_name = (provider or self.settings.default_embeddings_provider).lower()

        if provider_name not in self._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {self.get_available_providers()}"
            )

        api_key = self.settings.get_api_key(provider_name)
        # HuggingFace embeddings can work without API key (local models)
        if not api_key and provider_name != "huggingface":
            raise ValueError(f"API key not found for provider: {provider_name}")

        provider_instance = self._providers[provider_name](api_key)
        model = model_name or self.settings.default_embeddings_model

        logger.info(f"Creating embeddings: provider={provider_name}, model={model}")
        return provider_instance.get_embeddings(model_name=model, **kwargs)