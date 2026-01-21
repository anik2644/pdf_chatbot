"""
Factory for creating LLM instances from various providers.
"""

from typing import Optional, Dict, Type
from langchain_core.language_models.base import BaseLanguageModel

from core.config.settings import get_settings
from core.core.base_providers import BaseLLMProvider
from core.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def supported_models(self) -> list[str]:
        return [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
        ]

    def get_llm(
            self,
            model_name: Optional[str] = None,
            temperature: float = 0.7,
            **kwargs
    ) -> BaseLanguageModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self.api_key,
            temperature=temperature,
            **kwargs
        )


class GroqLLMProvider(BaseLLMProvider):
    """Groq LLM provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def supported_models(self) -> list[str]:
        return [
            # "llama-3.1-70b-versatile",
            # "llama-3.1-8b-instant",
            # "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]

    def get_llm(
            self,
            model_name: Optional[str] = None,
            temperature: float = 0.7,
            **kwargs
    ) -> BaseLanguageModel:
        from langchain_groq import ChatGroq

        model = model_name or "llama-3.1-8b-instant"
        return ChatGroq(
            model=model,
            groq_api_key=self.api_key,
            temperature=temperature,
            **kwargs
        )


class HuggingFaceLLMProvider(BaseLLMProvider):
    """HuggingFace LLM provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def supported_models(self) -> list[str]:
        return [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "google/gemma-7b-it",
        ]

    def get_llm(
            self,
            model_name: Optional[str] = None,
            temperature: float = 0.7,
            **kwargs
    ) -> BaseLanguageModel:
        from langchain_huggingface import HuggingFaceEndpoint

        model = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
        return HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=self.api_key,
            temperature=temperature,
            **kwargs
        )


class HuggingFacePipelineLLMProvider(BaseLLMProvider):
    """HuggingFace Pipeline LLM provider for local models."""

    def __init__(self, api_key: Optional[str] = None):
        # API key is optional as local models might not need one
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "huggingface_pipeline"

    @property
    def supported_models(self) -> list[str]:
        # List common local models or indicate generic support
        return [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            # Add other common local models if needed
        ]

    def get_llm(
            self,
            model_name: Optional[str] = None,
            temperature: float = 0.3, # Using user's default
            **kwargs
    ) -> BaseLanguageModel:
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline

        model_id = model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Default from user's snippet

        logger.info(f"Loading local LLM pipeline: {model_id}...")
        pipe = pipeline(
            "text-generation",
            model=model_id,
            max_new_tokens=kwargs.pop("max_new_tokens", 512), # Allow overriding max_new_tokens
            temperature=temperature,
            do_sample=kwargs.pop("do_sample", True), # Allow overriding do_sample
            **kwargs # Pass any remaining kwargs to the pipeline
        )
        return HuggingFacePipeline(pipeline=pipe)


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]

    def get_llm(
            self,
            model_name: Optional[str] = None,
            temperature: float = 0.7,
            **kwargs
    ) -> BaseLanguageModel:
        from langchain_openai import ChatOpenAI

        model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            api_key=self.api_key,
            temperature=temperature,
            **kwargs
        )


class LLMFactory:
    """Factory for creating LLM instances."""

    _providers: Dict[str, Type[BaseLLMProvider]] = {
        "gemini": GeminiLLMProvider,
        "groq": GroqLLMProvider,
        "huggingface": HuggingFaceLLMProvider,
        "huggingface_pipeline": HuggingFacePipelineLLMProvider, # Register new provider
        "openai": OpenAILLMProvider,
    }

    def __init__(self):
        self.settings = get_settings()

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Register a new LLM provider."""
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered LLM provider: {name}")

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers."""
        return list(cls._providers.keys())

    def create(
            self,
            provider: Optional[str] = None,
            model_name: Optional[str] = None,
            temperature: float = 0.7,
            **kwargs
    ) -> BaseLanguageModel:
        """Create an LLM instance."""
        provider_name = (provider or self.settings.default_llm_provider).lower()

        if provider_name not in self._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {self.get_available_providers()}"
            )

        api_key = self.settings.get_api_key(provider_name)
        # HuggingFacePipeline does not require an API key
        if not api_key and provider_name not in ["huggingface_pipeline"]:
            raise ValueError(f"API key not found for provider: {provider_name}")

        provider_instance = self._providers[provider_name](api_key)
        model = model_name or self.settings.get_default_model(provider_name)

        logger.info(f"Creating LLM: provider={provider_name}, model={model}")
        return provider_instance.get_llm(model_name=model, temperature=temperature, **kwargs)