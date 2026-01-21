"""
Configuration management using Pydantic Settings.
"""

import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Google / Gemini
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")

    # Groq
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")

    # HuggingFace
    huggingface_api_key: Optional[str] = Field(default=None, alias="HF_TOKEN")

    # Default LLM settings
    default_llm_provider: str = Field(default="huggingface_pipeline", alias="DEFAULT_LLM_PROVIDER")
    default_llm_model: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", alias="DEFAULT_LLM_MODEL")
    default_model_name: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", alias="DEFAULT_MODEL_NAME")

    # Default LLM settings
    # default_llm_provider: str = Field(default="groq", alias="DEFAULT_LLM_PROVIDER")
    # default_llm_model: str = Field(default="llama-3.1-8b-instant", alias="DEFAULT_LLM_MODEL")
    # default_model_name: str = Field(default="llama-3.1-8b-instant", alias="DEFAULT_MODEL_NAME")


    # Default Embeddings settings
    default_embeddings_provider: str = Field(default="huggingface", alias="DEFAULT_EMBEDDINGS_PROVIDER")
    default_embeddings_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="DEFAULT_EMBEDDINGS_MODEL")

    # Vector store settings
    vector_store_type: str = Field(default="faiss", alias="VECTOR_STORE_TYPE")
    default_vector_store: str = Field(default="faiss", alias="DEFAULT_VECTOR_STORE")
    chroma_persist_directory: str = Field(default="./vector_db/chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    faiss_persist_directory: str = Field(default="./vector_db/faiss_db", alias="FAISS_PERSIST_DIRECTORY")

    # Chunking settings
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # PDF Path
    pdf_path: Optional[str] = Field(default=None, alias="PDF_PATH")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        provider = provider.lower().strip()

        key_mapping = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key or self.gemini_api_key,
            "gemini": self.gemini_api_key or self.google_api_key,
            "groq": self.groq_api_key,
            "huggingface": self.huggingface_api_key,
            "hugging_face": self.huggingface_api_key,
            "huggingface_pipeline":self.huggingface_api_key
        }

        return key_mapping.get(provider)

    def get_default_model(self, provider: str) -> str:
        """Get default model for a specific provider."""
        provider = provider.lower().strip()

        model_mapping = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-sonnet-20240229",
            "google": "gemini-1.5-flash",
            "gemini": "gemini-1.5-flash",
            "groq": "llama-3.1-8b-instant",
            "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
            "huggingface_pipeline": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }

        return model_mapping.get(provider, self.default_llm_model)

    def get_default_embeddings_model(self, provider: str) -> str:
        """Get default embeddings model for a specific provider."""
        provider = provider.lower().strip()

        model_mapping = {
            "openai": "text-embedding-3-small",
            "google": "models/embedding-001",
            "gemini": "models/embedding-001",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }

        return model_mapping.get(provider, self.default_embeddings_model)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()