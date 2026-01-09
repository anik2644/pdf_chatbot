"""
Vector store factory for creating and managing vector stores.
"""

from typing import List, Optional, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

from pdf_qa_system.config.settings import get_settings
from pdf_qa_system.utils.logger import get_logger
from pdf_qa_system.utils.helpers import ensure_directory

logger = get_logger(__name__)


class VectorStoreFactory:
    """Factory for creating and managing vector stores."""

    # Default embedding model
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
            self,
            embeddings: Optional[Embeddings] = None,
            store_type: Optional[str] = None,
            embedding_model: Optional[str] = None
    ):
        self.settings = get_settings()
        self.store_type = (store_type or self.settings.default_vector_store).lower()

        if embeddings is None:
            self.embeddings = self._create_embeddings(embedding_model)
        else:
            self.embeddings = embeddings

    def _create_embeddings(self, model_name: Optional[str] = None) -> Embeddings:
        """Create HuggingFace embeddings with HF_TOKEN from environment."""
        import os
        from dotenv import load_dotenv

        load_dotenv()

        hf_token = os.getenv("HF_TOKEN")
        model = model_name or self.DEFAULT_EMBEDDING_MODEL

        logger.info(f"Creating HuggingFace embeddings with model: {model}")

        # Configure model kwargs with token if available
        model_kwargs = {}
        if hf_token:
            model_kwargs["token"] = hf_token
            logger.info("Using HF_TOKEN for authentication")
        else:
            logger.warning("HF_TOKEN not found in environment variables")

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True}
        )

    def create_from_documents(
            self,
            documents: List[Document],
            collection_name: str = "default",
            persist: bool = True,
            **kwargs
    ) -> VectorStore:
        """Create a vector store from documents."""
        logger.info(
            f"Creating {self.store_type} vector store with {len(documents)} documents"
        )

        if self.store_type == "chroma":
            return self._create_chroma(documents, collection_name, persist, **kwargs)
        elif self.store_type == "faiss":
            return self._create_faiss(documents, persist, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

    def _create_chroma(
            self,
            documents: List[Document],
            collection_name: str,
            persist: bool,
            **kwargs
    ) -> VectorStore:
        """Create a Chroma vector store."""
        persist_directory = None
        if persist:
            persist_directory = str(
                ensure_directory(self.settings.chroma_persist_directory)
            )

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )

    def _create_faiss(
            self,
            documents: List[Document],
            persist: bool = True,
            **kwargs
    ) -> VectorStore:
        """Create a FAISS vector store and optionally persist it."""
        faiss_index = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            **kwargs
        )

        if persist:
            persist_directory = ensure_directory(self.settings.faiss_persist_directory)
            faiss_index.save_local(str(persist_directory))
            logger.info(f"FAISS index saved to {persist_directory}")

        return faiss_index

    def load_existing(
            self,
            collection_name: str = "default",
            **kwargs
    ) -> Optional[VectorStore]:
        """Load an existing vector store."""
        logger.info(f"Loading existing {self.store_type} vector store")

        if self.store_type == "chroma":
            return self._load_chroma(collection_name, **kwargs)
        elif self.store_type == "faiss":
            return self._load_faiss(**kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

    def _load_chroma(
            self,
            collection_name: str,
            **kwargs
    ) -> Optional[VectorStore]:
        """Load an existing Chroma vector store."""
        persist_directory = self.settings.chroma_persist_directory
        if not Path(persist_directory).exists():
            logger.warning(f"Chroma directory not found: {persist_directory}")
            return None

        logger.info(f"Loading Chroma from {persist_directory}")
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            **kwargs
        )

    def _load_faiss(self, **kwargs) -> Optional[VectorStore]:
        """Load an existing FAISS vector store."""
        persist_directory = self.settings.faiss_persist_directory
        if not Path(persist_directory).exists():
            logger.warning(f"FAISS directory not found: {persist_directory}")
            return None

        logger.info(f"Loading FAISS from {persist_directory}")
        return FAISS.load_local(
            str(persist_directory),
            self.embeddings,
            allow_dangerous_deserialization=True,
            **kwargs
        )


def load_vector_db(
        vector_db_path: Optional[str] = None,
        embedding_model: Optional[str] = None
) -> FAISS:
    """
    Convenience function to load FAISS vector store.

    Args:
        vector_db_path: Path to the FAISS index. If None, uses settings.
        embedding_model: HuggingFace model name. If None, uses default.

    Returns:
        Loaded FAISS vector store.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("📦 Loading FAISS vector database...")

    # Get HF token
    hf_token = os.getenv("HF_TOKEN")

    # Create embeddings
    model_kwargs = {"token": hf_token} if hf_token else {}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model or VectorStoreFactory.DEFAULT_EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True}
    )

    # Determine path
    if vector_db_path is None:
        settings = get_settings()
        vector_db_path = settings.faiss_persist_directory

    # Load vector store
    vectorstore = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✅ Vector DB loaded successfully!\n")
    return vectorstore


def get_vector_store_factory(
        store_type: str = "faiss",
        embedding_model: Optional[str] = None
) -> VectorStoreFactory:
    """
    Convenience function to get a configured VectorStoreFactory.

    Args:
        store_type: Type of vector store ("faiss" or "chroma")
        embedding_model: HuggingFace model name

    Returns:
        Configured VectorStoreFactory instance
    """
    return VectorStoreFactory(
        store_type=store_type,
        embedding_model=embedding_model
    )