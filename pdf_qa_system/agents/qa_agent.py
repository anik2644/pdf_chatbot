"""
Main QA Agent implementation.
"""

from typing import Optional, List, Dict, Any

from pdf_qa_system.core.llm_factory import LLMFactory
from pdf_qa_system.core.embeddings_factory import EmbeddingsFactory
from pdf_qa_system.document_processing.processor import DocumentProcessor
from pdf_qa_system.vectorstore.store_factory import VectorStoreFactory
from pdf_qa_system.vectorstore.retriever import RetrieverFactory
from pdf_qa_system.chains.qa_chain import QAChainFactory
from pdf_qa_system.agents.graph.workflow import QAWorkflow
from pdf_qa_system.agents.tools.tool_registry import ToolRegistry
from pdf_qa_system.agents.tools.pdf_search_tool import PDFSearchTool
from pdf_qa_system.utils.logger import get_logger

logger = get_logger(__name__)


class QAAgent:
    """Main QA Agent that orchestrates all components."""

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        embeddings_provider: Optional[str] = None,
        embeddings_model: Optional[str] = None,
        use_langgraph: bool = True,
        enable_grading: bool = False,
        enable_hallucination_check: bool = False
    ):
        self.use_langgraph = use_langgraph
        self.enable_grading = enable_grading
        self.enable_hallucination_check = enable_hallucination_check

        # Store provider info for later reference
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.embeddings_provider = embeddings_provider
        self.embeddings_model = embeddings_model

        # Initialize factories
        self.llm_factory = LLMFactory()
        self.embeddings_factory = EmbeddingsFactory()

        # Create LLM and embeddings
        self.llm = self.llm_factory.create(
            provider=llm_provider,
            model_name=llm_model
        )
        self.embeddings = self.embeddings_factory.create(
            provider=embeddings_provider,
            model_name=embeddings_model
        )

        # Initialize tool registry
        self.tool_registry = ToolRegistry()

        # These will be set when documents are loaded
        self.vector_store = None
        self.retriever = None
        self.workflow = None
        self.chain = None
        self._is_initialized = False
        self._loaded_files: List[str] = []

        logger.info("QA Agent initialized")

    def load_documents(
        self,
        file_path: Optional[str] = None,
        directory_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        collection_name: str = "default",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> None:
        """Load and process documents."""
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if file_path:
            chunks = processor.process_file(file_path)
            self._loaded_files = [file_path]
        elif directory_path:
            chunks = processor.process_directory(directory_path)
            self._loaded_files = [directory_path]
        elif file_paths:
            chunks = processor.process_multiple_files(file_paths)
            self._loaded_files = file_paths
        else:
            raise ValueError("Provide file_path, directory_path, or file_paths")

        if not chunks:
            raise ValueError("No document chunks created")

        logger.info(f"Processed {len(chunks)} document chunks")

        # Create vector store
        store_factory = VectorStoreFactory(embeddings=self.embeddings)
        self.vector_store = store_factory.create_from_documents(
            documents=chunks,
            collection_name=collection_name
        )

        # Create retriever
        retriever_factory = RetrieverFactory(self.vector_store)
        self.retriever = retriever_factory.create_basic_retriever(k=4)

        # Register PDF search tool
        pdf_tool = PDFSearchTool(retriever=self.retriever)
        self.tool_registry.register(pdf_tool)

        # Create workflow or chain based on configuration
        self._setup_qa_pipeline()

        self._is_initialized = True
        logger.info("Documents loaded and agent ready")

    def _setup_qa_pipeline(self) -> None:
        """Set up the QA pipeline (workflow or chain)."""
        if self.use_langgraph:
            self.workflow = QAWorkflow(
                llm=self.llm,
                retriever=self.retriever,
                enable_grading=self.enable_grading,
                enable_hallucination_check=self.enable_hallucination_check
            )
            logger.info("LangGraph workflow created")
        else:
            chain_factory = QAChainFactory(llm=self.llm, retriever=self.retriever)
            self.chain = chain_factory.create_chain_with_sources()
            logger.info("QA chain created")

    def add_documents(
        self,
        file_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> None:
        """Add more documents to an existing vector store."""
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized. Use load_documents first.")

        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if file_path:
            chunks = processor.process_file(file_path)
            self._loaded_files.append(file_path)
        elif file_paths:
            chunks = processor.process_multiple_files(file_paths)
            self._loaded_files.extend(file_paths)
        else:
            raise ValueError("Provide file_path or file_paths")

        if not chunks:
            raise ValueError("No document chunks created")

        # Add to existing vector store
        self.vector_store.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector store")

    def load_existing_store(
        self,
        collection_name: str = "default"
    ) -> bool:
        """Load an existing vector store."""
        store_factory = VectorStoreFactory(embeddings=self.embeddings)
        self.vector_store = store_factory.load_existing(collection_name=collection_name)

        if self.vector_store is None:
            logger.warning("No existing vector store found")
            return False

        # Create retriever
        retriever_factory = RetrieverFactory(self.vector_store)
        self.retriever = retriever_factory.create_basic_retriever(k=4)

        # Register PDF search tool
        pdf_tool = PDFSearchTool(retriever=self.retriever)
        self.tool_registry.register(pdf_tool)

        # Create workflow or chain
        self._setup_qa_pipeline()

        self._is_initialized = True
        logger.info("Existing vector store loaded")
        return True

    def set_retriever_config(
        self,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None
    ) -> None:
        """Configure the retriever settings."""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        retriever_factory = RetrieverFactory(self.vector_store)

        if search_type == "mmr" and fetch_k is not None:
            self.retriever = retriever_factory.create_mmr_retriever(
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult or 0.5
            )
        elif search_type == "threshold" and score_threshold is not None:
            self.retriever = retriever_factory.create_threshold_retriever(
                k=k,
                score_threshold=score_threshold
            )
        else:
            self.retriever = retriever_factory.create_basic_retriever(
                search_type=search_type,
                k=k
            )

        # Update the workflow/chain with new retriever
        self._setup_qa_pipeline()

        # Update PDF search tool
        pdf_tool = PDFSearchTool(retriever=self.retriever, k=k)
        self.tool_registry.register(pdf_tool)

        logger.info(f"Retriever configured: type={search_type}, k={k}")

    def query(self, question: str) -> Dict[str, Any]:
        """Query the agent with a question."""
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized. Load documents first.")

        logger.info(f"Processing query: {question[:50]}...")

        if self.use_langgraph and self.workflow:
            result = self.workflow.run(question)
        elif self.chain:
            chain_result = self.chain.invoke(question)
            result = {
                "answer": chain_result.get("answer", ""),
                "sources": chain_result.get("sources", []),
                "error": None
            }
        else:
            raise RuntimeError("No workflow or chain available")

        return result

    async def aquery(self, question: str) -> Dict[str, Any]:
        """Async query the agent with a question."""
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized. Load documents first.")

        logger.info(f"Processing async query: {question[:50]}...")

        if self.use_langgraph and self.workflow:
            result = await self.workflow.arun(question)
        elif self.chain:
            chain_result = await self.chain.ainvoke(question)
            result = {
                "answer": chain_result.get("answer", ""),
                "sources": chain_result.get("sources", []),
                "error": None
            }
        else:
            raise RuntimeError("No workflow or chain available")

        return result

    def query_with_history(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Query with conversation history context."""
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized. Load documents first.")

        # Build context from history
        history_context = ""
        if chat_history:
            history_parts = []
            for msg in chat_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_parts.append(f"{role.capitalize()}: {content}")
            history_context = "\n".join(history_parts)

        # Enhance question with history context
        if history_context:
            enhanced_question = f"Given this conversation history:\n{history_context}\n\nAnswer this question: {question}"
        else:
            enhanced_question = question

        return self.query(enhanced_question)

    def get_relevant_documents(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Get relevant documents without generating an answer."""
        if not self.retriever:
            raise RuntimeError("Retriever not initialized")

        docs = self.retriever.invoke(query)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            }
            for doc in docs[:k]
        ]

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.tool_registry.list_tools()

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools."""
        return self.tool_registry.get_tool_descriptions()

    def use_tool(self, tool_name: str, query: str) -> str:
        """Use a specific tool directly."""
        tool = self.tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        return tool._run(query)

    def get_loaded_files(self) -> List[str]:
        """Get list of loaded files."""
        return self._loaded_files.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            "is_initialized": self._is_initialized,
            "loaded_files": len(self._loaded_files),
            "files": self._loaded_files,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embeddings_provider": self.embeddings_provider,
            "use_langgraph": self.use_langgraph,
            "available_tools": self.get_available_tools(),
        }

        if self.vector_store:
            try:
                stats["vector_store_type"] = type(self.vector_store).__name__
            except Exception:
                pass

        return stats

    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.vector_store = None
        self.retriever = None
        self.workflow = None
        self.chain = None
        self._is_initialized = False
        self._loaded_files = []
        self.tool_registry = ToolRegistry()

        logger.info("Agent reset to initial state")

    def switch_llm(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7
    ) -> None:
        """Switch to a different LLM."""
        self.llm_provider = provider or self.llm_provider
        self.llm_model = model_name or self.llm_model

        self.llm = self.llm_factory.create(
            provider=self.llm_provider,
            model_name=self.llm_model,
            temperature=temperature
        )

        # Rebuild the pipeline if initialized
        if self._is_initialized:
            self._setup_qa_pipeline()

        logger.info(f"Switched LLM to: {self.llm_provider}/{self.llm_model}")

    @property
    def is_ready(self) -> bool:
        """Check if agent is ready to answer questions."""
        return self._is_initialized

    def __repr__(self) -> str:
        """String representation of the agent."""
        status = "ready" if self._is_initialized else "not initialized"
        return f"QAAgent(status={status}, provider={self.llm_provider}, files={len(self._loaded_files)})"