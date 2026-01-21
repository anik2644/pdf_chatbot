"""
QA Chain implementations.
"""

from typing import Optional, Any, Dict
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from core.utils.logger import get_logger

logger = get_logger(__name__)


class QAChainFactory:
    """Factory for creating QA chains."""

    DEFAULT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- If the answer is not in the context, say "I cannot find this information in the provided documents"
- Be concise but comprehensive
- If relevant, cite which part of the context supports your answer

Answer:"""

    def __init__(
            self,
            llm: BaseLanguageModel,
            retriever: BaseRetriever
    ):
        self.llm = llm
        self.retriever = retriever

    def create_basic_chain(
            self,
            prompt_template: Optional[str] = None
    ):
        """Create a basic RAG chain."""
        template = prompt_template or self.DEFAULT_TEMPLATE
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
                RunnableParallel(
                    context=self.retriever | format_docs,
                    question=RunnablePassthrough()
                )
                | prompt
                | self.llm
                | StrOutputParser()
        )

        logger.info("Created basic QA chain")
        return chain

    def create_chain_with_sources(
            self,
            prompt_template: Optional[str] = None
    ):
        """Create a RAG chain that returns sources."""
        template = prompt_template or self.DEFAULT_TEMPLATE
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def get_sources(docs):
            sources = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                sources.append(f"{source} (page {page})")
            return list(set(sources))

        chain = RunnableParallel(
            answer=(
                    RunnableParallel(
                        context=self.retriever | format_docs,
                        question=RunnablePassthrough()
                    )
                    | prompt
                    | self.llm
                    | StrOutputParser()
            ),
            sources=self.retriever | get_sources
        )

        logger.info("Created QA chain with sources")
        return chain