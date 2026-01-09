"""
LangGraph node implementations.
"""

from typing import Any, Dict
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pdf_qa_system.agents.graph.state import GraphState
from pdf_qa_system.utils.logger import get_logger

logger = get_logger(__name__)


class GraphNodes:
    """Collection of nodes for the QA workflow graph."""

    def __init__(
            self,
            llm: BaseLanguageModel,
            retriever: BaseRetriever
    ):
        self.llm = llm
        self.retriever = retriever

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Retrieve relevant documents."""
        logger.info(f"Retrieving documents for: {state['question'][:50]}...")

        try:
            documents = self.retriever.invoke(state["question"])

            sources = []
            for doc in documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                sources.append(f"{source} (page {page})")

            context = "\n\n".join(doc.page_content for doc in documents)

            return {
                "documents": documents,
                "context": context,
                "sources": list(set(sources)),
                "error": None
            }
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {
                "documents": [],
                "context": "",
                "sources": [],
                "error": str(e)
            }

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Grade retrieved documents for relevance."""
        logger.info("Grading document relevance...")

        if not state["documents"]:
            return {"needs_more_context": True}

        # Simple relevance check - could be made more sophisticated
        grade_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question.

Document: {document}

Question: {question}

Does the document contain information relevant to answering the question?
Answer only 'yes' or 'no'."""
        )

        chain = grade_prompt | self.llm | StrOutputParser()

        relevant_docs = []
        for doc in state["documents"]:
            try:
                result = chain.invoke({
                    "document": doc.page_content[:1000],
                    "question": state["question"]
                })
                if "yes" in result.lower():
                    relevant_docs.append(doc)
            except Exception as e:
                logger.warning(f"Error grading document: {e}")
                relevant_docs.append(doc)  # Include if grading fails

        needs_more = len(relevant_docs) < 2

        return {
            "documents": relevant_docs,
            "context": "\n\n".join(doc.page_content for doc in relevant_docs),
            "needs_more_context": needs_more
        }

    def generate(self, state: GraphState) -> Dict[str, Any]:
        """Generate answer from context."""
        logger.info("Generating answer...")

        if state.get("error"):
            return {"answer": f"Error occurred: {state['error']}"}

        if not state["context"]:
            return {
                "answer": "I couldn't find relevant information to answer your question."
            }

        generate_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions based on provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the answer isn't in the context, say so clearly
- Be concise but thorough
- Reference specific parts of the context when relevant

Answer:"""
        )

        chain = generate_prompt | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({
                "context": state["context"],
                "question": state["question"]
            })
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"answer": f"Error generating answer: {str(e)}"}

    def check_hallucination(self, state: GraphState) -> Dict[str, Any]:
        """Check if the answer is grounded in the context."""
        logger.info("Checking answer grounding...")

        check_prompt = ChatPromptTemplate.from_template(
            """You are a grader checking if an answer is grounded in the provided context.

Context: {context}

Answer: {answer}

Is the answer fully supported by the context? Answer 'yes' or 'no' with a brief explanation."""
        )

        chain = check_prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "context": state["context"],
                "answer": state["answer"]
            })

            if "no" in result.lower()[:10]:
                return {
                    "answer": state[
                                  "answer"] + "\n\n[Note: This answer may not be fully supported by the source documents.]"
                }
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")

        return {}