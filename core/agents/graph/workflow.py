"""
LangGraph workflow implementation.
"""

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from core.agents.graph.state import GraphState
from core.agents.graph.nodes import GraphNodes
from core.utils.logger import get_logger

logger = get_logger(__name__)


class QAWorkflow:
    """QA workflow using LangGraph."""

    def __init__(
            self,
            llm: BaseLanguageModel,
            retriever: BaseRetriever,
            enable_grading: bool = True,
            enable_hallucination_check: bool = False
    ):
        self.llm = llm
        self.retriever = retriever
        self.enable_grading = enable_grading
        self.enable_hallucination_check = enable_hallucination_check

        self.nodes = GraphNodes(llm, retriever)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the workflow graph."""
        logger.info("Building QA workflow graph...")

        # Create the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("generate", self.nodes.generate)

        if self.enable_grading:
            workflow.add_node("grade", self.nodes.grade_documents)

        if self.enable_hallucination_check:
            workflow.add_node("check_hallucination", self.nodes.check_hallucination)

        # Define edges
        workflow.set_entry_point("retrieve")

        if self.enable_grading:
            workflow.add_edge("retrieve", "grade")
            workflow.add_conditional_edges(
                "grade",
                self._should_continue_after_grade,
                {
                    "generate": "generate",
                    "end": END
                }
            )
        else:
            workflow.add_edge("retrieve", "generate")

        if self.enable_hallucination_check:
            workflow.add_edge("generate", "check_hallucination")
            workflow.add_edge("check_hallucination", END)
        else:
            workflow.add_edge("generate", END)

        return workflow.compile()

    def _should_continue_after_grade(self, state: GraphState) -> str:
        """Determine next step after grading."""
        if state.get("error"):
            return "end"
        if not state.get("documents"):
            return "end"
        return "generate"

    def run(self, question: str) -> Dict[str, Any]:
        """Run the QA workflow."""
        logger.info(f"Running QA workflow for: {question[:50]}...")

        initial_state: GraphState = {
            "question": question,
            "documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "iterations": 0,
            "needs_more_context": False,
            "error": None
        }

        try:
            result = self.graph.invoke(initial_state)
            return {
                "answer": result.get("answer", "No answer generated"),
                "sources": result.get("sources", []),
                "error": result.get("error")
            }
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {
                "answer": f"Error running workflow: {str(e)}",
                "sources": [],
                "error": str(e)
            }

    async def arun(self, question: str) -> Dict[str, Any]:
        """Run the QA workflow asynchronously."""
        logger.info(f"Running async QA workflow for: {question[:50]}...")

        initial_state: GraphState = {
            "question": question,
            "documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "iterations": 0,
            "needs_more_context": False,
            "error": None
        }

        try:
            result = await self.graph.ainvoke(initial_state)
            return {
                "answer": result.get("answer", "No answer generated"),
                "sources": result.get("sources", []),
                "error": result.get("error")
            }
        except Exception as e:
            logger.error(f"Async workflow error: {e}")
            return {
                "answer": f"Error running workflow: {str(e)}",
                "sources": [],
                "error": str(e)
            }