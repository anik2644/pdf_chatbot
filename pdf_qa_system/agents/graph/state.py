"""
LangGraph state definitions.
"""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.documents import Document
import operator


class GraphState(TypedDict):
    """State for the QA graph workflow."""

    # Input
    question: str

    # Retrieved documents
    documents: List[Document]

    # Processing
    context: str

    # Output
    answer: str
    sources: List[str]

    # Metadata
    iterations: int
    needs_more_context: bool
    error: Optional[str]


class AgentState(TypedDict):
    """State for agent-based workflows."""

    # Messages history
    messages: Annotated[List, operator.add]

    # Current query
    query: str

    # Retrieved context
    context: str
    documents: List[Document]

    # Final output
    output: str

    # Control flow
    next_action: str
    iterations: int