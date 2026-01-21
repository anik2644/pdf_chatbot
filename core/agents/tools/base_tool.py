"""
Base tool class for agent tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from langchain_core.tools import BaseTool as LangChainBaseTool


class BaseTool(ABC):
    """Abstract base class for custom tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        pass

    @abstractmethod
    def _run(self, query: str, **kwargs) -> str:
        """Execute the tool."""
        pass

    def to_langchain_tool(self) -> LangChainBaseTool:
        """Convert to a LangChain tool."""
        from langchain_core.tools import Tool

        return Tool(
            name=self.name,
            description=self.description,
            func=self._run
        )