"""
Tool registry for managing available tools.
"""

from typing import Dict, List, Optional, Type
from langchain_core.tools import BaseTool as LangChainBaseTool

from core.agents.tools.base_tool import BaseTool
from core.utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for managing and retrieving tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def get_all(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_langchain_tools(self) -> List[LangChainBaseTool]:
        """Get all tools as LangChain tools."""
        return [tool.to_langchain_tool() for tool in self._tools.values()]

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all tools."""
        return {name: tool.description for name, tool in self._tools.items()}