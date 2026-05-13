"""Base workflow abstraction mirroring the agent graph pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langgraph.graph import StateGraph

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseWorkflow(ABC):
    """Base class for workflows backed by a LangGraph state graph."""

    def __init__(self, name: str):
        self.name = name
        self.graph = None
        self.compiled_graph = None
        logger.info(f"Initialized workflow: {self.name}")

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build and return the workflow state graph."""
        raise NotImplementedError("build_graph must be implemented by subclasses")

    def compile(self):
        """Compile the workflow graph for execution."""
        if self.graph is None:
            self.graph = self.build_graph()
        self.compiled_graph = self.graph.compile()
        logger.info(f"Compiled graph for workflow: {self.name}")

    def invoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Synchronously execute the compiled workflow graph."""
        if self.compiled_graph is None:
            self.compile()
        try:
            return self.compiled_graph.invoke(input_data)
        except Exception as exc:
            logger.error(f"Error during workflow invoke: {exc}")
            return {"success": False, "error": str(exc), **input_data}

    async def ainvoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Asynchronously execute the compiled workflow graph."""
        if self.compiled_graph is None:
            self.compile()
        try:
            return await self.compiled_graph.ainvoke(input_data)
        except Exception as exc:
            logger.error(f"Error during workflow ainvoke: {exc}")
            return {"success": False, "error": str(exc), **input_data}
