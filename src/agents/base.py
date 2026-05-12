from abc import ABC, abstractmethod
from typing import Dict, Any, List

from langgraph.graph import StateGraph

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class providing reusable functions for all agents"""

    def __init__(
            self,
            name: str,
            llm: Any = None,
        ):
            """
            Initialize base agent with configurable parameters

            Parameters
            ----------
            name : str
                The name of the agent
            llm : Any
                The language model instance (if needed by the agent)
            """
            self.name = name
            self.llm = llm

            # Create dedicated logger for this agent
            logger.info(f"Initialized agent: {self.name}")
            self.graph = None
            self.compiled_graph = None

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the state graph for this agent

        Returns
        -------
        StateGraph
            The constructed state graph
        """
        raise NotImplementedError(
            "build_graph must be implemented by subclasses")

    def compile(self):
        """Compile the graph for execution"""
        if self.graph is None:
            self.graph = self.build_graph()
        self.compiled_graph = self.graph.compile()
        logger.info(f"Compiled graph for agent: {self.name}")

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous invoke - executes the compiled graph

        Parameters
        ----------
        input_data : Dict[str, Any]
            Input data for the agent

        Returns
        -------
        Dict[str, Any]
            Agent response
        """
        if self.compiled_graph is None:
            self.compile()

        try:

            result = self.compiled_graph.invoke(input_data)
            return result
        except Exception as e:
            logger.error(f"Error during invoke: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous invoke - executes the compiled graph asynchronously

        Parameters
        ----------
        input_data : Dict[str, Any]
            Input data for the agent

        Returns
        -------
        Dict[str, Any]
            Agent response
        """
        if self.compiled_graph is None:
            self.compile()

        try:
            result = await self.compiled_graph.ainvoke(input_data)
            return result
        except Exception as e:
            logger.error(f"Error during ainvoke: {e}")
            return {
                "success": False,
                "error": str(e)
            }
