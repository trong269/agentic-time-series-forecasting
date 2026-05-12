from typing import Dict, Any, List

from .forecasting_agent import ForecastingAgent
from ..utils.logger import get_logger
from ..llm.factory import get_llm

logger = get_logger(__name__)


class AgentFactory:

    _agents: Dict[str, Any] = {}

    @classmethod
    def _build_llm(cls):
        """
        Build LLM instance based on configured provider

        Returns:
            LLM instance configured with current provider settings
        """
        try:
            llm = get_llm()
            logger.info(f"Built LLM instance: {llm.__class__.__name__}")
            return llm
        except Exception as e:
            logger.error(f"Failed to build LLM instance: {e}")
            raise

    @classmethod
    def build_all_agents(cls):
        """
        Build all agents at startup and store in memory
        """
        cls._build_forecasting_agent()

    @classmethod
    def _build_forecasting_agent(cls):
        """Build the forecasting agent for generating time series forecasts"""
        try:
            llm = cls._build_llm()
            agent = ForecastingAgent(
                llm=llm,
                agent_name='forecasting_agent',
            )
            cls._agents["forecasting_agent"] = agent
            logger.info("Built and stored Forecasting Agent")
        except Exception as e:
            logger.error(f"Failed to build Forecasting agent: {e}")
            raise

    @classmethod
    def get_agent(cls, agent_name: str) -> Any:
        """
        Get a pre-built agent from memory
        """
        if agent_name not in cls._agents:
            raise ValueError(
                f"Agent type '{agent_name}' not found. Available: {list(cls._agents.keys())}")

        return cls._agents[agent_name]

    @classmethod
    def list_agents(cls) -> Dict[str, str]:
        """List all available pre-built agents"""
        return {
            agent_name: agent.__class__.__name__
            for agent_name, agent in cls._agents.items()
        }

    @classmethod
    def is_ready(cls) -> bool:
        """Check if all agents are built and ready"""
        return len(cls._agents) > 0