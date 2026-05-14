from typing import Any

from .evaluator_agent import EvaluatorAgent
from .forecasting_agent import ForecastingAgent
from .improvement_agent import ImprovementAgent
from .reporter_agent import ReporterAgent
from ..utils.logger import get_logger
from ..llm.factory import get_llm

logger = get_logger(__name__)


class AgentFactory:

    _agents: dict[str, Any] = {}
    _builders: dict[str, str] = {
        "forecasting_agent": "_build_forecasting_agent",
        "evaluator_agent": "_build_evaluator_agent",
        "reporter_agent": "_build_reporter_agent",
        "improvement_agent": "_build_improvement_agent",
    }

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
        for agent_name in cls._builders:
            cls.get_agent(agent_name)

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
    def _build_evaluator_agent(cls):
        try:
            agent = EvaluatorAgent(llm=cls._build_llm(), agent_name="evaluator_agent")
            cls._agents["evaluator_agent"] = agent
            logger.info("Built and stored Evaluator Agent")
        except Exception as e:
            logger.error(f"Failed to build Evaluator agent: {e}")
            raise

    @classmethod
    def _build_reporter_agent(cls):
        try:
            agent = ReporterAgent(llm=cls._build_llm(), agent_name="reporter_agent")
            cls._agents["reporter_agent"] = agent
            logger.info("Built and stored Reporter Agent")
        except Exception as e:
            logger.error(f"Failed to build Reporter agent: {e}")
            raise

    @classmethod
    def _build_improvement_agent(cls):
        try:
            agent = ImprovementAgent(llm=cls._build_llm(), agent_name="improvement_agent")
            cls._agents["improvement_agent"] = agent
            logger.info("Built and stored Improvement Agent")
        except Exception as e:
            logger.error(f"Failed to build Improvement agent: {e}")
            raise

    @classmethod
    def get_agent(cls, agent_name: str) -> Any:
        """
        Get a pre-built agent from memory
        """
        if agent_name not in cls._agents:
            builder_name = cls._builders.get(agent_name)
            if builder_name is None:
                raise ValueError(
                    f"Agent type '{agent_name}' not found. Available: {list(cls._builders.keys())}")
            getattr(cls, builder_name)()

        return cls._agents[agent_name]

    @classmethod
    def list_agents(cls) -> dict[str, str]:
        """List all available pre-built agents"""
        return {
            agent_name: agent.__class__.__name__
            for agent_name, agent in cls._agents.items()
        }

    @classmethod
    def is_ready(cls) -> bool:
        """Check if all agents are built and ready"""
        return all(agent_name in cls._agents for agent_name in cls._builders)
