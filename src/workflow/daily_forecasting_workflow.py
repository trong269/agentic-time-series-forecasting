"""Daily forecasting workflow graph definition."""

from __future__ import annotations

import uuid
from datetime import date
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.factory import AgentFactory
from src.utils.config_manager import config_manager

from .base import BaseWorkflow
from .components.nodes import (
    evaluation_workflow_node,
    forecasting_workflow_node,
    improvement_workflow_node,
    load_inputs_node,
    promote_retrained_model_node,
    reject_retrained_model_node,
    reporting_workflow_node,
    resolve_latest_usable_model,
    should_retrain,
    should_rerun_after_improvement,
)
from .components.states import ForecastingWorkflowState


class DailyForecastingWorkflow(BaseWorkflow):
    """Define the daily forecasting workflow graph."""

    def __init__(
        self,
        preprocessing_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
        llm: Any = None,
        name: str = "daily_forecasting_workflow",
    ):
        super().__init__(name=name)
        self.preprocessing_config = preprocessing_config or config_manager.preprocessing
        self.model_config = model_config or config_manager.model
        self.agent_config = agent_config or config_manager.agent
        AgentFactory.build_all_agents()
        self.forecasting_agent = AgentFactory.get_agent("forecasting_agent")
        self.evaluator_agent = AgentFactory.get_agent("evaluator_agent")
        self.reporter_agent = AgentFactory.get_agent("reporter_agent")
        self.improvement_agent = AgentFactory.get_agent("improvement_agent")

    def build_graph(self) -> StateGraph:
        graph = StateGraph(ForecastingWorkflowState)
        graph.add_node(
            "load_inputs",
            partial(
                load_inputs_node,
                preprocessing_config=self.preprocessing_config,
                model_config=self.model_config,
                agent_config=self.agent_config,
            ),
        )
        graph.add_node(
            "forecasting",
            partial(
                forecasting_workflow_node,
                forecasting_agent=self.forecasting_agent,
                preprocessing_config=self.preprocessing_config,
                model_config=self.model_config,
            ),
        )
        graph.add_node(
            "evaluation",
            partial(
                evaluation_workflow_node,
                evaluator_agent=self.evaluator_agent,
                agent_config=self.agent_config,
            ),
        )
        graph.add_node(
            "reporting",
            partial(
                reporting_workflow_node,
                reporter_agent=self.reporter_agent,
                agent_config=self.agent_config,
            ),
        )
        graph.add_node(
            "improvement",
            partial(
                improvement_workflow_node,
                improvement_agent=self.improvement_agent,
                preprocessing_config=self.preprocessing_config,
                model_config=self.model_config,
                agent_config=self.agent_config,
            ),
        )
        graph.add_node("promote_retrained_model", promote_retrained_model_node)
        graph.add_node(
            "reject_retrained_model",
            partial(
                reject_retrained_model_node,
                reporter_agent=self.reporter_agent,
                agent_config=self.agent_config,
            ),
        )

        graph.add_edge(START, "load_inputs")
        graph.add_edge("load_inputs", "forecasting")
        graph.add_edge("forecasting", "evaluation")
        graph.add_edge("evaluation", "reporting")
        graph.add_conditional_edges(
            "reporting",
            should_retrain,
            {"improve": "improvement", "end": END},
        )
        graph.add_conditional_edges(
            "improvement",
            should_rerun_after_improvement,
            {"rerun": "promote_retrained_model", "end": "reject_retrained_model"},
        )
        graph.add_edge("promote_retrained_model", "forecasting")
        graph.add_edge("reject_retrained_model", END)
        return graph

    def run(self, ticker: str, horizon: int = 7, fetch_latest: bool = False) -> ForecastingWorkflowState:
        state: ForecastingWorkflowState = {
            "ticker": ticker,
            "run_id": uuid.uuid4().hex,
            "run_date": date.today().isoformat(),
            "horizon": horizon,
            "fetch_latest": fetch_latest,
            "is_retrain": False,
            "retrain_count": 0,
            "errors": [],
        }
        result = self.invoke(state)
        if result.get("success") is False:
            raise RuntimeError(result.get("error", "Workflow failed"))
        return result


__all__ = ["DailyForecastingWorkflow", "resolve_latest_usable_model"]
