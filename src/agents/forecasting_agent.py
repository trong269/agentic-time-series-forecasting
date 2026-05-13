"""Forecasting agent graph definition."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .base import BaseAgent
from .components.nodes import load_model_node, predict_node, evaluate_holdout_node
from .components.states import ForecastingAgentState


class ForecastingAgent(BaseAgent):
    """Define the forecasting agent graph."""

    def __init__(self, name: str = "forecasting_agent", llm: Any = None, **kwargs: Any):
        super().__init__(name=kwargs.get("agent_name", name), llm=llm)

    def build_graph(self) -> StateGraph:
        graph = StateGraph(ForecastingAgentState)
        graph.add_node("load_model", load_model_node)
        graph.add_node("predict", predict_node)
        graph.add_node("evaluate_holdout", evaluate_holdout_node)
        
        graph.add_edge(START, "load_model")
        graph.add_edge("load_model", "predict")
        graph.add_edge("predict", "evaluate_holdout")
        graph.add_edge("evaluate_holdout", END)
        return graph
