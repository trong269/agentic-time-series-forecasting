"""Improvement agent graph definition."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .base import BaseAgent
from .components.nodes import (
    DEFAULT_CANDIDATES,
    plan_candidates_node,
    retrain_candidates_node,
    select_best_candidate_node,
)
from .components.states import ImprovementAgentState


class ImprovementAgent(BaseAgent):
    """Define the improvement agent graph."""

    def __init__(self, name: str = "improvement_agent", llm: Any = None, **kwargs: Any):
        super().__init__(name=kwargs.get("agent_name", name), llm=llm)

    def build_graph(self) -> StateGraph:
        graph = StateGraph(ImprovementAgentState)
        graph.add_node("plan_candidates", plan_candidates_node)
        graph.add_node("retrain_candidates", retrain_candidates_node)
        graph.add_node("select_best", select_best_candidate_node)

        graph.add_edge(START, "plan_candidates")
        graph.add_edge("plan_candidates", "retrain_candidates")
        graph.add_edge("retrain_candidates", "select_best")
        graph.add_edge("select_best", END)
        return graph

__all__ = ["ImprovementAgent", "DEFAULT_CANDIDATES"]
