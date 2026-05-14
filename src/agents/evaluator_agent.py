"""Evaluator agent graph definition."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .base import BaseAgent
from .components.nodes import (
    build_news_context,
    calculate_risk_breakdown,
    gather_news_node,
    calculate_technical_risk_node,
    compute_trust_score_node,
    heuristic_enrich_article,
    normalize_enriched_article,
    recent_volatility_risk,
    summarize_decision,
    trend_alignment_risk,
)
from .components.states import EvaluatorAgentState


class EvaluatorAgent(BaseAgent):
    """Define the evaluator agent graph."""

    def __init__(self, name: str = "evaluator_agent", llm: Any = None, **kwargs: Any):
        super().__init__(name=kwargs.get("agent_name", name), llm=llm)

    def build_graph(self) -> StateGraph:
        graph = StateGraph(EvaluatorAgentState)
        graph.add_node("gather_news", gather_news_node)
        graph.add_node("calculate_technical_risk", calculate_technical_risk_node)
        graph.add_node("compute_trust_score", compute_trust_score_node)

        graph.add_edge(START, "gather_news")
        graph.add_edge("gather_news", "calculate_technical_risk")
        graph.add_edge("calculate_technical_risk", "compute_trust_score")
        graph.add_edge("compute_trust_score", END)
        return graph

__all__ = [
    "EvaluatorAgent",
    "build_news_context",
    "calculate_risk_breakdown",
    "heuristic_enrich_article",
    "normalize_enriched_article",
    "recent_volatility_risk",
    "summarize_decision",
    "trend_alignment_risk",
]
