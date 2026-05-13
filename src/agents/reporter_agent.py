"""Reporter agent graph definition."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .base import BaseAgent
from .components.nodes import (
    assess_history_trend,
    load_previous_reports,
    assess_trend_node,
    determine_action_node,
    generate_markdown_report_node,
    format_report_node,
)
from .components.states import ReporterAgentState


class ReporterAgent(BaseAgent):
    """Define the reporter agent graph."""

    def __init__(self, name: str = "reporter_agent", llm: Any = None, **kwargs: Any):
        super().__init__(name=kwargs.get("agent_name", name), llm=llm)

    def build_graph(self) -> StateGraph:
        graph = StateGraph(ReporterAgentState)
        graph.add_node("assess_trend", assess_trend_node)
        graph.add_node("determine_action", determine_action_node)
        graph.add_node("generate_markdown_report", generate_markdown_report_node)
        graph.add_node("format_report", format_report_node)

        graph.add_edge(START, "assess_trend")
        graph.add_edge("assess_trend", "determine_action")
        graph.add_edge("determine_action", "generate_markdown_report")
        graph.add_edge("generate_markdown_report", "format_report")
        graph.add_edge("format_report", END)
        return graph

__all__ = [
    "ReporterAgent",
    "assess_history_trend",
    "load_previous_reports",
]
