"""Tavily news search wrapper with a deterministic no-key fallback."""

from __future__ import annotations

import os
from typing import Any


def search_tavily_news(
    query: str,
    max_results: int = 8,
    search_depth: str = "advanced",
) -> list[dict[str, Any]]:
    """Search Tavily and return normalized result dictionaries.

    The pipeline can still run without Tavily credentials; in that case this
    function returns an empty list and the evaluator treats news risk as neutral.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        return []

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        include_answer=True,
    )
    results = response.get("results", []) if isinstance(response, dict) else []
    articles: list[dict[str, Any]] = []
    for result in results[:max_results]:
        articles.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "published_date": result.get("published_date"),
            "summary": result.get("content") or result.get("snippet") or "",
            "raw": result,
        })
    return articles
