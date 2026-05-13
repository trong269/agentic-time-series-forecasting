"""Shared workflow state types."""

from __future__ import annotations

from typing import Any, TypedDict


class ForecastingWorkflowState(TypedDict, total=False):
    ticker: str
    run_id: str
    run_date: str
    horizon: int
    fetch_latest: bool
    is_retrain: bool
    retrain_count: int              # number of retrain cycles this run
    raw_data: Any
    preprocessed_data: dict[str, Any]
    model_path: str
    model_version: int
    forecasting_output: dict[str, Any] | None
    evaluation_output: dict[str, Any] | None
    reporter_output: dict[str, Any] | None
    improvement_output: dict[str, Any] | None
    previous_reports: list[dict[str, Any]]
    _cached_news_context: dict[str, Any] | None  # cached to avoid re-fetching on retrain
    errors: list[dict[str, Any]]
