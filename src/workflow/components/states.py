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
    retrain_count: int
    raw_data: Any
    model_path: str
    model_version: int
    original_model_path: str
    original_model_version: int
    forecasting_output: dict[str, Any] | None
    evaluation_output: dict[str, Any] | None
    reporter_output: dict[str, Any] | None
    original_forecasting_output: dict[str, Any] | None
    original_evaluation_output: dict[str, Any] | None
    original_reporter_output: dict[str, Any] | None
    improvement_output: dict[str, Any] | None
    forced_action: str | None
    forced_reason: str | None
    previous_reports: list[dict[str, Any]]
    _cached_news_context: dict[str, Any] | None
    errors: list[dict[str, Any]]
