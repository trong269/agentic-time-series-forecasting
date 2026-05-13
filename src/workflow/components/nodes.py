"""Workflow node implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Literal

from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.improvement_agent import ImprovementAgent
from src.agents.reporter_agent import ReporterAgent
from src.agents.components.nodes import load_previous_reports
from src.forecasting.trainer import train_xgboost_forecaster
from src.ingestion import fetch_stock_data, get_stock_data
from src.preprocessing import preprocess_data
from src.utils.config_manager import config_manager
from src.utils.logger import get_logger

from .states import ForecastingWorkflowState

logger = get_logger(__name__)


# =============================================================================
# Daily Forecasting Workflow Nodes
# =============================================================================


def load_inputs_node(
    state: ForecastingWorkflowState,
    preprocessing_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    agent_config: dict[str, Any] | None = None,
) -> ForecastingWorkflowState:
    preprocessing_cfg = preprocessing_config or config_manager.preprocessing
    model_cfg = model_config or config_manager.model
    agent_cfg = agent_config or config_manager.agent
    ticker = state["ticker"]

    if state.get("fetch_latest", False):
        result = fetch_stock_data(ticker)
        if result.get("status") == "error":
            logger.warning("Latest data fetch failed: %s", result.get("error"))

    raw_data = get_stock_data(ticker)
    if raw_data.empty:
        raise ValueError(f"No local stock data found for {ticker}")
    raw_data = raw_data.sort_values("date").reset_index(drop=True)
    preprocessed_data = preprocess_data(raw_data, preprocessing_cfg)
    model_path, model_version = resolve_latest_usable_model(ticker, model_cfg)
    if model_path is None:
        logger.info("No usable model found for %s; training initial model", ticker)
        training_result = train_xgboost_forecaster(
            ticker=ticker,
            df=raw_data,
            config={**preprocessing_cfg, "model": model_cfg},
        )
        model_path = str(training_result["version_dir"])
        model_version = int(training_result["version"])
        mark_model_status(Path(model_path), "accepted", reason="Initial model trained by workflow.")

    report_cfg = agent_cfg.get("reporter", {})
    previous_reports = load_previous_reports(
        report_cfg.get("reports_dir", "artifacts/reports"),
        ticker,
        int(report_cfg.get("history_n", 7)),
    )
    state.update({
        "raw_data": raw_data,
        "preprocessed_data": preprocessed_data,
        "model_path": str(model_path),
        "model_version": int(model_version),
        "previous_reports": previous_reports,
    })
    return state


def forecasting_workflow_node(
    state: ForecastingWorkflowState,
    forecasting_agent: ForecastingAgent,
    preprocessing_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> ForecastingWorkflowState:
    agent_state = forecasting_agent.invoke({
        "ticker": state["ticker"],
        "df_raw": state["raw_data"],
        "preprocessed_data": state["preprocessed_data"],
        "model_path": state["model_path"],
        "horizon": state.get("horizon", 7),
        "preprocessing_config": preprocessing_config or config_manager.preprocessing,
        "model_config": model_config or config_manager.model,
    })
    if agent_state.get("success") is False:
        raise RuntimeError(f"ForecastingAgent failed: {agent_state.get('error')}")
    state["forecasting_output"] = agent_state["forecasting_output"]
    return state


def evaluation_workflow_node(
    state: ForecastingWorkflowState,
    evaluator_agent: EvaluatorAgent,
    agent_config: dict[str, Any] | None = None,
) -> ForecastingWorkflowState:
    agent_state = evaluator_agent.invoke({
        "ticker": state["ticker"],
        "forecasting_output": state["forecasting_output"],
        "df_recent": state["raw_data"].tail(30),
        "agent_config": agent_config or config_manager.agent,
        "llm": evaluator_agent.llm,
        # Pass cached news_context so re-evaluation after retrain skips Tavily
        "news_context": state.get("_cached_news_context"),
    })
    if agent_state.get("success") is False:
        raise RuntimeError(f"EvaluatorAgent failed: {agent_state.get('error')}")
    state["evaluation_output"] = agent_state["evaluation_output"]
    # Cache news_context for potential retrain re-evaluation (Fix #6)
    state["_cached_news_context"] = agent_state.get("news_context") or agent_state["evaluation_output"].get("news_context")
    return state


def reporting_workflow_node(
    state: ForecastingWorkflowState,
    reporter_agent: ReporterAgent,
    agent_config: dict[str, Any] | None = None,
) -> ForecastingWorkflowState:
    agent_state = reporter_agent.invoke({
        "ticker": state["ticker"],
        "run_id": state["run_id"],
        "run_date": state["run_date"],
        "is_retrain": state["is_retrain"],
        "forecasting_output": state["forecasting_output"],
        "evaluation_output": state["evaluation_output"],
        "improvement_output": state.get("improvement_output"),
        "previous_reports": state.get("previous_reports", []),
        "agent_config": agent_config or config_manager.agent,
        "llm": reporter_agent.llm,
    })
    if agent_state.get("success") is False:
        raise RuntimeError(f"ReporterAgent failed: {agent_state.get('error')}")
    state["reporter_output"] = agent_state["reporter_output"]
    return state


def improvement_workflow_node(
    state: ForecastingWorkflowState,
    improvement_agent: ImprovementAgent,
    preprocessing_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    agent_config: dict[str, Any] | None = None,
) -> ForecastingWorkflowState:
    agent_state = improvement_agent.invoke({
        "ticker": state["ticker"],
        "df_raw": state["raw_data"],
        "model_path": state["model_path"],
        "model_version": state["model_version"],
        "forecasting_output": state["forecasting_output"],
        "evaluation_output": state["evaluation_output"],
        "reporter_output": state["reporter_output"],
        "horizon": state.get("horizon", 7),
        "preprocessing_config": preprocessing_config or config_manager.preprocessing,
        "model_config": model_config or config_manager.model,
        "agent_config": agent_config or config_manager.agent,
        "llm": improvement_agent.llm,
    })
    if agent_state.get("success") is False:
        raise RuntimeError(f"ImprovementAgent failed: {agent_state.get('error')}")
    state["improvement_output"] = agent_state["improvement_output"]
    return state


def promote_retrained_model_node(state: ForecastingWorkflowState) -> ForecastingWorkflowState:
    improvement = state["improvement_output"]
    state["is_retrain"] = True
    state["retrain_count"] = state.get("retrain_count", 0) + 1
    state["model_version"] = improvement["new_model_version"]
    state["model_path"] = improvement["new_model_path"]
    return state


def reject_retrained_model_node(
    state: ForecastingWorkflowState,
    reporter_agent: ReporterAgent,
    agent_config: dict[str, Any] | None = None,
) -> ForecastingWorkflowState:
    improvement = state.get("improvement_output") or {}
    agent_state = reporter_agent.invoke({
        "ticker": state["ticker"],
        "run_id": state["run_id"],
        "run_date": state["run_date"],
        "is_retrain": True,
        "forecasting_output": state["forecasting_output"],
        "evaluation_output": state["evaluation_output"],
        "improvement_output": improvement,
        "previous_reports": state.get("previous_reports", []),
        "agent_config": agent_config or config_manager.agent,
        "llm": reporter_agent.llm,
        "forced_action": "reject_retrained_keep_old",
        "forced_reason": improvement.get("reason") if improvement else "Retrain failed.",
    })
    state["reporter_output"] = agent_state["reporter_output"]
    return state


# =============================================================================
# Daily Forecasting Workflow Routing
# =============================================================================


def should_retrain(state: ForecastingWorkflowState) -> Literal["improve", "end"]:
    """Route to improvement if retrain is needed and budget allows."""
    reporter = state.get("reporter_output") or {}
    max_retrain = 1  # hard cap — never retrain more than once per run
    current_count = state.get("retrain_count", 0)
    if (
        reporter.get("action") == "retrain"
        and current_count < max_retrain
    ):
        return "improve"
    return "end"


def should_rerun_after_improvement(state: ForecastingWorkflowState) -> Literal["rerun", "end"]:
    improvement = state.get("improvement_output") or {}
    if improvement.get("promoted"):
        return "rerun"
    return "end"


# =============================================================================
# Daily Forecasting Workflow Helpers
# =============================================================================


def resolve_latest_usable_model(
    ticker: str,
    model_config: dict[str, Any] | None = None,
) -> tuple[str | None, int | None]:
    model_cfg = model_config or config_manager.model
    base_dir = Path(model_cfg.get("artifacts_dir", "artifacts/models"))
    if not base_dir.exists():
        return None, None
    versions: list[tuple[int, Path]] = []
    for path in base_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("ver_"):
            continue
        try:
            version = int(path.name[4:])
        except ValueError:
            continue
        if not list(path.glob(f"{ticker}_q*.pkl")):
            continue
        metadata = _read_metadata(path)
        if metadata.get("status") == "rejected":
            continue
        versions.append((version, path))
    if not versions:
        return None, None
    version, path = max(versions, key=lambda item: item[0])
    return str(path), version


def mark_model_status(version_dir: Path, status: str, **updates: Any) -> None:
    metadata_path = version_dir / "metadata.json"
    metadata = _read_metadata(version_dir)
    metadata.update({"status": status, **updates})
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _read_metadata(version_dir: Path) -> dict[str, Any]:
    metadata_path = version_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
