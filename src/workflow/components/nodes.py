"""Workflow node implementations."""

import json
import shutil
from pathlib import Path
from typing import Any
from typing import Literal

from langchain_core.runnables import RunnableConfig

from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.improvement_agent import ImprovementAgent
from src.agents.reporter_agent import ReporterAgent
from src.agents.components.nodes import load_previous_reports
from src.forecasting.trainer import train_xgboost_forecaster
from src.forecasting.trainer import get_next_model_version
from src.ingestion import fetch_stock_data, get_stock_data
from src.utils.config_manager import config_manager
from src.utils.logger import get_logger
from src.utils.scoring import composite_score

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
    agent_config: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
) -> ForecastingWorkflowState:
    agent_state = forecasting_agent.invoke({
        "ticker": state["ticker"],
        "df_raw": state["raw_data"],
        "model_path": state["model_path"],
        "horizon": state.get("horizon", 7),
        "preprocessing_config": preprocessing_config or config_manager.preprocessing,
        "model_config": model_config or config_manager.model,
        "agent_config": agent_config or config_manager.agent,
    }, config=config)
    if agent_state.get("success") is False:
        raise RuntimeError(f"ForecastingAgent failed: {agent_state.get('error')}")
    state["forecasting_output"] = agent_state["forecasting_output"]
    return state


def evaluation_workflow_node(
    state: ForecastingWorkflowState,
    evaluator_agent: EvaluatorAgent,
    agent_config: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
) -> ForecastingWorkflowState:
    agent_state = evaluator_agent.invoke({
        "ticker": state["ticker"],
        "forecasting_output": state["forecasting_output"],
        "df_recent": state["raw_data"].tail(30),
        "agent_config": agent_config or config_manager.agent,
        "llm": evaluator_agent.llm,
        # Pass cached news_context so re-evaluation after retrain skips Tavily
        "news_context": state.get("_cached_news_context"),
    }, config=config)
    if agent_state.get("success") is False:
        raise RuntimeError(f"EvaluatorAgent failed: {agent_state.get('error')}")
    state["evaluation_output"] = agent_state["evaluation_output"]
    # Cache news_context for potential retrain re-evaluation (Fix #6)
    state["_cached_news_context"] = agent_state.get("news_context") or agent_state["evaluation_output"].get("news_context")
    if _has_pending_retrain_candidate(state):
        _finalize_retrain_candidate(state)
    return state


def reporting_workflow_node(
    state: ForecastingWorkflowState,
    reporter_agent: ReporterAgent,
    agent_config: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
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
        "forced_action": state.get("forced_action"),
        "forced_reason": state.get("forced_reason"),
    }, config=config)
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
    config: RunnableConfig | None = None,
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
    }, config=config)
    if agent_state.get("success") is False:
        raise RuntimeError(f"ImprovementAgent failed: {agent_state.get('error')}")
    state["improvement_output"] = agent_state["improvement_output"]
    if state["improvement_output"].get("candidate_model_path"):
        state["original_model_path"] = state["model_path"]
        state["original_model_version"] = state["model_version"]
        state["original_forecasting_output"] = state["forecasting_output"]
        state["original_evaluation_output"] = state["evaluation_output"]
        state["original_reporter_output"] = state["reporter_output"]
        state["is_retrain"] = True
        state["retrain_count"] = state.get("retrain_count", 0) + 1
        state["model_path"] = state["improvement_output"]["candidate_model_path"]
        state["forced_action"] = None
        state["forced_reason"] = None
    elif not state["improvement_output"].get("is_retrain"):
        state["is_retrain"] = True
        state["forced_action"] = "reject_retrained_keep_old"
        state["forced_reason"] = state["improvement_output"].get("reason", "Retrain failed.")
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
    if improvement.get("candidate_model_path"):
        return "rerun"
    return "end"


def _has_pending_retrain_candidate(state: ForecastingWorkflowState) -> bool:
    improvement = state.get("improvement_output") or {}
    return (
        bool(state.get("is_retrain"))
        and bool(improvement.get("candidate_model_path"))
        and not bool(improvement.get("promotion_decided"))
    )


def _finalize_retrain_candidate(state: ForecastingWorkflowState) -> None:
    """Promote or reject the selected retrain candidate after re-evaluation."""
    improvement = state["improvement_output"] or {}
    candidate_path = Path(improvement["candidate_model_path"])
    evaluation_output = state["evaluation_output"] or {}
    old_reporter = state.get("original_reporter_output") or {}
    old_composite = float(improvement.get("old_composite_score", old_reporter.get("composite_score", 0.0)))
    new_composite = composite_score(
        evaluation_output.get("trust_score", 0.0),
        evaluation_output.get("risk_breakdown", {}),
    )
    decision_band = str(evaluation_output.get("decision_band", ""))
    should_promote = decision_band != "retrain" and new_composite > old_composite

    improvement.update({
        "promotion_decided": True,
        "candidate_temp_path": str(candidate_path),
        "candidate_evaluation_output": evaluation_output,
        "candidate_forecasting_output": state.get("forecasting_output"),
        "new_composite_score": round(new_composite, 2),
    })

    if should_promote:
        _promote_retrain_candidate(state, candidate_path, improvement, old_composite, new_composite)
        state["forced_action"] = None
        state["forced_reason"] = None
        return

    reason = (
        f"Rejected retrained candidate because evaluation band is '{decision_band}' "
        f"and composite score {new_composite:.2f} did not clear old {old_composite:.2f}."
    )
    _reject_retrain_candidate(state, candidate_path, improvement, reason)


def _promote_retrain_candidate(
    state: ForecastingWorkflowState,
    candidate_path: Path,
    improvement: dict[str, Any],
    old_composite: float,
    new_composite: float,
) -> None:
    artifacts_dir = candidate_path.parent
    next_version = get_next_model_version(artifacts_dir)
    version_dir = artifacts_dir / f"ver_{next_version}"
    shutil.move(str(candidate_path), str(version_dir))

    metadata = _read_metadata(version_dir)
    metadata.update({
        "version": next_version,
        "ticker": state["ticker"],
        "status": "promoted",
        "old_model_version": state.get("original_model_version"),
        "old_composite_score": old_composite,
        "new_composite_score": new_composite,
        "promotion_reason": (
            f"post-retrain evaluation band {state['evaluation_output'].get('decision_band')} "
            f"and composite {new_composite:.2f} beat old {old_composite:.2f}"
        ),
        "post_retrain_evaluation": state.get("evaluation_output"),
    })
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    improvement.update({
        "candidate_model_path": str(version_dir),
        "promoted": True,
        "new_model_version": next_version,
        "new_model_path": str(version_dir),
        "reason": metadata["promotion_reason"],
    })
    state["model_version"] = next_version
    state["model_path"] = str(version_dir)
    if state.get("forecasting_output"):
        state["forecasting_output"]["model_version"] = next_version
        state["forecasting_output"]["model_path"] = str(version_dir)


def _reject_retrain_candidate(
    state: ForecastingWorkflowState,
    candidate_path: Path,
    improvement: dict[str, Any],
    reason: str,
) -> None:
    shutil.rmtree(candidate_path, ignore_errors=True)
    improvement.update({
        "candidate_model_path": None,
        "candidate_temp_path_deleted": True,
        "promoted": False,
        "new_model_version": None,
        "new_model_path": None,
        "reason": reason,
    })
    state["model_path"] = state["original_model_path"]
    state["model_version"] = state["original_model_version"]
    state["forecasting_output"] = state["original_forecasting_output"]
    state["evaluation_output"] = state["original_evaluation_output"]
    state["reporter_output"] = state["original_reporter_output"]
    state["forced_action"] = "reject_retrained_keep_old"
    state["forced_reason"] = reason


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
