"""Agent node implementations.

Each section below owns the business logic for one agent. Agent classes should
only wire these nodes into graphs.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting import load_models, predict_with_intervals
from src.forecasting.trainer import train_xgboost_forecaster
from src.tools.tavily_search import search_tavily_news
from src.utils.config_manager import config_manager
from src.utils.logger import get_logger
from src.utils.report_writer import write_json_report, write_markdown_report

logger = get_logger(__name__)

from src.utils.scoring import clamp, composite_score, normalized_mape_risk, normalized_rmse_pct_risk, risk_from_ratio, safe_float
from src.prompts.factory import prompt_factory
from .states import (
    EvaluatorAgentState,
    ForecastingAgentState,
    ImprovementAgentState,
    ReporterAgentState,
)


# =============================================================================
# Forecasting Agent Nodes
# =============================================================================


def load_model_node(state: ForecastingAgentState) -> ForecastingAgentState:
    ticker = state["ticker"]
    model_path = Path(state["model_path"])
    model_config = state.get("model_config") or config_manager.model
    quantiles = model_config.get("quantiles", [0.025, 0.10, 0.50, 0.90, 0.975])

    models = load_models(ticker, artifacts_dir=model_path, quantiles=quantiles, version_dir=model_path)
    if not models:
        raise FileNotFoundError(f"No quantile models found in {model_path}")
    state["loaded_models"] = models
    return state


def predict_node(state: ForecastingAgentState) -> ForecastingAgentState:
    ticker = state["ticker"]
    df_raw = state["df_raw"].copy()
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    
    models = state["loaded_models"]
    preprocessing_config = state.get("preprocessing_config") or config_manager.preprocessing
    model_config = state.get("model_config") or config_manager.model
    horizon = int(state.get("horizon", 7))

    forecast = predict_with_intervals(
        ticker=ticker,
        horizon=horizon,
        models=models,
        df_raw=df_raw,
        feature_config=preprocessing_config.get("features", {}),
        preprocessing_config=preprocessing_config,
        model_config=model_config,
    )
    
    state["predictions"] = forecast["predictions"]
    state["holdout_metrics"] = forecast["holdout_metrics"]
    return state


def evaluate_holdout_node(state: ForecastingAgentState) -> ForecastingAgentState:
    ticker = state["ticker"]
    model_path = Path(state["model_path"])
    predictions = state["predictions"]
    df_raw = state["df_raw"]
    
    avg_80_width_pct = _average_interval_width_pct(predictions, "confidence_80")
    avg_95_width_pct = _average_interval_width_pct(predictions, "confidence_95")
    latest_close = float(df_raw["close"].iloc[-1])
    last_point = float(predictions[-1]["point_forecast"]) if predictions else latest_close
    forecast_return = ((last_point - latest_close) / latest_close) * 100.0 if latest_close else 0.0

    state["forecasting_output"] = {
        "ticker": ticker,
        "model_version": _version_from_path(model_path),
        "model_path": str(model_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_metrics": state["holdout_metrics"],
        "predictions": predictions,
        "forecast_diagnostics": {
            "avg_80_width_pct": round(avg_80_width_pct, 4),
            "avg_95_width_pct": round(avg_95_width_pct, 4),
            "forecast_7d_return_pct": round(forecast_return, 4),
            "latest_close": latest_close,
        },
    }
    return state


def _version_from_path(model_path: str | Path) -> int | None:
    match = re.search(r"ver_(\d+)$", str(model_path))
    return int(match.group(1)) if match else None


def _average_interval_width_pct(predictions: list[dict[str, Any]], interval_key: str) -> float:
    widths = []
    for row in predictions:
        point = row.get("point_forecast")
        interval = row.get(interval_key, {})
        lower = interval.get("lower")
        upper = interval.get("upper")
        if point and lower is not None and upper is not None:
            widths.append(((float(upper) - float(lower)) / abs(float(point))) * 100.0)
    return float(sum(widths) / len(widths)) if widths else 0.0


# =============================================================================
# Evaluator Agent Nodes
# =============================================================================


def gather_news_node(state: EvaluatorAgentState) -> EvaluatorAgentState:
    ticker = state.get("ticker") or state["forecasting_output"]["ticker"]
    cfg = state.get("agent_config") or config_manager.agent
    news_cfg = cfg.get("evaluator", {}).get("news", {})

    news_context = state.get("news_context")
    if news_context is None:
        query = (
            f"{ticker} stock latest news earnings guidance analyst regulation "
            f"product macro last {news_cfg.get('lookback_days', 7)} days"
        )
        raw_articles = search_tavily_news(
            query=query,
            max_results=int(news_cfg.get("max_results", 8)),
            search_depth=news_cfg.get("search_depth", "advanced"),
        )
        articles = _enrich_articles_batch(raw_articles, ticker, state.get("llm"))
        news_context = build_news_context(query, articles)

    state["news_context"] = news_context
    return state


def compute_live_accuracy_node(state: EvaluatorAgentState) -> EvaluatorAgentState:
    """Compare prior forecasts against actual close prices to compute live MAPE.

    Uses the previous daily reports stored in artifacts/reports. For each
    report, finds predictions whose dates now exist in df_recent (i.e. the
    actual price is known) and computes the absolute percentage error.
    This live_mape is a stronger signal than holdout_mape because it measures
    real-world forecast accuracy rather than in-sample test-set accuracy.
    """
    cfg = state.get("agent_config") or config_manager.agent
    report_dir = cfg.get("reporter", {}).get("reports_dir", "artifacts/reports")
    ticker = state.get("ticker") or state["forecasting_output"]["ticker"]
    df_recent = state["df_recent"].copy()

    # Build a date → actual_close lookup from df_recent
    if "date" not in df_recent.columns:
        state["live_mape"] = None
        return state
    df_recent["date"] = pd.to_datetime(df_recent["date"]).dt.date.astype(str)
    actual_by_date: dict[str, float] = dict(
        zip(df_recent["date"], df_recent["close"].astype(float))
    )

    prev_reports = load_previous_reports(report_dir, ticker, limit=14)

    errors: list[float] = []
    for report in prev_reports:
        for pred in report.get("forecasting", {}).get("predictions", []):
            pred_date = str(pred.get("date", ""))[:10]
            actual = actual_by_date.get(pred_date)
            if actual is None or actual <= 0:
                continue
            predicted = safe_float(pred.get("point_forecast"))
            if predicted > 0:
                errors.append(abs(actual - predicted) / actual)

    live_mape = float(sum(errors) / len(errors)) if errors else None
    state["live_mape"] = live_mape
    return state


def calculate_technical_risk_node(state: EvaluatorAgentState) -> EvaluatorAgentState:
    forecasting_output = state["forecasting_output"]
    df_recent = state["df_recent"].copy()
    news_context = state["news_context"]
    live_mape = state.get("live_mape")  # may be None if no prior forecasts exist

    risk_breakdown = calculate_risk_breakdown(forecasting_output, df_recent, news_context, live_mape)
    state["risk_breakdown"] = risk_breakdown
    return state


def compute_trust_score_node(state: EvaluatorAgentState) -> EvaluatorAgentState:
    cfg = state.get("agent_config") or config_manager.agent
    evaluator_cfg = cfg.get("evaluator", {})
    risk_breakdown = state["risk_breakdown"]
    news_context = state["news_context"]
    
    weights = evaluator_cfg.get("trust_score", {}).get("weights", {})
    weighted_risk = (
        risk_breakdown["holdout_mape_risk"] * float(weights.get("holdout_mape", 0.30))
        + risk_breakdown["holdout_rmse_pct_risk"] * float(weights.get("holdout_rmse_pct", 0.15))
        + risk_breakdown["interval_width_risk"] * float(weights.get("interval_width", 0.15))
        + risk_breakdown["recent_volatility_risk"] * float(weights.get("recent_volatility", 0.15))
        + risk_breakdown["trend_alignment_risk"] * float(weights.get("trend_alignment", 0.10))
        + risk_breakdown["news_risk"] * float(weights.get("news_risk", 0.15))
    )
    trust_score = round(clamp(100.0 - weighted_risk), 2)

    # Black Swan Veto
    aggregate_news_risk = safe_float(news_context.get("aggregate_news_risk", 0))
    has_extreme_severity = any(safe_float(a.get("event_severity", 0)) >= 0.9 for a in news_context.get("articles", []))
    is_black_swan = aggregate_news_risk >= 90 or has_extreme_severity
    state["is_black_swan"] = is_black_swan

    accept_threshold = float(evaluator_cfg.get("accept_threshold", 70))
    retrain_threshold = float(evaluator_cfg.get("retrain_threshold", 50))
    
    if is_black_swan:
        decision_band = "retrain"
        trust_score = 0.0
        risk_breakdown["black_swan_penalty"] = 100.0
    elif trust_score < retrain_threshold:
        decision_band = "retrain"
    elif trust_score >= accept_threshold:
        decision_band = "accept"
    else:
        decision_band = "warning"
        
    state["trust_score"] = trust_score
    state["decision_band"] = decision_band

    state["evaluation_output"] = {
        "trust_score": trust_score,
        "decision_band": decision_band,
        "reason": summarize_decision(decision_band, risk_breakdown),
        "risk_breakdown": {key: round(value, 2) for key, value in risk_breakdown.items()},
        "news_context": news_context,
    }
    return state


def _enrich_articles_batch(articles: list[dict[str, Any]], ticker: str, llm: Any = None) -> list[dict[str, Any]]:
    if not articles:
        return []
    if llm is not None:
        enriched = _llm_enrich_articles_batch(articles, ticker, llm)
        if enriched and len(enriched) == len(articles):
            return enriched
    return [heuristic_enrich_article(a) for a in articles]


def _llm_enrich_articles_batch(articles: list[dict[str, Any]], ticker: str, llm: Any) -> list[dict[str, Any]] | None:
    prompt = prompt_factory.get_prompt(
        "evaluator_agent",
        ticker=ticker,
        articles=json.dumps(articles)[:15000]
    )
    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", str(response)).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        if isinstance(data, list):
            return [normalize_enriched_article({**orig, **enriched}) for orig, enriched in zip(articles, data)]
        return None
    except Exception:
        return None


def calculate_risk_breakdown(
    forecasting_output: dict[str, Any],
    df_recent: pd.DataFrame,
    news_context: dict[str, Any],
    live_mape: float | None = None,
) -> dict[str, float]:
    metrics = forecasting_output.get("holdout_metrics", {})
    diagnostics = forecasting_output.get("forecast_diagnostics", {})
    latest_close = safe_float(diagnostics.get("latest_close"))
    if latest_close <= 0 and not df_recent.empty:
        latest_close = safe_float(df_recent["close"].iloc[-1])

    # Use 95% CI width for comparison: our expected width is also 95% (1.96 * sigma * sqrt(7))
    avg_95_width_pct = safe_float(diagnostics.get("avg_95_width_pct"))

    # Use live MAPE (actual vs predicted from prior forecasts) when available;
    # it is a stronger signal than holdout MAPE computed at training time.
    effective_mape = live_mape if live_mape is not None else safe_float(metrics.get("MAPE"))

    breakdown = {
        "holdout_mape_risk": normalized_mape_risk(safe_float(metrics.get("MAPE"))),
        "holdout_rmse_pct_risk": normalized_rmse_pct_risk(safe_float(metrics.get("RMSE")), latest_close),
        "interval_width_risk": dynamic_interval_width_risk(avg_95_width_pct, df_recent),
        "recent_volatility_risk": recent_volatility_risk(df_recent),
        "trend_alignment_risk": trend_alignment_risk(forecasting_output, df_recent),
        "news_risk": safe_float(news_context.get("aggregate_news_risk")),
    }
    if live_mape is not None:
        breakdown["live_mape_risk"] = normalized_mape_risk(live_mape)
        # Override holdout_mape_risk with effective (live) MAPE risk
        breakdown["holdout_mape_risk"] = normalized_mape_risk(effective_mape)
    return breakdown


def dynamic_interval_width_risk(avg_width_pct: float, df_recent: pd.DataFrame) -> float:
    """Compute interval-width risk relative to the stock's own realized volatility.

    A wide confidence interval is expected and honest for high-volatility stocks.
    Penalising absolute width causes high-beta stocks like NVDA to be incorrectly
    flagged as having poor models. Instead we compare the observed CI width against
    the theoretically expected width given realized daily volatility.

    Expected 95% CI width ≈ 1.96 * sigma_daily * sqrt(horizon=7).
    We tolerate up to 3× this expected width before reaching maximum risk, because
    quantile regression models naturally produce wider intervals than Gaussian theory
    (they capture distributional skew and asymmetry).
    """
    if df_recent.empty or len(df_recent) < 5:
        return risk_from_ratio(avg_width_pct / 100.0, 0.20)
    returns = df_recent["close"].astype(float).pct_change().dropna()
    sigma_daily = float(returns.std())
    # 95% CI of a 7-step random walk: 1.96 * sigma * sqrt(7)
    expected_width = max(1.96 * sigma_daily * (7 ** 0.5), 0.05)
    # Allow up to 3× expected before penalising fully
    bad_at = 3.0 * expected_width
    return risk_from_ratio(avg_width_pct / 100.0, bad_at)


def recent_volatility_risk(df_recent: pd.DataFrame) -> float:
    if len(df_recent) < 3:
        return 50.0
    returns = df_recent["close"].astype(float).pct_change().dropna()
    return risk_from_ratio(float(returns.std()), 0.05)


def trend_alignment_risk(forecasting_output: dict[str, Any], df_recent: pd.DataFrame) -> float:
    """Gradient risk — not binary. Risk scales with the size of the directional mismatch."""
    if len(df_recent) < 8:
        return 50.0
    close = df_recent["close"].astype(float)
    trend_7 = (close.iloc[-1] - close.iloc[-8]) / close.iloc[-8]
    lookback = min(30, len(close) - 1)
    trend_30 = (close.iloc[-1] - close.iloc[-lookback - 1]) / close.iloc[-lookback - 1]
    forecast_return = safe_float(
        forecasting_output.get("forecast_diagnostics", {}).get("forecast_7d_return_pct")
    ) / 100.0
    composite_trend = 0.65 * trend_7 + 0.35 * trend_30
    # Flat zone: neither forecast nor trend has a meaningful direction
    if abs(forecast_return) < 0.005 or abs(composite_trend) < 0.005:
        return 20.0
    if np.sign(composite_trend) == np.sign(forecast_return):
        # Same direction — risk scales with magnitude mismatch (over/under-shoot)
        magnitude_ratio = abs(forecast_return) / (abs(composite_trend) + 1e-9)
        return clamp(10.0 * abs(1.0 - magnitude_ratio), 0.0, 40.0)
    else:
        # Opposite direction — base risk 50, scaled up by how strong the conflict is
        conflict_strength = min(abs(composite_trend), abs(forecast_return)) / 0.10
        return clamp(50.0 + 50.0 * conflict_strength, 50.0, 100.0)


# Tiered keyword sets for richer heuristic enrichment
_NEGATIVE_HIGH = [
    "bankruptcy", "fraud", "sec investigation", "export ban", "chip ban",
    "indictment", "delisted", "recall", "catastrophic",
]
_NEGATIVE_MED = [
    "lawsuit", "probe", "cut", "miss", "downgrade", "ban", "weak",
    "falls", "drop", "concern", "below estimate", "penalty", "fine",
]
_POSITIVE_HIGH = [
    "blowout", "record revenue", "massive deal", "acquisition approved",
]
_POSITIVE_MED = [
    "beat", "raise guidance", "approval", "surge", "upgrade", "strong",
    "growth", "record", "partnership",
]
# Source credibility lookup (domain substring → score)
_SOURCE_CREDIBILITY: dict[str, float] = {
    "reuters.com": 0.95, "bloomberg.com": 0.95, "wsj.com": 0.90,
    "ft.com": 0.90, "cnbc.com": 0.80, "marketwatch.com": 0.78,
    "seekingalpha.com": 0.70, "yahoo.com": 0.70,
    "tradingview.com": 0.60, "tradingeconomics.com": 0.55,
}


def _source_credibility_score(url: str) -> float:
    url_lower = url.lower()
    for domain, score in _SOURCE_CREDIBILITY.items():
        if domain in url_lower:
            return score
    return 0.65  # unknown source — moderate credibility


def heuristic_enrich_article(article: dict[str, Any]) -> dict[str, Any]:
    text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
    sentiment = 0.0
    severity = 0.20
    if any(w in text for w in _NEGATIVE_HIGH):
        sentiment -= 1.0
        severity = max(severity, 0.90)
    if any(w in text for w in _NEGATIVE_MED):
        sentiment -= 0.5
        severity = max(severity, 0.60)
    if any(w in text for w in _POSITIVE_HIGH):
        sentiment += 1.0
        severity = max(severity, 0.80)
    if any(w in text for w in _POSITIVE_MED):
        sentiment += 0.5
        severity = max(severity, 0.50)
    # Neutral article with no signal — keep severity low
    if sentiment == 0.0:
        severity = 0.20
    sentiment = max(-1.0, min(1.0, sentiment))
    credibility = _source_credibility_score(str(article.get("url", "")))

    # Dynamic relevance — boost if article mentions ticker or company keywords
    relevance = _compute_relevance(text, article)

    # Dynamic recency — decay based on published_date age
    recency = _compute_recency_score(article.get("published_date"))

    return normalize_enriched_article({
        **article,
        "relevance": relevance,
        "sentiment": sentiment,
        "event_severity": severity,
        "recency_score": recency,
        "source_credibility": credibility,
        "volatility_impact": severity,
        "risk_reason": "Heuristic enrichment (LLM unavailable). Tiered keyword + source credibility scoring.",
    })


def _compute_relevance(text: str, article: dict[str, Any]) -> float:
    """Estimate article relevance from text content (0.0–1.0).

    A base score of 0.40 is given to all articles. The score is boosted
    when the article title/summary contains stock-related terms like the
    ticker symbol, 'stock', 'shares', 'earnings', 'guidance', etc.
    """
    score = 0.40
    url = str(article.get("url", "")).lower()
    # Check for common stock-specific terms
    stock_terms = ["stock", "shares", "earnings", "guidance", "forecast",
                   "revenue", "profit", "margin", "analyst", "target price",
                   "quarterly", "dividend"]
    for term in stock_terms:
        if term in text:
            score += 0.05
    # Check URL for finance domains (more likely to be relevant)
    if any(d in url for d in ["finance", "stock", "market", "invest"]):
        score += 0.10
    return min(1.0, score)


def _compute_recency_score(published_date: Any) -> float:
    """Score recency from 1.0 (today) to 0.10 (14+ days old).

    If published_date is missing or unparseable, returns a conservative 0.50.
    """
    if not published_date:
        return 0.50
    try:
        from dateutil.parser import parse as dateparse
        pub = dateparse(str(published_date)).date()
        age_days = (date.today() - pub).days
        if age_days <= 0:
            return 1.0
        if age_days >= 14:
            return 0.10
        # Linear decay from 1.0 to 0.10 over 14 days
        return round(1.0 - (0.90 * age_days / 14.0), 2)
    except Exception:
        return 0.50


def normalize_enriched_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(article.get("title", "")),
        "url": str(article.get("url", "")),
        "published_date": article.get("published_date"),
        "summary": str(article.get("summary", ""))[:800],
        "relevance": clamp(safe_float(article.get("relevance")), 0, 1),
        "sentiment": max(-1.0, min(1.0, safe_float(article.get("sentiment")))),
        "event_severity": clamp(safe_float(article.get("event_severity")), 0, 1),
        "recency_score": clamp(safe_float(article.get("recency_score")), 0, 1),
        "source_credibility": clamp(safe_float(article.get("source_credibility")), 0, 1),
        "volatility_impact": clamp(safe_float(article.get("volatility_impact")), 0, 1),
        "risk_reason": str(article.get("risk_reason", "")),
    }


def build_news_context(query: str, articles: list[dict[str, Any]]) -> dict[str, Any]:
    if not articles:
        return {
            "query": query,
            "articles": [],
            "aggregate_news_risk": 0.0,
            "summary": "No Tavily news context available; news risk set to 0.",
        }
    risks = []
    for article in articles:
        negative_sentiment = max(0.0, -safe_float(article.get("sentiment")))
        article_risk = 100.0 * (
            0.25 * safe_float(article.get("relevance"))
            + 0.25 * safe_float(article.get("event_severity"))
            + 0.20 * safe_float(article.get("volatility_impact"))
            + 0.15 * negative_sentiment
            + 0.10 * safe_float(article.get("recency_score"))
            + 0.05 * safe_float(article.get("source_credibility"))
        )
        risks.append(clamp(article_risk))
    aggregate = round(float(sum(risks) / len(risks)), 2)
    high_risk_titles = [a["title"] for a, risk in zip(articles, risks) if risk >= 60][:3]
    summary = (
        f"Reviewed {len(articles)} article(s). Aggregate news risk is {aggregate:.2f}."
        + (f" Highest-risk items: {', '.join(high_risk_titles)}." if high_risk_titles else "")
    )
    return {
        "query": query,
        "articles": articles,
        "aggregate_news_risk": aggregate,
        "summary": summary,
    }


def summarize_decision(decision_band: str, risk_breakdown: dict[str, float]) -> str:
    top_risk = max(risk_breakdown.items(), key=lambda item: item[1])
    return f"{decision_band} band; highest normalized risk is {top_risk[0]} at {top_risk[1]:.2f}."


# =============================================================================
# Reporter Agent Nodes
# =============================================================================


def assess_trend_node(state: ReporterAgentState) -> ReporterAgentState:
    ticker = state.get("ticker") or state["forecasting_output"]["ticker"]
    cfg = state.get("agent_config") or config_manager.agent
    reporter_cfg = cfg.get("reporter", {})
    report_dir = Path(reporter_cfg.get("reports_dir", "artifacts/reports"))
    history_n = int(reporter_cfg.get("history_n", 7))
    
    previous_reports = state.get("previous_reports")
    if previous_reports is None:
        previous_reports = load_previous_reports(report_dir, ticker, history_n)
        
    evaluation_output = state["evaluation_output"]
    score = composite_score(
        evaluation_output.get("trust_score", 0.0),
        evaluation_output.get("risk_breakdown", {}),
    )
    trend = assess_history_trend(
        score,
        previous_reports,
        float(reporter_cfg.get("strong_degradation_threshold", 8.0)),
    )
    
    state["composite_score"] = score
    state["trend_assessment"] = trend
    state["previous_reports"] = previous_reports
    return state


def determine_action_node(state: ReporterAgentState) -> ReporterAgentState:
    evaluation_output = state["evaluation_output"]
    is_retrain = bool(state.get("is_retrain", False))
    cfg = state.get("agent_config") or config_manager.agent
    evaluator_cfg = cfg.get("evaluator", {})
    trend = state["trend_assessment"]

    trust_score = safe_float(evaluation_output.get("trust_score"))
    accept_threshold = float(evaluator_cfg.get("accept_threshold", 70))
    retrain_threshold = float(evaluator_cfg.get("retrain_threshold", 50))

    forced_action = state.get("forced_action")
    forced_reason = state.get("forced_reason")
    if forced_action:
        action = forced_action
        reason = forced_reason or "Action forced by workflow."
    elif trust_score < retrain_threshold and not is_retrain:
        action = "retrain"
        reason = f"Trust score {trust_score:.2f} is below retrain threshold {retrain_threshold:.2f}."
    elif is_retrain:
        action = "accept_after_retrain"
        reason = "Retrained model accepted for final reporting."
    elif trust_score >= accept_threshold:
        action = "accept"
        reason = f"Trust score {trust_score:.2f} meets accept threshold {accept_threshold:.2f}."
    elif trend == "degrading":
        action = "retrain"
        reason = "Trust score is in warning band and composite score is degrading versus history."
    else:
        action = "accept"
        reason = f"Trust score {trust_score:.2f} is in warning band; no strong degradation found."
        
    state["action"] = action
    state["reason"] = reason
    return state


def generate_markdown_report_node(state: ReporterAgentState) -> ReporterAgentState:
    """Use LLM to generate the full markdown report content.

    Builds a rich, context-aware prompt from all available pipeline outputs
    and asks the LLM to author a professional analyst report in Markdown.
    Falls back to an empty string so that format_report_node can use the
    rule-based writer as a safety net.
    """
    llm = state.get("llm")
    state["llm_markdown"] = ""  # default: empty → fallback to rule-based

    if not llm:
        return state

    forecasting_output = state["forecasting_output"]
    evaluation_output = state["evaluation_output"]
    ticker = state.get("ticker") or forecasting_output["ticker"]
    improvement = state.get("improvement_output") or {}

    # Build forecast table string for the prompt
    forecast_table = _build_forecast_table(forecasting_output.get("predictions", []))

    # Summarise improvement outcome
    if improvement.get("is_retrain"):
        improvement_info = (
            f"Retrain attempted. Promoted: {improvement.get('promoted')}. "
            f"Reason: {improvement.get('reason', '')}. "
            f"New model v{improvement.get('new_model_version')} composite "
            f"{improvement.get('new_composite_score', 0):.2f} vs old "
            f"{improvement.get('old_composite_score', 0):.2f}."
        )
    elif improvement.get("skip_retrain") is True or (
        improvement and not improvement.get("is_retrain")
    ):
        improvement_info = improvement.get("reason", "No retrain performed.")
    else:
        improvement_info = "No retrain performed."

    metrics = forecasting_output.get("holdout_metrics", {})
    holdout_str = (
        f"MAE={metrics.get('MAE', 0):.4f}, "
        f"RMSE={metrics.get('RMSE', 0):.4f}, "
        f"MAPE={metrics.get('MAPE', 0):.2%}"
    )

    try:
        prompt = prompt_factory.get_prompt(
            "reporter_agent_markdown",
            ticker=ticker,
            run_date=state.get("run_date", date.today().isoformat()),
            model_version=forecasting_output.get("model_version", "?"),
            action=state["action"],
            trust_score=f"{safe_float(evaluation_output.get('trust_score')):.2f}",
            composite_score=f"{state.get('composite_score', 0):.2f}",
            decision_band=evaluation_output.get("decision_band", ""),
            reason=state["reason"],
            trend_assessment=state.get("trend_assessment", "unknown"),
            holdout_metrics=holdout_str,
            forecast_table=forecast_table,
            risk_breakdown=json.dumps(
                {k: round(float(v), 2) for k, v in evaluation_output.get("risk_breakdown", {}).items()},
                indent=2,
            ),
            news_context=evaluation_output.get("news_context", {}).get("summary", "N/A"),
            improvement_info=improvement_info,
        )
        response = llm.invoke(prompt)
        state["llm_markdown"] = getattr(response, "content", str(response)).strip()
    except Exception as e:
        state["llm_markdown"] = ""  # trigger fallback in format_report_node
        state["insight_summary"] = f"LLM markdown generation failed: {e}"
        return state

    state["insight_summary"] = ""  # no longer needed; full report from LLM
    return state


def _build_forecast_table(predictions: list[dict[str, Any]]) -> str:
    """Render the 7-day prediction list as a Markdown table string."""
    if not predictions:
        return "No predictions available."
    lines = [
        "| Date | Point Forecast | 80% CI Low | 80% CI High | 95% CI Low | 95% CI High |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in predictions:
        c80 = row.get("confidence_80", {})
        c95 = row.get("confidence_95", {})
        lines.append(
            f"| {row.get('date', '')} "
            f"| **{row.get('point_forecast', 0):.2f}** "
            f"| {c80.get('lower', 0):.2f} | {c80.get('upper', 0):.2f} "
            f"| {c95.get('lower', 0):.2f} | {c95.get('upper', 0):.2f} |"
        )
    return "\n".join(lines)


def format_report_node(state: ReporterAgentState) -> ReporterAgentState:
    ticker = state.get("ticker") or state["forecasting_output"]["ticker"]
    run_date = state.get("run_date") or date.today().isoformat()
    cfg = state.get("agent_config") or config_manager.agent
    reporter_cfg = cfg.get("reporter", {})
    report_dir = Path(reporter_cfg.get("reports_dir", "artifacts/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    # Always use a single canonical filename — no _retrain suffix.
    # The final report (after retrain if any) overwrites the initial one.
    json_path = report_dir / f"{run_date}_{ticker}_report.json"
    md_path   = report_dir / f"{run_date}_{ticker}_report.md"

    reporter_output = {
        "action": state["action"],
        "reason": state["reason"],
        "insight_summary": state.get("insight_summary", ""),
        "trend_assessment": state["trend_assessment"],
        "is_retrain": bool(state.get("is_retrain", False)),
        "report_paths": {"json": str(json_path), "markdown": str(md_path)},
        "composite_score": state["composite_score"],
    }
    payload = {
        "ticker": ticker,
        "run_id": state.get("run_id"),
        "run_date": run_date,
        "forecasting": state["forecasting_output"],
        "evaluation": state["evaluation_output"],
        "reporter": reporter_output,
        "improvement": state.get("improvement_output"),
    }
    write_json_report(json_path, payload)

    # Use LLM-generated markdown when available; fall back to rule-based writer
    llm_markdown = state.get("llm_markdown", "")
    if llm_markdown:
        with open(md_path, "w") as f:
            f.write(llm_markdown.rstrip() + "\n")
    else:
        write_markdown_report(md_path, payload)

    state["reporter_output"] = reporter_output
    state["report_paths"] = reporter_output["report_paths"]
    return state


def load_previous_reports(report_dir: str | Path, ticker: str, limit: int = 7) -> list[dict[str, Any]]:
    report_dir = Path(report_dir)
    if not report_dir.exists():
        return []
    reports = []
    for path in report_dir.glob(f"*_{ticker}*_report.json"):
        try:
            with open(path) as f:
                payload = json.load(f)
            reports.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    reports.sort(key=lambda item: item.get("run_date") or item.get("forecasting", {}).get("generated_at", ""))
    return reports[-limit:]


def assess_history_trend(
    current_composite: float,
    previous_reports: list[dict[str, Any]],
    degradation_threshold: float,
) -> str:
    if len(previous_reports) < 3:
        return "insufficient_history"
    previous_scores = [
        safe_float(report.get("reporter", {}).get("composite_score"))
        for report in previous_reports
        if report.get("reporter", {}).get("composite_score") is not None
    ]
    if len(previous_scores) < 3:
        return "insufficient_history"
    avg = sum(previous_scores) / len(previous_scores)
    delta = current_composite - avg
    if delta < -abs(degradation_threshold):
        return "degrading"
    if delta > abs(degradation_threshold):
        return "improving"
    return "stable"


# =============================================================================
# Improvement Agent Nodes
# =============================================================================


DEFAULT_CANDIDATES = [
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.03},
    {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.07},
]


def diagnose_failure_node(state: ImprovementAgentState) -> ImprovementAgentState:
    """Diagnose the primary cause of model failure before generating retrain candidates.

    This prevents wasting retrains on market-driven volatility spikes that
    hyperparameter tuning cannot fix. When market volatility is the root cause
    (not model inaccuracy), we set skip_retrain=True so the improvement agent
    returns without training — the old model stays.
    """
    risk = state.get("evaluation_output", {}).get("risk_breakdown", {})
    primary_cause = max(risk.items(), key=lambda x: x[1])[0] if risk else "unknown"

    is_market_volatile = safe_float(risk.get("recent_volatility_risk", 0)) > 70
    is_accuracy_poor = safe_float(risk.get("holdout_mape_risk", 0)) > 70
    is_interval_wide = safe_float(risk.get("interval_width_risk", 0)) > 70
    is_trend_misaligned = safe_float(risk.get("trend_alignment_risk", 0)) > 70

    diagnosis = {
        "primary_cause": primary_cause,
        "is_market_volatile": is_market_volatile,
        "is_accuracy_poor": is_accuracy_poor,
        "is_interval_wide": is_interval_wide,
        "is_trend_misaligned": is_trend_misaligned,
    }

    # Market volatility alone (without accuracy degradation) is not a model problem.
    # Retraining cannot fix an unpredictable market regime — skip to avoid noise.
    if is_market_volatile and not is_accuracy_poor and not is_interval_wide:
        state["skip_retrain"] = True
        state["skip_reason"] = (
            f"Primary cause is market volatility (recent_volatility_risk={risk.get('recent_volatility_risk', 0):.1f}), "
            "not model degradation. Retraining skipped."
        )
    else:
        state["skip_retrain"] = False
        state["skip_reason"] = None

    state["diagnosis"] = diagnosis
    return state


def generate_candidates_node(state: ImprovementAgentState) -> ImprovementAgentState:
    # If diagnosis determined retrain is unnecessary, emit an empty improvement result
    if state.get("skip_retrain", False):
        state["candidates"] = []
        return state

    ticker = state["ticker"]
    model_config = state.get("model_config") or config_manager.model
    agent_config = state.get("agent_config") or config_manager.agent
    news_context = state.get("evaluation_output", {}).get("news_context")
    llm = state.get("llm")

    candidates = []
    if llm:
        old_params = model_config.get("xgb_params", {})
        metrics = state.get("forecasting_output", {}).get("holdout_metrics", {})
        news_summary = news_context.get("summary", "") if news_context else ""
        risk_breakdown = state.get("evaluation_output", {}).get("risk_breakdown", {})
        diagnosis = state.get("diagnosis", {})

        prompt = prompt_factory.get_prompt(
            "improvement_agent",
            ticker=ticker,
            old_params=json.dumps(old_params),
            metrics=json.dumps(metrics),
            risk_breakdown=json.dumps(risk_breakdown),
            news_summary=news_summary
        )
        try:
            response = llm.invoke(prompt)
            text = getattr(response, "content", str(response)).strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed_candidates = json.loads(text.strip())
            if isinstance(parsed_candidates, list) and len(parsed_candidates) > 0:
                candidates = parsed_candidates
        except Exception:
            pass

    if not candidates:
        candidates = agent_config.get("improvement", {}).get("candidates", DEFAULT_CANDIDATES)

    state["candidates"] = candidates
    return state


def evaluate_candidates_node(state: ImprovementAgentState) -> ImprovementAgentState:
    """Train all candidates to temporary directories.

    We intentionally do NOT commit candidates to permanent version directories
    here. Each candidate is trained into an isolated tmp_* folder inside
    artifacts_dir. select_best_candidate_node then renames the winner to the
    next official ver_N and deletes the losers.
    """
    # Short-circuit: diagnose_failure_node may have decided retrain is not warranted
    if state.get("skip_retrain", False):
        state["candidate_results"] = []
        return state

    import shutil
    from src.forecasting.trainer import (
        train_quantile_models, save_quantile_models, compute_metrics,
        _select_point_quantile,
    )
    from src.preprocessing import preprocess_data

    ticker = state["ticker"]
    df_raw = state["df_raw"]
    preprocessing_config = state.get("preprocessing_config") or config_manager.preprocessing
    model_config = state.get("model_config") or config_manager.model
    agent_config = state.get("agent_config") or config_manager.agent
    news_context = state.get("evaluation_output", {}).get("news_context")
    candidates = state["candidates"]

    artifacts_dir = Path(model_config.get("artifacts_dir", "artifacts/models"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess ONCE — all candidates share the same data split
    preprocessing_result = preprocess_data(df_raw, preprocessing_config)
    X_train = preprocessing_result["X_train"]
    X_test  = preprocessing_result["X_test"]
    y_train = preprocessing_result["y_train"]
    y_test  = preprocessing_result["y_test"]
    feature_columns = preprocessing_result["feature_columns"]

    candidate_results = []

    for i, candidate in enumerate(candidates):
        candidate_model_cfg = _merge_candidate_model_config(model_config, candidate)

        # Train into a temporary directory inside artifacts_dir
        tmp_dir = artifacts_dir / f"tmp_{i}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        quantiles = candidate_model_cfg.get("quantiles", [0.025, 0.10, 0.50, 0.90, 0.975])
        xgb_params = candidate_model_cfg.get("xgb_params", {})

        models = train_quantile_models(X_train, y_train, quantiles, xgb_params)
        save_quantile_models(models, ticker, tmp_dir)

        point_q = _select_point_quantile(models)
        y_pred  = models[point_q].predict(X_test)
        test_metrics = compute_metrics(y_pred, y_test.values)

        logger.info(
            f"Candidate {i+1}/{len(candidates)} trained | "
            f"MAE: {test_metrics['MAE']:.2f}, MAPE: {test_metrics['MAPE']:.2%}"
        )

        # Save metadata into tmp dir
        json.dump(
            {"tmp": True, "candidate_index": i, "xgb_params": xgb_params,
             "test_metrics": test_metrics, "trained_at": datetime.now(timezone.utc).isoformat(),
             "feature_columns": feature_columns},
            open(tmp_dir / "metadata.json", "w"), indent=2,
        )

        # Run full forecast + evaluation on this candidate
        fc_state = {
            "ticker": ticker,
            "df_raw": df_raw,
            "model_path": str(tmp_dir),
            "horizon": state.get("horizon", 7),
            "preprocessing_config": preprocessing_config,
            "model_config": candidate_model_cfg,
        }
        fc_state = load_model_node(fc_state)
        fc_state = predict_node(fc_state)
        fc_state = evaluate_holdout_node(fc_state)
        forecasting_output = fc_state["forecasting_output"]

        ev_state = {
            "ticker": ticker,
            "forecasting_output": forecasting_output,
            "df_recent": df_raw.tail(30),
            "agent_config": agent_config,
            "news_context": news_context,
            "llm": state.get("llm"),
        }
        ev_state = gather_news_node(ev_state)
        ev_state = calculate_technical_risk_node(ev_state)
        ev_state = compute_trust_score_node(ev_state)
        evaluation_output = ev_state["evaluation_output"]

        score = composite_score(
            evaluation_output.get("trust_score", 0.0),
            evaluation_output.get("risk_breakdown", {}),
        )
        candidate_results.append({
            "tmp_dir": str(tmp_dir),
            "metrics": forecasting_output["holdout_metrics"],
            "training_metrics": test_metrics,
            "forecasting_output": forecasting_output,
            "evaluation_output": evaluation_output,
            "composite_score": score,
            "params": candidate,
            "models": models,
            "feature_columns": preprocessing_result["feature_columns"],
        })

    state["candidate_results"] = candidate_results
    return state


def select_best_candidate_node(state: ImprovementAgentState) -> ImprovementAgentState:
    """Pick the winner, commit it as an official ver_N, delete all losers.

    This is the ONLY place where a permanent model directory is created.
    All tmp_* directories are removed after this node regardless of outcome.
    """
    import shutil
    from src.forecasting.trainer import get_next_model_version, save_quantile_models
    from datetime import datetime

    # Short-circuit: if retrain was skipped due to diagnosis, emit a skip outcome
    if state.get("skip_retrain", False):
        state["improvement_output"] = {
            "is_retrain": False,
            "promoted": False,
            "reason": state.get("skip_reason", "Retrain skipped by diagnosis."),
            "diagnosis": state.get("diagnosis"),
        }
        return state

    old_model_version = state.get("model_version")
    old_model_path = state.get("model_path")
    old_reporter = state["reporter_output"]
    old_composite = float(old_reporter.get("composite_score", 0.0))
    model_config = state.get("model_config") or config_manager.model
    artifacts_dir = Path(model_config.get("artifacts_dir", "artifacts/models"))
    candidate_results = state["candidate_results"]

    if not candidate_results:
        state["improvement_output"] = {
            "is_retrain": False,
            "promoted": False,
            "reason": "No candidates were trained.",
            "diagnosis": state.get("diagnosis"),
        }
        return state

    # Find the best candidate among all trained
    best = max(candidate_results, key=lambda c: c["composite_score"])
    losers = [c for c in candidate_results if c is not best]

    promoted = best["composite_score"] > old_composite

    if promoted:
        # Rename winner's tmp dir → official ver_N
        next_version = get_next_model_version(artifacts_dir)
        version_dir = artifacts_dir / f"ver_{next_version}"
        shutil.move(str(best["tmp_dir"]), str(version_dir))

        # Write proper metadata
        import json as _json
        metadata = {
            "version": next_version,
            "ticker": state["ticker"],
            "trained_at": datetime.now().isoformat(),
            "status": "promoted",
            "test_metrics": best["training_metrics"],
            "xgb_params": best["params"],
            "feature_columns": best.get("feature_columns", []),
            "old_model_version": old_model_version,
            "old_composite_score": old_composite,
            "new_composite_score": best["composite_score"],
            "promotion_reason": (
                f"composite {best['composite_score']:.2f} beat old {old_composite:.2f}"
            ),
        }
        _json.dump(metadata, open(version_dir / "metadata.json", "w"), indent=2, default=str)

        reason = (
            f"Promoted retrained version {next_version} because composite "
            f"{best['composite_score']:.2f} beat old {old_composite:.2f}."
        )
        best["version"] = next_version
        best["version_dir"] = str(version_dir)
    else:
        reason = (
            f"Best candidate composite {best['composite_score']:.2f} did not "
            f"beat old {old_composite:.2f}. No new model saved."
        )
        best["version"] = None
        best["version_dir"] = None

    # Delete ALL tmp dirs (winners already moved or not promoted)
    for candidate in candidate_results:
        tmp = Path(candidate["tmp_dir"])
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    state["best_candidate"] = best
    state["improvement_output"] = {
        "is_retrain": True,
        "old_model_version": old_model_version,
        "new_model_version": best["version"],
        "new_model_path": best["version_dir"],
        "promoted": promoted,
        "reason": reason,
        "old_composite_score": old_composite,
        "new_composite_score": best["composite_score"],
        "retrained_metrics": best["metrics"],
        "diagnosis": state.get("diagnosis"),
    }
    return state


def _merge_candidate_model_config(model_config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    merged = dict(model_config)
    xgb_params = dict(model_config.get("xgb_params", {}))
    xgb_params.update(candidate)
    merged["xgb_params"] = xgb_params
    return merged


def _update_model_metadata(version_dir: Path, **updates: Any) -> None:
    metadata_path = version_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            metadata = {}
    metadata.setdefault("version_dir", str(version_dir))
    metadata.update(updates)
    metadata["promotion_checked_at"] = datetime.now(timezone.utc).isoformat()
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
