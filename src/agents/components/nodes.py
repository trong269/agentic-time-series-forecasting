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
    model_path = Path(state["model_path"])
    
    models = state["loaded_models"]
    preprocessing_config = state.get("preprocessing_config") or config_manager.preprocessing
    model_config = state.get("model_config") or config_manager.model
    agent_config = state.get("agent_config") or config_manager.agent
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
    predictions = state["predictions"]
    
    avg_80_width_pct = _average_interval_width_pct(predictions, "confidence_80")
    avg_95_width_pct = _average_interval_width_pct(predictions, "confidence_95")
    latest_close = float(df_raw["close"].iloc[-1])
    last_point = float(predictions[-1]["point_forecast"]) if predictions else latest_close
    forecast_return = ((last_point - latest_close) / latest_close) * 100.0 if latest_close else 0.0
    live_mape = _compute_live_mape_from_previous_reports(
        df_raw,
        ticker,
        agent_config,
    )

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
            "live_mape": live_mape,
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


def _compute_live_mape_from_previous_reports(
    df_raw: pd.DataFrame,
    ticker: str,
    agent_config: dict[str, Any],
) -> float | None:
    """Compare previous published forecasts with actual closes now in df_raw."""
    if "date" not in df_raw.columns:
        return None

    report_cfg = agent_config.get("reporter", {})
    report_dir = report_cfg.get("reports_dir", "artifacts/reports")
    df_actuals = df_raw.copy()
    df_actuals["date"] = pd.to_datetime(df_actuals["date"]).dt.date.astype(str)
    actual_by_date: dict[str, float] = dict(
        zip(df_actuals["date"], df_actuals["close"].astype(float))
    )

    errors: list[float] = []
    for report in load_previous_reports(report_dir, ticker, limit=14):
        for pred in report.get("forecasting", {}).get("predictions", []):
            pred_date = str(pred.get("date", ""))[:10]
            actual = actual_by_date.get(pred_date)
            if actual is None or actual <= 0:
                continue
            predicted = safe_float(pred.get("point_forecast"))
            if predicted > 0:
                errors.append(abs(actual - predicted) / actual)

    return float(sum(errors) / len(errors)) if errors else None


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


def calculate_technical_risk_node(state: EvaluatorAgentState) -> EvaluatorAgentState:
    forecasting_output = state["forecasting_output"]
    df_recent = state["df_recent"].copy()
    news_context = state["news_context"]
    live_mape = forecasting_output.get("forecast_diagnostics", {}).get("live_mape")

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


def decide_action_node(state: ReporterAgentState) -> ReporterAgentState:
    ticker = state.get("ticker") or state["forecasting_output"]["ticker"]
    cfg = state.get("agent_config") or config_manager.agent
    evaluator_cfg = cfg.get("evaluator", {})
    reporter_cfg = cfg.get("reporter", {})
    report_dir = Path(reporter_cfg.get("reports_dir", "artifacts/reports"))
    history_n = int(reporter_cfg.get("history_n", 7))
    degradation_threshold = float(reporter_cfg.get("strong_degradation_threshold", 8.0))
    
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
        degradation_threshold,
    )
    is_retrain = bool(state.get("is_retrain", False))

    trust_score = safe_float(evaluation_output.get("trust_score"))
    accept_threshold = float(evaluator_cfg.get("accept_threshold", 70))
    retrain_threshold = float(evaluator_cfg.get("retrain_threshold", 50))
    trend_factors = _build_reporter_trend_factors(
        state=state,
        ticker=ticker,
        report_dir=report_dir,
        history_n=history_n,
        current_composite=score,
        score_trend=trend,
        previous_reports=previous_reports,
    )
    trend = _classify_reporter_trend(trend, trend_factors)

    forced_action = state.get("forced_action")
    forced_reason = state.get("forced_reason")
    if forced_action:
        action = forced_action
        reason = forced_reason or "Action forced by workflow."
    elif is_retrain:
        action = "accept_after_retrain"
        reason = "Retrained model accepted for final reporting."
    elif trust_score < retrain_threshold and not is_retrain:
        model_signal = safe_float(trend_factors.get("model_failure_signal"))
        persistence = safe_float(trend_factors.get("risk_persistence_score"))
        market_signal = safe_float(trend_factors.get("market_regime_shift_signal"))
        news_signal = safe_float(trend_factors.get("news_driven_signal"))
        if max(market_signal, news_signal) >= 70 and model_signal < 55 and persistence < 65:
            action = "accept"
            reason = (
                f"Trust score {trust_score:.2f} is below retrain threshold, but recent reports indicate "
                "the risk is primarily market/news driven rather than persistent model failure."
            )
        else:
            action = "retrain"
            reason = f"Trust score {trust_score:.2f} is below retrain threshold {retrain_threshold:.2f}."
    elif trust_score >= accept_threshold:
        action = "accept"
        reason = f"Trust score {trust_score:.2f} meets accept threshold {accept_threshold:.2f}."
    elif trend in {"degrading", "degrading_model"}:
        action = "retrain"
        reason = (
            "Trust score is in warning band and history indicates persistent model degradation."
        )
    elif (
        safe_float(trend_factors.get("risk_persistence_score")) >= 75
        and safe_float(trend_factors.get("model_failure_signal")) >= 60
    ):
        action = "retrain"
        reason = "Trust score is in warning band with persistent model-risk signals in prior reports."
    else:
        action = "accept"
        reason = f"Trust score {trust_score:.2f} is in warning band; no strong degradation found."

    state["composite_score"] = score
    state["trend_assessment"] = trend
    state["trend_factors"] = trend_factors
    state["previous_reports"] = previous_reports
    state["action"] = action
    state["reason"] = reason
    return state


def _build_reporter_trend_factors(
    *,
    state: ReporterAgentState,
    ticker: str,
    report_dir: Path,
    history_n: int,
    current_composite: float,
    score_trend: str,
    previous_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    fallback = _rule_based_trend_factors(
        state=state,
        current_composite=current_composite,
        score_trend=score_trend,
        previous_reports=previous_reports,
    )
    llm = state.get("llm")
    if not llm:
        return fallback

    markdown_reports = _load_previous_markdown_reports(report_dir, ticker, history_n)
    if not markdown_reports:
        return fallback

    extracted = _extract_trend_factors_with_llm(
        llm=llm,
        markdown_reports=markdown_reports,
        forecasting_output=state["forecasting_output"],
        evaluation_output=state["evaluation_output"],
        fallback=fallback,
    )
    if safe_float(extracted.get("llm_confidence")) < 0.35:
        return fallback
    return extracted


def _rule_based_trend_factors(
    *,
    state: ReporterAgentState,
    current_composite: float,
    score_trend: str,
    previous_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    previous_scores = [
        safe_float(report.get("reporter", {}).get("composite_score"))
        for report in previous_reports
        if report.get("reporter", {}).get("composite_score") is not None
    ]
    previous_avg = float(sum(previous_scores) / len(previous_scores)) if previous_scores else current_composite
    score_delta = current_composite - previous_avg
    risk = state["evaluation_output"].get("risk_breakdown", {})
    top_risks = sorted(risk.items(), key=lambda item: safe_float(item[1]), reverse=True)[:3]
    top_risk_names = [name for name, _ in top_risks]
    model_failure_signal = max(
        safe_float(risk.get("holdout_mape_risk")),
        safe_float(risk.get("holdout_rmse_pct_risk")),
        safe_float(risk.get("interval_width_risk")),
    )
    market_signal = safe_float(risk.get("recent_volatility_risk"))
    news_signal = safe_float(risk.get("news_risk"))
    direction_signal = safe_float(risk.get("trend_alignment_risk"))
    risk_persistence = clamp(
        max(model_failure_signal, direction_signal) + max(0.0, -score_delta) * 2.0,
        0.0,
        100.0,
    )
    return {
        "source": "rules",
        "historical_pattern": score_trend,
        "score_delta": round(score_delta, 2),
        "previous_composite_avg": round(previous_avg, 2),
        "repeated_top_risks": top_risk_names,
        "risk_persistence_score": round(risk_persistence, 2),
        "model_failure_signal": round(model_failure_signal, 2),
        "market_regime_shift_signal": round(market_signal, 2),
        "news_driven_signal": round(news_signal, 2),
        "forecast_direction_error_signal": round(direction_signal, 2),
        "retrain_effectiveness": _rule_based_retrain_effectiveness(previous_reports),
        "llm_confidence": 0.0,
        "rationale": "Rule-based fallback from current risk breakdown and historical composite scores.",
    }


def _extract_trend_factors_with_llm(
    *,
    llm: Any,
    markdown_reports: list[dict[str, str]],
    forecasting_output: dict[str, Any],
    evaluation_output: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    reports_payload = json.dumps(markdown_reports, ensure_ascii=True)[:24000]
    current_payload = json.dumps(
        {
            "forecasting_output": _compact_forecasting_output(forecasting_output),
            "evaluation_output": evaluation_output,
            "fallback_rule_factors": fallback,
        },
        default=str,
        ensure_ascii=True,
    )[:12000]
    try:
        prompt = prompt_factory.get_prompt(
            "reporter_trend_extractor",
            markdown_reports=reports_payload,
            current_context=current_payload,
        )
        response = llm.invoke(prompt)
        raw_text = getattr(response, "content", str(response)).strip()
        parsed = _parse_json_object(raw_text)
        return _normalize_trend_factors(parsed, fallback)
    except Exception as exc:
        fallback = dict(fallback)
        fallback["rationale"] = f"Rule-based fallback because LLM trend extraction failed: {exc}"
        return fallback


def _compact_forecasting_output(forecasting_output: dict[str, Any]) -> dict[str, Any]:
    predictions = forecasting_output.get("predictions", [])
    return {
        "ticker": forecasting_output.get("ticker"),
        "model_version": forecasting_output.get("model_version"),
        "model_path": forecasting_output.get("model_path"),
        "holdout_metrics": forecasting_output.get("holdout_metrics", {}),
        "forecast_diagnostics": forecasting_output.get("forecast_diagnostics", {}),
        "prediction_count": len(predictions),
        "first_prediction": predictions[0] if predictions else None,
        "last_prediction": predictions[-1] if predictions else None,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from trend extractor.")
    return data


def _normalize_trend_factors(data: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    historical_pattern = str(data.get("historical_pattern") or fallback["historical_pattern"]).strip().lower()
    if historical_pattern not in {"improving", "stable", "degrading", "volatile", "insufficient", "mixed"}:
        historical_pattern = fallback["historical_pattern"]
    repeated_top_risks = data.get("repeated_top_risks")
    if not isinstance(repeated_top_risks, list):
        repeated_top_risks = fallback["repeated_top_risks"]
    normalized = dict(fallback)
    normalized.update({
        "source": "llm",
        "historical_pattern": historical_pattern,
        "repeated_top_risks": [str(item) for item in repeated_top_risks[:5]],
        "risk_persistence_score": round(clamp(safe_float(data.get("risk_persistence_score")), 0.0, 100.0), 2),
        "model_failure_signal": round(clamp(safe_float(data.get("model_failure_signal")), 0.0, 100.0), 2),
        "market_regime_shift_signal": round(clamp(safe_float(data.get("market_regime_shift_signal")), 0.0, 100.0), 2),
        "news_driven_signal": round(clamp(safe_float(data.get("news_driven_signal")), 0.0, 100.0), 2),
        "forecast_direction_error_signal": round(clamp(safe_float(data.get("forecast_direction_error_signal")), 0.0, 100.0), 2),
        "retrain_effectiveness": str(data.get("retrain_effectiveness") or fallback["retrain_effectiveness"]),
        "llm_confidence": round(clamp(safe_float(data.get("confidence", data.get("llm_confidence"))), 0.0, 1.0), 2),
        "rationale": str(data.get("rationale") or fallback["rationale"])[:600],
    })
    return normalized


def _classify_reporter_trend(score_trend: str, trend_factors: dict[str, Any]) -> str:
    model_signal = safe_float(trend_factors.get("model_failure_signal"))
    persistence = safe_float(trend_factors.get("risk_persistence_score"))
    market_signal = safe_float(trend_factors.get("market_regime_shift_signal"))
    news_signal = safe_float(trend_factors.get("news_driven_signal"))
    historical_pattern = str(trend_factors.get("historical_pattern", score_trend))
    if model_signal >= 70 and persistence >= 65:
        return "degrading_model"
    if max(market_signal, news_signal) >= 70 and model_signal < 55:
        return "external_risk"
    if historical_pattern in {"degrading", "volatile", "mixed"}:
        return historical_pattern
    return score_trend


def generate_report_node(state: ReporterAgentState) -> ReporterAgentState:
    """Generate and persist the final JSON and Markdown reports."""
    forecasting_output = state["forecasting_output"]
    evaluation_output = state["evaluation_output"]
    ticker = state.get("ticker") or forecasting_output["ticker"]
    run_date = state.get("run_date") or date.today().isoformat()
    cfg = state.get("agent_config") or config_manager.agent
    reporter_cfg = cfg.get("reporter", {})
    report_dir = Path(reporter_cfg.get("reports_dir", "artifacts/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    # Always use a single canonical filename. The final report after retrain overwrites the initial one.
    json_path = report_dir / f"{run_date}_{ticker}_report.json"
    md_path = report_dir / f"{run_date}_{ticker}_report.md"
    insight_summary = state.get("insight_summary") or _build_report_insight_summary(state)

    reporter_output = {
        "action": state["action"],
        "reason": state["reason"],
        "insight_summary": insight_summary,
        "trend_assessment": state["trend_assessment"],
        "trend_factors": state.get("trend_factors", {}),
        "is_retrain": bool(state.get("is_retrain", False)),
        "report_paths": {"json": str(json_path), "markdown": str(md_path)},
        "composite_score": state["composite_score"],
    }
    payload = {
        "ticker": ticker,
        "run_id": state.get("run_id"),
        "run_date": run_date,
        "forecasting": forecasting_output,
        "evaluation": evaluation_output,
        "reporter": reporter_output,
        "improvement": state.get("improvement_output"),
    }
    write_json_report(json_path, payload)

    llm_markdown = _generate_report_markdown_with_llm(state, payload)
    if llm_markdown:
        with open(md_path, "w") as f:
            f.write(llm_markdown.rstrip() + "\n")
    else:
        write_markdown_report(md_path, payload)

    state["insight_summary"] = insight_summary
    state["reporter_output"] = reporter_output
    return state


def _build_report_insight_summary(state: ReporterAgentState) -> str:
    evaluation = state["evaluation_output"]
    risk = evaluation.get("risk_breakdown", {})
    trend_factors = state.get("trend_factors", {})
    top_risks = sorted(risk.items(), key=lambda item: safe_float(item[1]), reverse=True)[:3]
    top_risk_text = ", ".join(f"{name}={safe_float(value):.1f}" for name, value in top_risks) or "none"
    source = trend_factors.get("source", "rules")
    rationale = str(trend_factors.get("rationale", "")).strip()
    if rationale:
        rationale = f" Trend evidence ({source}): {rationale}"
    return (
        f"Action={state['action']} with trust score {safe_float(evaluation.get('trust_score')):.2f} "
        f"and composite score {safe_float(state.get('composite_score')):.2f}. "
        f"Primary risks: {top_risk_text}. "
        f"Trend assessment={state.get('trend_assessment', 'unknown')}. "
        f"Reason: {state['reason']}."
        f"{rationale}"
    )


def _generate_report_markdown_with_llm(
    state: ReporterAgentState,
    payload: dict[str, Any],
) -> str:
    llm = state.get("llm")
    if not llm:
        return ""

    forecasting_output = payload["forecasting"]
    evaluation_output = payload["evaluation"]
    reporter_output = payload["reporter"]
    ticker = payload["ticker"]
    improvement = state.get("improvement_output") or {}

    forecast_table = _build_forecast_table(forecasting_output.get("predictions", []))
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
            run_date=payload.get("run_date", date.today().isoformat()),
            model_version=forecasting_output.get("model_version", "?"),
            action=reporter_output["action"],
            trust_score=f"{safe_float(evaluation_output.get('trust_score')):.2f}",
            composite_score=f"{reporter_output.get('composite_score', 0):.2f}",
            decision_band=evaluation_output.get("decision_band", ""),
            reason=reporter_output["reason"],
            trend_assessment=reporter_output.get("trend_assessment", "unknown"),
            trend_factors=json.dumps(reporter_output.get("trend_factors", {}), indent=2, default=str),
            holdout_metrics=holdout_str,
            forecast_diagnostics=json.dumps(
                forecasting_output.get("forecast_diagnostics", {}),
                indent=2,
                default=str,
            ),
            forecast_table=forecast_table,
            risk_breakdown=json.dumps(
                {k: round(float(v), 2) for k, v in evaluation_output.get("risk_breakdown", {}).items()},
                indent=2,
            ),
            news_context=evaluation_output.get("news_context", {}).get("summary", "N/A"),
            improvement_info=_summarize_improvement_info(improvement),
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response)).strip()
    except Exception as e:
        state["insight_summary"] = f"LLM markdown generation failed: {e}"
        return ""


def _summarize_improvement_info(improvement: dict[str, Any]) -> str:
    if improvement.get("is_retrain"):
        old_training_score = improvement.get("old_training_score")
        new_training_score = improvement.get("new_training_score")
        score_text = ""
        if old_training_score is not None and new_training_score is not None:
            score_text = (
                f" New training score {safe_float(new_training_score):.4f} vs old "
                f"{safe_float(old_training_score):.4f}."
            )
        elif improvement.get("old_composite_score") is not None and improvement.get("new_composite_score") is not None:
            score_text = (
                f" New composite {safe_float(improvement.get('new_composite_score')):.2f} vs old "
                f"{safe_float(improvement.get('old_composite_score')):.2f}."
            )
        return (
            f"Retrain attempted. Promoted: {improvement.get('promoted')}. "
            f"Reason: {improvement.get('reason', '')}. "
            f"New model version: {improvement.get('new_model_version')}."
            f"{score_text}"
        )
    if improvement.get("skip_retrain") is True or (improvement and not improvement.get("is_retrain")):
        return improvement.get("reason", "No retrain performed.")
    return "No retrain performed."


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


def _load_previous_markdown_reports(report_dir: str | Path, ticker: str, limit: int = 7) -> list[dict[str, str]]:
    report_dir = Path(report_dir)
    if not report_dir.exists():
        return []
    reports: list[dict[str, str]] = []
    for path in report_dir.glob(f"*_{ticker}*_report.md"):
        try:
            reports.append({
                "filename": path.name,
                "content": path.read_text(encoding="utf-8")[:8000],
            })
        except OSError:
            continue
    reports.sort(key=lambda item: item["filename"])
    return reports[-limit:]


def _rule_based_retrain_effectiveness(previous_reports: list[dict[str, Any]]) -> str:
    retrain_reports = [
        report for report in previous_reports
        if report.get("improvement") or report.get("reporter", {}).get("is_retrain")
    ]
    if not retrain_reports:
        return "no_retrain_history"
    improved = 0
    worsened = 0
    for report in retrain_reports:
        improvement = report.get("improvement") or {}
        old_score = improvement.get("old_training_score", improvement.get("old_composite_score"))
        new_score = improvement.get("new_training_score", improvement.get("new_composite_score"))
        if old_score is None or new_score is None:
            continue
        if "old_training_score" in improvement or "new_training_score" in improvement:
            is_improved = safe_float(new_score) < safe_float(old_score)
            is_worsened = safe_float(new_score) > safe_float(old_score)
        else:
            is_improved = safe_float(new_score) > safe_float(old_score)
            is_worsened = safe_float(new_score) < safe_float(old_score)
        if is_improved:
            improved += 1
        elif is_worsened:
            worsened += 1
    if improved and not worsened:
        return "improved"
    if worsened and not improved:
        return "worsened"
    if improved or worsened:
        return "mixed"
    return "no_retrain_history"


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


def plan_candidates_node(state: ImprovementAgentState) -> ImprovementAgentState:
    """Diagnose failure and generate retrain candidate configs."""
    diagnosis, skip_retrain, skip_reason = _diagnose_improvement_need(state)
    state["diagnosis"] = diagnosis
    state["skip_retrain"] = skip_retrain
    state["skip_reason"] = skip_reason

    if skip_retrain:
        state["candidates"] = []
        return state

    state["candidates"] = _generate_improvement_candidates(state)
    return state


def _diagnose_improvement_need(state: ImprovementAgentState) -> tuple[dict[str, Any], bool, str | None]:
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
        return (
            diagnosis,
            True,
            f"Primary cause is market volatility (recent_volatility_risk={risk.get('recent_volatility_risk', 0):.1f}), "
            "not model degradation. Retraining skipped."
        )
    return diagnosis, False, None


def _generate_improvement_candidates(state: ImprovementAgentState) -> list[dict[str, Any]]:
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

    return candidates


def retrain_candidates_node(state: ImprovementAgentState) -> ImprovementAgentState:
    """Train all candidates to temporary directories and score training holdout metrics.

    We intentionally do NOT commit candidates to permanent version directories
    here. Each candidate is trained into an isolated tmp_* folder inside
    artifacts_dir. select_best_candidate_node keeps only the best temporary
    candidate so the workflow can forecast and evaluate it before promotion.
    """
    # Short-circuit: plan_candidates_node may have decided retrain is not warranted
    if state.get("skip_retrain", False):
        state["candidate_results"] = []
        return state

    from src.forecasting.trainer import (
        train_quantile_models, save_quantile_models, compute_metrics,
        _select_point_quantile,
    )
    from src.preprocessing import preprocess_data

    ticker = state["ticker"]
    df_raw = state["df_raw"]
    preprocessing_config = state.get("preprocessing_config") or config_manager.preprocessing
    model_config = state.get("model_config") or config_manager.model
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

        candidate_results.append({
            "tmp_dir": str(tmp_dir),
            "metrics": test_metrics,
            "training_metrics": test_metrics,
            "training_score": _candidate_metric_score(test_metrics),
            "params": candidate,
            "models": models,
            "feature_columns": preprocessing_result["feature_columns"],
        })

    state["candidate_results"] = candidate_results
    return state


def select_best_candidate_node(state: ImprovementAgentState) -> ImprovementAgentState:
    """Pick the best temporary candidate and delete the losers."""
    import shutil

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
    old_reporter = state["reporter_output"]
    old_composite = float(old_reporter.get("composite_score", 0.0))
    old_metrics = state.get("forecasting_output", {}).get("holdout_metrics", {})
    old_training_score = _candidate_metric_score(old_metrics)
    candidate_results = state["candidate_results"]

    if not candidate_results:
        state["improvement_output"] = {
            "is_retrain": False,
            "promoted": False,
            "reason": "No candidates were trained.",
            "diagnosis": state.get("diagnosis"),
        }
        return state

    # Find the best candidate by training holdout metrics only. Promotion is
    # intentionally left to evaluation_workflow_node after a full rerun.
    best = min(candidate_results, key=lambda c: c["training_score"])

    # Delete losing tmp dirs; keep the winner for the workflow rerun.
    for candidate in candidate_results:
        if candidate is best:
            continue
        tmp = Path(candidate["tmp_dir"])
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    state["best_candidate"] = best
    state["improvement_output"] = {
        "is_retrain": True,
        "old_model_version": old_model_version,
        "candidate_model_path": best["tmp_dir"],
        "candidate_params": best["params"],
        "promoted": None,
        "promotion_decided": False,
        "reason": (
            f"Selected retrained candidate with training score {best['training_score']:.4f}; "
            "promotion will be decided after forecasting and evaluation."
        ),
        "old_composite_score": old_composite,
        "old_training_score": old_training_score,
        "new_training_score": best["training_score"],
        "old_training_metrics": old_metrics,
        "new_training_metrics": best["training_metrics"],
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


def _candidate_metric_score(metrics: dict[str, Any]) -> float:
    """Lower-is-better score for comparing retrained candidates during training."""
    mape = safe_float(metrics.get("MAPE"), default=1.0)
    rmse = safe_float(metrics.get("RMSE"), default=0.0)
    mae = safe_float(metrics.get("MAE"), default=0.0)
    # MAPE is the primary scale-free metric. RMSE/MAE are tiny tie-breakers.
    return mape + (1e-6 * rmse) + (1e-7 * mae)
