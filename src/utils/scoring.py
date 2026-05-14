"""Deterministic scoring helpers shared by evaluator, reporter, and improver."""

from __future__ import annotations

from typing import Any


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def risk_from_ratio(ratio: float, bad_at: float) -> float:
    """Map a non-negative ratio to 0..100 risk."""
    if bad_at <= 0:
        return 100.0
    return clamp((max(0.0, float(ratio)) / bad_at) * 100.0)


def normalized_mape_risk(mape: float) -> float:
    return risk_from_ratio(mape, 0.10)


def normalized_rmse_pct_risk(rmse: float, latest_close: float) -> float:
    if latest_close <= 0:
        return 100.0
    return risk_from_ratio(rmse / latest_close, 0.10)


def composite_score(
    trust_score: float,
    risk_breakdown: dict[str, float],
) -> float:
    """Promotion/report score from trust score and interval calibration quality.

    trust_score already encodes all 6 risk dimensions (MAPE, RMSE, interval width,
    volatility, trend alignment, news). Adding them again here causes double-counting.
    We keep interval_width as an independent calibration quality signal (20%) since
    it measures model honesty (not overclaiming precision), separate from accuracy.
    """
    interval_quality = 100.0 - clamp(risk_breakdown.get("interval_width_risk", 100.0))
    return round(clamp(
        0.80 * clamp(trust_score)
        + 0.20 * interval_quality
    ), 2)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
