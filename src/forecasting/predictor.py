"""XGBoost predictor with recursive prediction and confidence intervals."""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion import get_stock_data
from src.preprocessing import preprocess_for_training
from src.utils.config_manager import config_manager
from src.utils.logger import logger
from src.forecasting.models.xgboost_model import load_model


def _get_next_business_day(start_date: date) -> date:
    """Get next business day (skip weekends)."""
    next_day = start_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day


def _ema(prices: np.ndarray, period: int) -> float:
    """Calculate EMA for a price array."""
    alpha = 2 / (period + 1)
    ema = float(prices[0])
    for price in prices[1:]:
        ema = alpha * float(price) + (1 - alpha) * ema
    return ema


def _format_quantile(q: float) -> str:
    """Format quantile for model filename (e.g., 0.025 -> '0_025')."""
    return str(q).replace(".", "_")


def _get_latest_version_dir(artifacts_dir: Path) -> Path | None:
    """Find the directory for the latest model version.

    Args:
        artifacts_dir: Base directory containing versioned model folders.

    Returns:
        Path to latest version directory, or None if no versions exist.
    """
    import re
    if not artifacts_dir.exists():
        return None
    versions = []
    for d in artifacts_dir.iterdir():
        if d.is_dir() and re.match(r"^ver_\d+$", d.name):
            try:
                versions.append((int(d.name[4:]), d))
            except ValueError:
                pass
    if not versions:
        return None
    return max(versions, key=lambda x: x[0])[1]


def predict_with_intervals(ticker: str, horizon: int = 7) -> dict:
    """Generate 7-day recursive predictions with confidence intervals.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA").
        horizon: Number of days to forecast ahead.

    Returns:
        Dict containing:
        - predictions: List of daily prediction dicts with date, point_forecast, confidence_80, confidence_95
        - holdout_metrics: Dict with MAE, RMSE, MAPE for last 7 known days (non-recursive direct prediction)
    """
    model_cfg = config_manager.model
    artifacts_dir = Path(model_cfg.get("artifacts_dir", "artifacts/models/"))
    quantiles = model_cfg.get("quantiles", [0.025, 0.10, 0.50, 0.90, 0.975])

    # Find latest version directory
    version_dir = _get_latest_version_dir(artifacts_dir)

    # Load models from latest version
    models = {}
    if version_dir is not None:
        for q in quantiles:
            model_path = version_dir / f"{ticker}_q{_format_quantile(q)}.pkl"
            if model_path.exists():
                models[q] = load_model(model_path)
        # Check if all quantiles loaded successfully
        if len(models) != len(quantiles):
            models = {}  # Partial load, trigger retrain

    if not models:
        logger.warning("No models found, triggering retraining")
        # Fallback: retrain if no models exist
        from src.forecasting.trainer import train_xgboost_forecaster
        train_result = train_xgboost_forecaster(ticker)
        models = train_result["models"]
    else:
        logger.info(f"Loaded models from {version_dir.name}")

    # Compute holdout metrics: predict last N known days vs actuals
    holdout_metrics = _compute_holdout_metrics(ticker, models)

    # Get raw stock data for feature building
    df_raw = get_stock_data(ticker)
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # Last actual date in data
    last_date = pd.to_datetime(df_raw["date"].iloc[-1]).date()

    # Feature config
    prep_cfg = config_manager.preprocessing
    price_lags = prep_cfg.get("features", {}).get("price_lags", [1, 7, 30])
    ma_windows = prep_cfg.get("features", {}).get("ma_windows", [7, 21, 50])
    return_periods = prep_cfg.get("features", {}).get("return_periods", [21])
    volatility_windows = prep_cfg.get("features", {}).get("volatility_windows", [21])
    macd_fast = prep_cfg.get("features", {}).get("macd_fast", 12)
    macd_slow = prep_cfg.get("features", {}).get("macd_slow", 26)
    macd_signal = prep_cfg.get("features", {}).get("macd_signal", 9)
    bb_window = prep_cfg.get("features", {}).get("bb_window", 20)
    bb_std = prep_cfg.get("features", {}).get("bb_std", 2.0)
    close_to_ma_windows = prep_cfg.get("features", {}).get("close_to_ma_windows", [50])

    # Maintain rolling close list for recursive feature updates
    close_list = df_raw["close"].iloc[-60:].tolist()

    predictions = []

    for _ in range(horizon):
        forecast_date = _get_next_business_day(last_date)
        last_date = forecast_date

        # Build features for this day using recent close values
        features = _build_daily_features(
            close_list=close_list,
            price_lags=price_lags,
            ma_windows=ma_windows,
            return_periods=return_periods,
            volatility_windows=volatility_windows,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_window=bb_window,
            bb_std=bb_std,
            close_to_ma_windows=close_to_ma_windows,
            forecast_date=forecast_date,
        )

        # Ensure features match model's expected columns
        model = models[0.50]  # Use median model for feature names
        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
            for col in expected_features:
                if col not in features.columns:
                    features[col] = np.nan
            features = features[expected_features]

        # Predict with each quantile model
        pred_values = {}
        for q, model in models.items():
            pred_values[q] = float(model.predict(features)[0])

        # Extract point forecast and intervals
        point = pred_values.get(0.50, pred_values.get(0.10, None))
        lower_80 = pred_values.get(0.10, None)
        upper_80 = pred_values.get(0.90, None)
        lower_95 = pred_values.get(0.025, None)
        upper_95 = pred_values.get(0.975, None)

        predictions.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "point_forecast": round(float(point), 2),
            "confidence_80": {
                "lower": round(float(lower_80), 2) if lower_80 is not None else None,
                "upper": round(float(upper_80), 2) if upper_80 is not None else None,
            },
            "confidence_95": {
                "lower": round(float(lower_95), 2) if lower_95 is not None else None,
                "upper": round(float(upper_95), 2) if upper_95 is not None else None,
            },
        })

        # Update close list with prediction for recursive feature building
        close_list.append(point)

        # Keep list manageable
        if len(close_list) > 100:
            close_list.pop(0)

    logger.info(
        f"Predictions generated for {ticker} | Horizon: {horizon} days | "
        f"MAE: {holdout_metrics['MAE']:.2f}, MAPE: {holdout_metrics['MAPE']:.2%}"
    )

    return {"ticker": ticker, "predictions": predictions, "holdout_metrics": holdout_metrics}


def _compute_holdout_metrics(ticker: str, models: dict) -> dict[str, float]:
    """Compute holdout metrics on last N known days (configurable via config).

    Args:
        ticker: Stock ticker symbol.
        models: Dict mapping quantiles to fitted models.

    Returns:
        Dict with MAE, RMSE, MAPE for median predictions.
    """
    from src.preprocessing import preprocess_for_training

    horizon = config_manager.preprocessing.get("target", {}).get("horizon", 7)

    # Get preprocessing result to access train/test split
    result = preprocess_for_training(ticker)
    X_test = result["X_test"]
    y_test = result["y_test"]

    # Last 'horizon' days of test set are holdout
    y_holdout_actual = y_test.iloc[-horizon:].values
    X_holdout = X_test.iloc[-horizon:]

    # Predict on holdout using median model (non-recursive for metrics computation)
    y_holdout_pred = models[0.50].predict(X_holdout)

    # Compute metrics
    mae = float(np.mean(np.abs(y_holdout_pred - y_holdout_actual)))
    rmse = float(np.sqrt(np.mean((y_holdout_pred - y_holdout_actual) ** 2)))
    mape = float(np.mean(np.abs(y_holdout_pred - y_holdout_actual) / np.abs(y_holdout_actual)))

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _build_daily_features(
    close_list: list,
    price_lags: list[int],
    ma_windows: list[int],
    return_periods: list[int],
    volatility_windows: list[int],
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    bb_window: int,
    bb_std: float,
    close_to_ma_windows: list[int],
    forecast_date: date,
) -> pd.DataFrame:
    """Build feature row for a single prediction day from recent close values.

    Args:
        close_list: List of recent close prices (actual + predicted so far).
        forecast_date: The date being forecasted.

    Returns:
        DataFrame with single row of features.
    """
    close_arr = np.array(close_list)

    features = {}

    # Price lag features
    for lag in price_lags:
        if len(close_arr) >= lag:
            features[f"close_lag_{lag}"] = close_arr[-lag - 1]  # -1 because we need the value BEFORE prediction day
        else:
            features[f"close_lag_{lag}"] = np.nan

    # Return features
    for period in return_periods:
        if len(close_arr) >= period + 1:
            ret = (close_arr[-2] - close_arr[-period - 2]) / close_arr[-period - 2]
            features[f"return_{period}d"] = ret
        else:
            features[f"return_{period}d"] = np.nan

    # Moving averages (based on closes before prediction day)
    closes_before_pred = close_arr[:-1] if len(close_arr) > 1 else close_arr
    for window in ma_windows:
        if len(closes_before_pred) >= window:
            features[f"MA_{window}"] = np.mean(closes_before_pred[-window:])
        else:
            features[f"MA_{window}"] = np.nan

    # Close-to-MA ratio
    if 50 in close_to_ma_windows and len(closes_before_pred) >= 50:
        ma_50 = np.mean(closes_before_pred[-50:])
        features["close_to_MA_50"] = (closes_before_pred[-1] - ma_50) / ma_50
    else:
        features["close_to_MA_50"] = np.nan

    # Volatility
    for window in volatility_windows:
        if len(closes_before_pred) >= window:
            features[f"volatility_{window}d"] = np.std(closes_before_pred[-window:])
        else:
            features[f"volatility_{window}d"] = np.nan

    # MACD
    if len(closes_before_pred) >= macd_slow:
        ema_fast = _ema(closes_before_pred, macd_fast)
        ema_slow = _ema(closes_before_pred, macd_slow)
        macd_line = ema_fast - ema_slow
        # Signal line approximation using exponential smoothing
        signal_alpha = 2 / (macd_signal + 1)
        features["MACD_signal"] = macd_line * signal_alpha + features.get("MACD_signal", macd_line) * (1 - signal_alpha)
    else:
        features["MACD_signal"] = np.nan

    # Bollinger Bands position
    if len(closes_before_pred) >= bb_window:
        bb_mean = np.mean(closes_before_pred[-bb_window:])
        bb_std_val = np.std(closes_before_pred[-bb_window:])
        bb_upper = bb_mean + bb_std * bb_std_val
        bb_lower = bb_mean - bb_std * bb_std_val
        if bb_upper > bb_lower:
            features["BB_position"] = (closes_before_pred[-1] - bb_lower) / (bb_upper - bb_lower)
        else:
            features["BB_position"] = 0.5
    else:
        features["BB_position"] = np.nan

    # Calendar features
    features["quarter"] = (forecast_date.month - 1) // 3 + 1
    features["month"] = forecast_date.month
    features["week_of_year"] = forecast_date.isocalendar()[1]

    # Return as single-row DataFrame
    return pd.DataFrame([features])