"""XGBoost predictor with recursive prediction and confidence intervals."""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config_manager import config_manager
from src.utils.logger import logger
from src.forecasting.models.xgboost_model import load_model
from src.preprocessing.technical import calculate_bollinger_bands, calculate_macd


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


def _select_point_quantile(values: dict[float, Any]) -> float:
    """Select the quantile used as the point forecast."""
    if 0.50 in values:
        return 0.50
    return min(values, key=lambda q: abs(q - 0.50))


def _monotonize_quantiles(values: dict[float, float]) -> dict[float, float]:
    """Enforce non-decreasing predictions as quantiles increase."""
    quantiles = sorted(values)
    sorted_predictions = sorted(values[q] for q in quantiles)
    return dict(zip(quantiles, sorted_predictions))


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


# =============================================================================
# EXTRACTED STANDALONE FUNCTIONS (for agent use)
# =============================================================================


def compute_holdout_metrics(y_actual: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute holdout metrics from actual and predicted values.

    Pure function - no config reading, no data fetching.

    Args:
        y_actual: Actual target values.
        y_pred: Predicted target values.

    Returns:
        Dict with MAE, RMSE, MAPE.
    """
    mae = float(np.mean(np.abs(y_pred - y_actual)))
    rmse = float(np.sqrt(np.mean((y_pred - y_actual) ** 2)))
    denominator = np.where(np.abs(y_actual) == 0, np.nan, np.abs(y_actual))
    mape = float(np.nanmean(np.abs(y_pred - y_actual) / denominator))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def build_prediction_features(
    close_list: list,
    feature_config: dict,
    feature_date: date,
) -> pd.DataFrame:
    """Build single-row features for a forecast anchor date from close prices.

    The returned row mirrors training-time features for the latest known or
    recursively predicted close. With a one-step target, that row predicts the
    next business day's close.

    Args:
        close_list: Full close-price history plus any recursive predictions.
        feature_config: Feature config dict with keys:
            - price_lags: list[int]
            - ma_windows: list[int]
            - return_periods: list[int]
            - volatility_windows: list[int]
            - macd_fast, macd_slow, macd_signal: int
            - bb_window: int
            - bb_std: float
            - close_to_ma_windows: list[int]
        feature_date: Date of the latest close represented by close_list[-1].

    Returns:
        DataFrame with single row of features.
    """
    price_lags = feature_config.get("price_lags", [1, 7, 30])
    ma_windows = feature_config.get("ma_windows", [7, 21, 50])
    return_periods = feature_config.get("return_periods", [21])
    volatility_windows = feature_config.get("volatility_windows", [21])
    macd_fast = feature_config.get("macd_fast", 12)
    macd_slow = feature_config.get("macd_slow", 26)
    macd_signal = feature_config.get("macd_signal", 9)
    bb_window = feature_config.get("bb_window", 20)
    bb_std = feature_config.get("bb_std", 2.0)
    close_to_ma_windows = feature_config.get("close_to_ma_windows", [50])

    close_arr = np.asarray(close_list, dtype=float)
    close_series = pd.Series(close_arr)

    features = {}

    # Price lag features
    for lag in price_lags:
        if len(close_arr) >= lag + 1:
            features[f"close_lag_{lag}"] = close_arr[-lag - 1]  # -1 because we need the value BEFORE prediction day
        else:
            features[f"close_lag_{lag}"] = np.nan

    # Return features
    for period in return_periods:
        if len(close_arr) >= period + 1:
            ret = (close_arr[-1] - close_arr[-period - 1]) / close_arr[-period - 1]
            features[f"return_{period}d"] = ret
        else:
            features[f"return_{period}d"] = np.nan

    # Moving averages include the anchor close, matching training preprocessing.
    for window in ma_windows:
        if len(close_arr) >= window:
            features[f"MA_{window}"] = float(close_arr[-window:].mean())
        else:
            features[f"MA_{window}"] = np.nan

    # Close-to-MA ratio
    for window in close_to_ma_windows:
        col = f"close_to_MA_{window}"
        if len(close_arr) >= window:
            ma = float(close_arr[-window:].mean())
            features[col] = (close_arr[-1] - ma) / ma
        else:
            features[col] = np.nan

    # Volatility
    for window in volatility_windows:
        if len(close_arr) >= window:
            features[f"volatility_{window}d"] = float(close_series.iloc[-window:].std())
        else:
            features[f"volatility_{window}d"] = np.nan

    # MACD
    if len(close_arr) >= macd_slow:
        _, signal_line, _ = calculate_macd(
            close_series, fast=macd_fast, slow=macd_slow, signal=macd_signal
        )
        features["MACD_signal"] = float(signal_line.iloc[-1])
    else:
        features["MACD_signal"] = np.nan

    # Bollinger Bands position
    if len(close_arr) >= bb_window:
        bb_upper_series, _, bb_lower_series = calculate_bollinger_bands(
            close_series, window=bb_window, num_std=bb_std
        )
        bb_upper = float(bb_upper_series.iloc[-1])
        bb_lower = float(bb_lower_series.iloc[-1])
        if bb_upper > bb_lower:
            features["BB_position"] = (close_arr[-1] - bb_lower) / (bb_upper - bb_lower)
        else:
            features["BB_position"] = 0.5
    else:
        features["BB_position"] = np.nan

    # Calendar features
    features["quarter"] = (feature_date.month - 1) // 3 + 1
    features["month"] = feature_date.month
    features["week_of_year"] = feature_date.isocalendar()[1]

    # Return as single-row DataFrame
    return pd.DataFrame([features])


def predict_single_day(
    models: dict,
    close_list: list,
    feature_config: dict,
    feature_date: date,
) -> dict[float, float]:
    """Predict one day using loaded models.

    Args:
        models: Dict mapping quantiles to fitted XGBRegressor models.
        close_list: Full close-price history plus any recursive predictions.
        feature_config: Feature config dict (same as build_prediction_features).
        feature_date: Date of the latest close represented by close_list[-1].

    Returns:
        Dict mapping quantiles to predicted values.
    """
    # Build features
    features = build_prediction_features(close_list, feature_config, feature_date)

    # Align features with model's expected column order
    point_quantile = _select_point_quantile(models)
    median_model = models[point_quantile]
    if hasattr(median_model, "feature_names_in_"):
        expected_features = median_model.feature_names_in_
        for col in expected_features:
            if col not in features.columns:
                features[col] = np.nan
        features = features[expected_features]

    # Predict with each quantile model
    pred_values = {}
    for q, model in models.items():
        pred_values[q] = float(model.predict(features)[0])

    return _monotonize_quantiles(pred_values)


def load_models(
    ticker: str,
    artifacts_dir: Path | None = None,
    quantiles: list[float] | None = None,
    version_dir: Path | None = None,
) -> dict:
    """Load quantile models from a version directory or the latest version.

    Args:
        ticker: Stock ticker symbol.
        artifacts_dir: Base directory containing versioned model folders.
                     If None, uses config_manager.model.get("artifacts_dir").
        version_dir: Explicit directory such as artifacts/models/ver_20.

    Returns:
        Dict mapping quantiles to loaded models. Empty dict if no models found.
    """
    if artifacts_dir is None:
        model_cfg = config_manager.model
        artifacts_dir = Path(model_cfg.get("artifacts_dir", "artifacts/models/"))

    quantiles = quantiles or config_manager.model.get("quantiles", [0.025, 0.10, 0.50, 0.90, 0.975])

    if version_dir is None and artifacts_dir.name.startswith("ver_"):
        version_dir = artifacts_dir
    elif version_dir is None:
        version_dir = _get_latest_version_dir(artifacts_dir)
    if version_dir is None:
        return {}

    models = {}
    for q in quantiles:
        model_path = version_dir / f"{ticker}_q{_format_quantile(q)}.pkl"
        if model_path.exists():
            models[q] = load_model(model_path)

    return models


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================


def predict_with_intervals(
    ticker: str,
    horizon: int = 7,
    models: dict | None = None,
    df_raw: pd.DataFrame | None = None,
    feature_config: dict | None = None,
    preprocessing_config: dict | None = None,
    model_config: dict | None = None,
) -> dict[str, Any]:
    """Generate recursive predictions with confidence intervals.

    This function accepts explicit parameters - no hidden config reading or data fetching
    (except for fallback cases where models/df_raw are None).

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA").
        horizon: Number of days to forecast ahead (default: 7).
        models: Pre-loaded quantile models dict {quantile: model}. If None, auto-loads from disk.
        df_raw: Pre-fetched raw stock DataFrame. If None, fetches from DB.
        feature_config: Feature config dict. If None, uses preprocessing_config["features"].
        preprocessing_config: Full preprocessing config for holdout metrics and fallback feature config.
        model_config: Full model config for artifact loading and quantile selection.

    Returns:
        Dict containing:
        - predictions: List of daily prediction dicts with date, point_forecast, confidence_80, confidence_95
        - holdout_metrics: Dict with MAE, RMSE, MAPE for last 7 known days
    """
    model_cfg = model_config or config_manager.model
    preprocessing_cfg = preprocessing_config or config_manager.preprocessing
    artifacts_dir = Path(model_cfg.get("artifacts_dir", "artifacts/models/"))
    quantiles = model_cfg.get("quantiles", [0.025, 0.10, 0.50, 0.90, 0.975])

    # Load models if not provided
    if models is None:
        models = load_models(ticker, artifacts_dir, quantiles)

    if not models:
        logger.warning("No models found, triggering retraining")
        from src.forecasting.trainer import train_xgboost_forecaster
        train_result = train_xgboost_forecaster(
            ticker,
            df=df_raw,
            config={**preprocessing_cfg, "model": model_cfg},
        )
        models = train_result["models"]
        feature_config = feature_config or preprocessing_cfg.get("features", {})
    else:
        logger.info(f"Using pre-loaded models")

    # Get feature config
    if feature_config is None:
        feature_config = preprocessing_cfg.get("features", {})

    # Fetch raw data if not provided
    if df_raw is None:
        from src.ingestion import get_stock_data
        df_raw = get_stock_data(ticker)
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # Compute holdout metrics: predict last N known days vs actuals
    holdout_metrics = _compute_holdout_metrics(
        ticker,
        models,
        df_raw=df_raw,
        preprocessing_config=preprocessing_cfg,
        holdout_size=horizon,
    )

    # Last actual or recursively predicted close date represented by close_list[-1].
    last_date = pd.to_datetime(df_raw["date"].iloc[-1]).date()

    # Keep full close history so EMA-based features match training preprocessing.
    close_list = df_raw["close"].tolist()

    predictions = []

    for _ in range(horizon):
        feature_date = last_date
        forecast_date = _get_next_business_day(feature_date)

        # Predict this day
        pred_values = predict_single_day(models, close_list, feature_config, feature_date)

        # Extract point forecast and intervals
        point_quantile = _select_point_quantile(pred_values)
        point = pred_values.get(point_quantile)
        lower_80 = pred_values.get(0.10, None)
        upper_80 = pred_values.get(0.90, None)
        lower_95 = pred_values.get(0.025, None)
        upper_95 = pred_values.get(0.975, None)

        predictions.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "point_forecast": round(float(point), 2) if point is not None else None,
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
        if point is not None:
            close_list.append(point)
            last_date = forecast_date

    logger.info(
        f"Predictions generated for {ticker} | Horizon: {horizon} days | "
        f"MAE: {holdout_metrics['MAE']:.2f}, MAPE: {holdout_metrics['MAPE']:.2%}"
    )

    return {"ticker": ticker, "predictions": predictions, "holdout_metrics": holdout_metrics}


def _compute_holdout_metrics(
    ticker: str,
    models: dict,
    df_raw: pd.DataFrame | None = None,
    preprocessing_config: dict | None = None,
    holdout_size: int | None = None,
) -> dict[str, float]:
    """Compute holdout metrics on last N known days.

    Internal function that uses preprocess_for_training for backward compatibility.

    Args:
        ticker: Stock ticker symbol.
        models: Dict mapping quantiles to fitted models.

    Returns:
        Dict with MAE, RMSE, MAPE for median predictions.
    """
    from src.preprocessing import preprocess_data, preprocess_for_training

    cfg = preprocessing_config or config_manager.preprocessing
    holdout_size = holdout_size or cfg.get("target", {}).get("horizon", 1)

    # Get preprocessing result to access train/test split
    if df_raw is None:
        result = preprocess_for_training(ticker, cfg)
    else:
        result = preprocess_data(df_raw, cfg)
    X_test = result["X_test"]
    y_test = result["y_test"]

    # Last N rows of test set are holdout
    y_holdout_actual = y_test.iloc[-holdout_size:].values
    X_holdout = X_test.iloc[-holdout_size:]

    # Predict on holdout using median model (non-recursive for metrics computation)
    point_quantile = _select_point_quantile(models)
    y_holdout_pred = models[point_quantile].predict(X_holdout)

    # Compute metrics using the pure function
    return compute_holdout_metrics(y_holdout_actual, y_holdout_pred)
