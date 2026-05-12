"""Preprocessing functions for XGBoost training and prediction."""

from typing import Any

import pandas as pd

from src.utils.config_manager import config_manager

from .feature_functions import create_features, split_train_test, trim_dataframe
from .validator import handle_missing_values, validate_data


def preprocess_data(df: pd.DataFrame, config: dict) -> dict[str, Any]:
    """One-shot preprocessing: create features + trim + split.

    This is the core preprocessing function that agents can call directly
    with explicit config and DataFrame.

    Args:
        df: Input DataFrame with OHLCV columns and date.
        config: Full preprocessing config dict (from config_manager.preprocessing).

    Returns:
        Dictionary with X_train, X_test, y_train, y_test, feature_columns.
    """
    # Validate data
    is_valid, errors = validate_data(df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {errors}")

    df = handle_missing_values(df, fill_method="ffill")

    features_cfg = config.get("features", {})
    target_cfg = config.get("target", {})
    split_cfg = config.get("split", {})

    # Determine max lag and horizon
    price_lags = features_cfg.get("price_lags", [1, 7, 30])
    max_lag = max(price_lags)
    target_horizon = target_cfg.get("horizon", 1)
    test_days = split_cfg.get("test_days", 60)
    gap = split_cfg.get("gap", 0)

    # Create all features
    full_config = {"features": features_cfg, "target": target_cfg}
    df_features = create_features(df, full_config)

    # Get feature columns (exclude all-NaN columns)
    exclude_cols = ["date", "target"]
    all_feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    feature_columns = [col for col in all_feature_cols if df_features[col].notna().any()]

    # Trim
    df_trimmed = trim_dataframe(df_features, feature_columns, max_lag, target_horizon)

    # Split
    split_result = split_train_test(
        df_trimmed,
        feature_columns,
        target_column="target",
        test_days=test_days,
        gap=gap,
    )

    return {
        "X_train": split_result["X_train"],
        "X_test": split_result["X_test"],
        "y_train": split_result["y_train"],
        "y_test": split_result["y_test"],
        "feature_columns": feature_columns,
        "train_size": split_result["train_size"],
        "test_size": split_result["test_size"],
    }


def preprocess_for_training(ticker: str | None = None, config: dict | None = None) -> dict[str, Any]:
    """Preprocess stock data for XGBoost training (high-level wrapper).

    This function fetches data from DB and runs pipeline.
    Agents should use preprocess_data() directly for explicit control.

    Args:
        ticker: Stock ticker symbol. If None, uses config default.
        config: Optional config override. If None, uses config_manager.preprocessing.

    Returns:
        Dictionary with X_train, X_test, y_train, y_test, feature_columns.
    """
    from src.ingestion import get_stock_data

    cfg = config or config_manager.preprocessing
    ticker = ticker or cfg.get("ticker", "NVDA")

    df = get_stock_data(ticker)
    return preprocess_data(df, cfg)


def preprocess_for_prediction(ticker: str, config: dict | None = None) -> dict[str, Any]:
    """Prepare data for prediction - fetch recent data and create features.

    This function fetches the most recent stock data and prepares it for the
    prediction pipeline. Returns both the raw DataFrame and the last row's features.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA").
        config: Optional config override. If None, uses config_manager.preprocessing.

    Returns:
        Dictionary with:
        - df_raw: Raw DataFrame with available stock history
        - last_features: DataFrame with single row of features for the last available day
        - last_date: The date of the last available data
        - close_list: Full close-price history for recursive prediction
    """
    from src.ingestion import get_stock_data

    cfg = config or config_manager.preprocessing

    df_raw = get_stock_data(ticker)
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # Validate and handle missing
    is_valid, errors = validate_data(df_raw)
    if not is_valid:
        raise ValueError(f"Data validation failed: {errors}")

    df_raw = handle_missing_values(df_raw, fill_method="ffill")

    # Get configs
    features_cfg = cfg.get("features", {})
    target_cfg = cfg.get("target", {})

    full_config = {"features": features_cfg, "target": target_cfg}

    # Create features
    df_features = create_features(df_raw, full_config)

    # Get feature columns
    exclude_cols = ["date", "target"]
    feature_columns = [
        col for col in df_features.columns
        if col not in exclude_cols and df_features[col].notna().any()
    ]

    # Get last row with valid features
    df_valid = df_features.dropna(subset=feature_columns)
    if df_valid.empty:
        raise ValueError("No valid features after preprocessing")

    last_row = df_valid.iloc[[-1]]

    # Keep full close history so EMA-based features match training preprocessing.
    close_list = df_raw["close"].tolist()

    return {
        "df_raw": df_raw,
        "last_features": last_row[feature_columns],
        "last_date": pd.to_datetime(df_raw["date"].iloc[-1]).date(),
        "close_list": close_list,
        "feature_columns": feature_columns,
    }
