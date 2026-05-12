"""Feature functions for standalone use by agents.

These functions accept explicit config dicts and DataFrames - no hidden config reading.
Logic inlined from feature_engineering.py to eliminate the dependency.
"""

from typing import Any

import pandas as pd

from .technical import (
    calculate_bollinger_bands,
    calculate_macd,
)
from .calendar import create_calendar_features


# =============================================================================
# INDIVIDUAL FEATURE CREATION FUNCTIONS
# =============================================================================


def create_lag_features(df: pd.DataFrame, lags: list[int], column: str = "close") -> pd.DataFrame:
    """Create lag features for a column."""
    result = df.copy()
    for lag in lags:
        result[f"{column}_lag_{lag}"] = result[column].shift(lag)
    return result


def create_return_features(df: pd.DataFrame, periods: list[int] = [21]) -> pd.DataFrame:
    """Create return features for various periods."""
    result = df.copy()
    for period in periods:
        result[f"return_{period}d"] = result["close"].pct_change(period)
    return result


def create_moving_average_features(
    df: pd.DataFrame,
    windows: list[int] = [7, 21, 50],
    include_close_to_ma: bool = True,
    close_to_ma_windows: list[int] = [50]
) -> pd.DataFrame:
    """Create moving average features."""
    result = df.copy()
    for window in windows:
        ma = result["close"].rolling(window=window, min_periods=window).mean()
        result[f"MA_{window}"] = ma
        if include_close_to_ma and window in close_to_ma_windows:
            result[f"close_to_MA_{window}"] = (result["close"] - ma) / ma
    return result


def create_technical_features(
    df: pd.DataFrame,
    rsi_period: int | None = None,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    volatility_windows: list[int] = [21],
    include_high_low_ratio: bool = False,
    bb_include_bands: bool = False,
    macd_include_histogram: bool = False,
) -> pd.DataFrame:
    """Create technical indicator features."""
    result = df.copy()

    if rsi_period is not None:
        from .technical import calculate_rsi
        result["RSI"] = calculate_rsi(result["close"], period=rsi_period)

    _, signal_line, histogram = calculate_macd(
        result["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
    )
    result["MACD_signal"] = signal_line
    if macd_include_histogram:
        result["MACD_histogram"] = histogram

    bb_upper, _, bb_lower = calculate_bollinger_bands(
        result["close"], window=bb_window, num_std=bb_std
    )
    result["BB_position"] = (result["close"] - bb_lower) / (bb_upper - bb_lower)
    if bb_include_bands:
        result["BB_upper"] = bb_upper
        result["BB_lower"] = bb_lower

    for window in volatility_windows:
        result[f"volatility_{window}d"] = result["close"].rolling(
            window=window, min_periods=window
        ).std()

    if include_high_low_ratio:
        result["high_low_ratio"] = (result["close"] - result["low"]) / (
            result["high"] - result["low"]
        )

    return result


def create_target(
    df: pd.DataFrame,
    horizon: int = 1,
    target_type: str = "close"
) -> pd.DataFrame:
    """Create forward-looking target variable."""
    result = df.copy()

    if target_type == "close":
        result["target"] = result["close"].shift(-horizon)
    elif target_type == "return":
        future_close = result["close"].shift(-horizon)
        result["target"] = (future_close - result["close"]) / result["close"]
    elif target_type == "direction":
        future_close = result["close"].shift(-horizon)
        result["target"] = (future_close > result["close"]).astype(int)

    return result


def create_all_features(
    df: pd.DataFrame,
    price_lags: list[int] = [1, 7, 30],
    ma_windows: list[int] = [7, 21, 50],
    return_periods: list[int] = [21],
    volatility_windows: list[int] = [21],
    rsi_period: int | None = None,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    include_calendar: bool = True,
    calendar_features: list[str] | None = None,
    include_close_to_ma: bool = True,
    close_to_ma_windows: list[int] = [50],
    include_high_low_ratio: bool = False,
    bb_include_bands: bool = False,
    macd_include_histogram: bool = False,
    target_horizon: int = 1,
    target_type: str = "close"
) -> pd.DataFrame:
    """Create all features from raw stock data."""
    result = df.copy()

    result = create_lag_features(result, price_lags, column="close")
    result = create_return_features(result, periods=return_periods)
    result = create_moving_average_features(
        result,
        windows=ma_windows,
        include_close_to_ma=include_close_to_ma,
        close_to_ma_windows=close_to_ma_windows
    )
    result = create_technical_features(
        result,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        bb_window=bb_window,
        bb_std=bb_std,
        volatility_windows=volatility_windows,
        include_high_low_ratio=include_high_low_ratio,
        bb_include_bands=bb_include_bands,
        macd_include_histogram=macd_include_histogram,
    )

    if include_calendar:
        result = create_calendar_features(result, features=calendar_features)

    result = create_target(result, horizon=target_horizon, target_type=target_type)

    # Drop raw OHLCV columns
    for col in ["open", "high", "low", "close", "volume", "adj_close"]:
        if col in result.columns:
            result = result.drop(columns=[col])

    return result


# =============================================================================
# MAIN ENTRY POINTS FOR AGENTS
# =============================================================================


def create_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create all features from raw OHLCV data based on config dict.

    Args:
        df: Input DataFrame with OHLCV columns and date.
        config: Feature config dict with keys:
            - price_lags: list[int]
            - ma_windows: list[int]
            - return_periods: list[int]
            - volatility_windows: list[int]
            - macd_fast, macd_slow, macd_signal: int
            - bb_window, bb_std: int/float
            - include_calendar: bool
            - calendar_features: list[str]
            - include_close_to_ma: bool
            - close_to_ma_windows: list[int]
            - target_horizon: int
            - target_type: str ("close", "return", "direction")

    Returns:
        DataFrame with all features + target column.
    """
    features = config.get("features", {})
    target = config.get("target", {})

    return create_all_features(
        df,
        price_lags=features.get("price_lags", [1, 7, 30]),
        ma_windows=features.get("ma_windows", [7, 21, 50]),
        return_periods=features.get("return_periods", [21]),
        volatility_windows=features.get("volatility_windows", [21]),
        rsi_period=features.get("rsi_period"),
        macd_fast=features.get("macd_fast", 12),
        macd_slow=features.get("macd_slow", 26),
        macd_signal=features.get("macd_signal", 9),
        macd_include_histogram=features.get("macd_include_histogram", False),
        bb_window=features.get("bb_window", 20),
        bb_std=features.get("bb_std", 2.0),
        bb_include_bands=features.get("bb_include_bands", False),
        include_calendar=features.get("include_calendar", True),
        calendar_features=features.get("calendar_features", ["quarter", "month", "week_of_year"]),
        include_close_to_ma=features.get("include_close_to_ma", True),
        close_to_ma_windows=features.get("close_to_ma_windows", [50]),
        include_high_low_ratio=features.get("include_high_low_ratio", False),
        target_horizon=target.get("horizon", 1),
        target_type=target.get("type", "close"),
    )


def split_train_test(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target",
    test_days: int = 60,
    gap: int = 0,
) -> dict[str, Any]:
    """Time-based train/test split.

    Args:
        df: DataFrame with features and target.
        feature_columns: List of feature column names.
        target_column: Name of target column.
        test_days: Number of days for test set.
        gap: Gap days between train and test.

    Returns:
        dict with X_train, X_test, y_train, y_test, train_size, test_size.
    """
    if gap > 0:
        train_end = len(df) - test_days - gap
        X_train = df.iloc[:train_end][feature_columns]
        y_train = df.iloc[:train_end][target_column]
        X_test = df.iloc[train_end + gap :][feature_columns]
        y_test = df.iloc[train_end + gap :][target_column]
    else:
        train_end = len(df) - test_days
        X_train = df.iloc[:train_end][feature_columns]
        y_train = df.iloc[:train_end][target_column]
        X_test = df.iloc[train_end:][feature_columns]
        y_test = df.iloc[train_end:][target_column]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def trim_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_lag: int,
    horizon: int,
) -> pd.DataFrame:
    """Remove rows with NaN from lags (start) and target (end).

    Args:
        df: DataFrame with features and target.
        feature_columns: List of feature column names.
        max_lag: Maximum lag period (rows to trim from start).
        horizon: Target horizon (rows to trim from end).

    Returns:
        Trimmed DataFrame with NaN rows removed.
    """
    trim_start = max_lag
    trim_end = horizon

    df_trimmed = df.iloc[trim_start:-trim_end] if trim_end > 0 else df.iloc[trim_start:]
    df_trimmed = df_trimmed.dropna(subset=feature_columns)

    return df_trimmed
