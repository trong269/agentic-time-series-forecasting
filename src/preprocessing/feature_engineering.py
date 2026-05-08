"""Feature engineering for time series data."""

from typing import Any

import pandas as pd

from .technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
)


def create_lag_features(df: pd.DataFrame, lags: list[int], column: str = "close") -> pd.DataFrame:
    """Create lag features for a column.

    Args:
        df: Input DataFrame.
        lags: List of lag periods, e.g. [1, 5, 21].
        column: Column to create lags for.

    Returns:
        DataFrame with added lag columns.
    """
    result = df.copy()
    for lag in lags:
        result[f"{column}_lag_{lag}"] = result[column].shift(lag)
    return result


def create_return_features(df: pd.DataFrame, periods: list[int] = [1, 5, 7, 21]) -> pd.DataFrame:
    """Create return features for various periods.

    Args:
        df: Input DataFrame with 'close' column.
        periods: List of periods for returns.

    Returns:
        DataFrame with added return columns.
    """
    result = df.copy()
    for period in periods:
        result[f"return_{period}d"] = result["close"].pct_change(period)
    return result


def create_moving_average_features(
    df: pd.DataFrame,
    windows: list[int] = [7, 21, 50]
) -> pd.DataFrame:
    """Create moving average features.

    Args:
        df: Input DataFrame with 'close' column.
        windows: List of window sizes.

    Returns:
        DataFrame with added MA and price-to-MA features.
    """
    result = df.copy()
    for window in windows:
        ma = result["close"].rolling(window=window, min_periods=window).mean()
        result[f"MA_{window}"] = ma
        result[f"close_to_MA_{window}"] = (result["close"] - ma) / ma

    return result


def create_volume_features(
    df: pd.DataFrame,
    lags: list[int] = [1, 5],
    ma_windows: list[int] = [7, 21]
) -> pd.DataFrame:
    """Create volume-based features.

    Args:
        df: Input DataFrame with 'volume' column.
        lags: List of volume lag periods.
        ma_windows: List of volume MA windows.

    Returns:
        DataFrame with added volume features.
    """
    result = df.copy()

    # Volume lags
    for lag in lags:
        result[f"volume_lag_{lag}"] = result["volume"].shift(lag)

    # Volume moving averages
    for window in ma_windows:
        result[f"volume_MA_{window}"] = result["volume"].rolling(
            window=window, min_periods=window
        ).mean()

    # Volume ratio
    result["volume_ratio"] = result["volume"] / result["volume_MA_7"]

    return result


def create_technical_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    volatility_windows: list[int] = [7, 21]
) -> pd.DataFrame:
    """Create technical indicator features.

    Args:
        df: Input DataFrame with OHLCV columns.
        rsi_period: RSI period.
        macd_fast, macd_slow, macd_signal: MACD parameters.
        bb_window, bb_std: Bollinger Bands parameters.
        atr_period: ATR period.
        volatility_windows: Windows for volatility calculation.

    Returns:
        DataFrame with added technical features.
    """
    result = df.copy()

    # RSI
    result["RSI"] = calculate_rsi(result["close"], period=rsi_period)

    # MACD
    macd_line, signal_line, histogram = calculate_macd(
        result["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
    )
    result["MACD"] = macd_line
    result["MACD_signal"] = signal_line
    result["MACD_histogram"] = histogram

    # Bollinger Bands position
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        result["close"], window=bb_window, num_std=bb_std
    )
    result["BB_position"] = (result["close"] - bb_lower) / (bb_upper - bb_lower)
    result["BB_upper"] = bb_upper
    result["BB_lower"] = bb_lower

    # ATR
    result["ATR"] = calculate_atr(
        result["high"], result["low"], result["close"], period=atr_period
    )

    # Volatility (rolling std)
    for window in volatility_windows:
        result[f"volatility_{window}d"] = result["close"].rolling(
            window=window, min_periods=window
        ).std()

    # High-low ratio
    result["high_low_ratio"] = (result["close"] - result["low"]) / (
        result["high"] - result["low"]
    )

    return result


def create_target(
    df: pd.DataFrame,
    horizon: int = 7,
    target_type: str = "return"
) -> pd.DataFrame:
    """Create forward-looking target variable.

    Args:
        df: Input DataFrame with 'close' column.
        horizon: Number of days ahead for prediction.
        target_type: "return" for regression, "direction" for classification.

    Returns:
        DataFrame with added target column.
    """
    result = df.copy()

    if target_type == "return":
        # 7-day forward return
        future_close = result["close"].shift(-horizon)
        result["target"] = (future_close - result["close"]) / result["close"]
    elif target_type == "direction":
        # 1 if price goes up, 0 if down
        future_close = result["close"].shift(-horizon)
        result["target"] = (future_close > result["close"]).astype(int)

    return result


def create_all_features(
    df: pd.DataFrame,
    price_lags: list[int] = [1, 2, 3, 5, 7, 14, 21, 30],
    volume_lags: list[int] = [1, 5],
    ma_windows: list[int] = [7, 21, 50],
    return_periods: list[int] = [1, 5, 7, 21],
    volatility_windows: list[int] = [7, 21],
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    include_calendar: bool = True,
    target_horizon: int = 7,
    target_type: str = "return"
) -> pd.DataFrame:
    """Create all features from raw stock data.

    Args:
        df: Input DataFrame with OHLCV and date columns.
        price_lags: Lag periods for price features.
        volume_lags: Lag periods for volume features.
        ma_windows: Windows for moving averages.
        return_periods: Periods for return calculations.
        volatility_windows: Windows for volatility.
        rsi_period: RSI period.
        macd_fast, macd_slow, macd_signal: MACD parameters.
        bb_window, bb_std: Bollinger parameters.
        include_calendar: Whether to add calendar features.
        target_horizon: Forward days for target.
        target_type: "return" or "direction".

    Returns:
        DataFrame with all features.
    """
    result = df.copy()

    # Price lag features
    result = create_lag_features(result, price_lags, column="close")

    # Return features
    result = create_return_features(result, periods=return_periods)

    # Moving average features
    result = create_moving_average_features(result, windows=ma_windows)

    # Volume features
    result = create_volume_features(result, lags=volume_lags)

    # Technical indicators
    result = create_technical_features(
        result,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        bb_window=bb_window,
        bb_std=bb_std,
        volatility_windows=volatility_windows,
    )

    # Calendar features
    if include_calendar:
        from .calendar import create_calendar_features

        result = create_calendar_features(result)

    # Target variable
    result = create_target(result, horizon=target_horizon, target_type=target_type)

    return result