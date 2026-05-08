"""Technical indicators for stock data analysis."""

import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index).

    Args:
        prices: Series of close prices.
        period: RSI period (default 14).

    Returns:
        Series with RSI values (0-100).
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of close prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        Tuple of (MACD_line, signal_line, histogram).
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        prices: Series of close prices.
        window: Moving average window (default 20).
        num_std: Number of standard deviations (default 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    middle_band = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std()

    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    return upper_band, middle_band, lower_band


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range (ATR).

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        period: ATR period (default 14).

    Returns:
        Series with ATR values.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()

    return atr