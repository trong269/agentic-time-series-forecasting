"""Calendar-based features for stock data."""

import pandas as pd


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features from date column.

    Args:
        df: Input DataFrame with 'date' column.

    Returns:
        DataFrame with added calendar features.
    """
    result = df.copy()

    # Ensure date is datetime
    date_series = result["date"]
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series)
        result["date"] = date_series

    # Day of week (0=Monday, 4=Friday for stock market)
    result["day_of_week"] = result["date"].dt.dayofweek

    # Day of month
    result["day_of_month"] = result["date"].dt.day

    # Month
    result["month"] = result["date"].dt.month

    # Quarter
    result["quarter"] = result["date"].dt.quarter

    # Is month start/end
    result["is_month_start"] = result["date"].dt.is_month_start.astype(int)
    result["is_month_end"] = result["date"].dt.is_month_end.astype(int)

    # Is quarter start
    result["is_quarter_start"] = result["date"].dt.is_quarter_start.astype(int)

    # Is year start
    result["is_year_start"] = result["date"].dt.is_year_start.astype(int)

    # Week of year
    result["week_of_year"] = result["date"].dt.isocalendar().week

    return result