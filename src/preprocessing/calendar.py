"""Calendar-based features for stock data."""

import pandas as pd


def create_calendar_features(
    df: pd.DataFrame,
    features: list[str] | None = None
) -> pd.DataFrame:
    """Create calendar-based features from date column.

    Args:
        df: Input DataFrame with 'date' column.
        features: List of features to include. Options: quarter, month, week_of_year,
                  day_of_week, day_of_month, is_month_start, is_month_end,
                  is_quarter_start, is_year_start.
                  Default: [quarter, month, week_of_year]

    Returns:
        DataFrame with added calendar features.
    """
    result = df.copy()

    # Ensure date is datetime
    date_series = result["date"]
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series)
        result["date"] = date_series

    default_features = ["quarter", "month", "week_of_year"]
    features = features or default_features

    available_features = {
        "day_of_week": result["date"].dt.dayofweek,
        "day_of_month": result["date"].dt.day,
        "month": result["date"].dt.month,
        "quarter": result["date"].dt.quarter,
        "is_month_start": result["date"].dt.is_month_start.astype(int),
        "is_month_end": result["date"].dt.is_month_end.astype(int),
        "is_quarter_start": result["date"].dt.is_quarter_start.astype(int),
        "is_year_start": result["date"].dt.is_year_start.astype(int),
        "week_of_year": result["date"].dt.isocalendar().week,
    }

    for feat in features:
        if feat in available_features:
            result[feat] = available_features[feat]

    return result