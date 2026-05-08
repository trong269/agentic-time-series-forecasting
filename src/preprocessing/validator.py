"""Data validation utilities for preprocessing."""

import pandas as pd


def validate_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate data quality.

    Args:
        df: Input DataFrame to validate.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []

    # Check required columns
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check for empty DataFrame
    if df.empty:
        errors.append("DataFrame is empty")

    # Check for NaN values in critical columns
    critical_columns = ["close", "volume"]
    for col in critical_columns:
        if col in df.columns and df[col].isna().all():
            errors.append(f"Column '{col}' has all NaN values")

    # Check date monotonicity
    if "date" in df.columns:
        if not pd.to_datetime(df["date"]).is_monotonic_increasing:
            errors.append("Date column is not monotonically increasing")

    # Check for future dates
    if "date" in df.columns:
        latest_date = pd.to_datetime(df["date"]).max()
        if latest_date > pd.Timestamp.now():
            errors.append(f"Data contains future dates: {latest_date}")

    is_valid = len(errors) == 0
    return is_valid, errors


def handle_missing_values(df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """Handle missing values in stock data.

    Args:
        df: Input DataFrame.
        fill_method: Method for filling NaN values. Options:
            - "ffill": Forward fill (last known value)
            - "bfill": Backward fill (next known value)
            - "drop": Drop rows with NaN

    Returns:
        DataFrame with handled missing values.
    """
    result = df.copy()

    # Forward fill for stock data (appropriate for non-trading days)
    if fill_method == "ffill":
        result = result.ffill()
    elif fill_method == "bfill":
        result = result.bfill()
    elif fill_method == "drop":
        result = result.dropna()

    return result


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics of the data.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": {
            "start": str(df["date"].min()) if "date" in df.columns else None,
            "end": str(df["date"].max()) if "date" in df.columns else None,
        },
        "missing_values": df.isna().sum().to_dict(),
        "numeric_stats": df.describe().to_dict(),
    }

    return summary