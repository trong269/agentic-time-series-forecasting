"""Stock data preprocessing module for XGBoost.

This module provides preprocessing functionality to prepare stock data
for XGBoost training and inference.
"""

from .calendar import create_calendar_features
from .feature_engineering import (
    create_all_features,
    create_lag_features,
    create_moving_average_features,
    create_return_features,
    create_target,
    create_technical_features,
    create_volume_features,
)
from .pipeline import PreprocessingPipeline
from .technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
)
from .validator import get_data_summary, handle_missing_values, validate_data

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    # Feature engineering
    "create_all_features",
    "create_lag_features",
    "create_return_features",
    "create_moving_average_features",
    "create_volume_features",
    "create_technical_features",
    "create_target",
    # Technical indicators
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_atr",
    # Calendar
    "create_calendar_features",
    # Validation
    "validate_data",
    "handle_missing_values",
    "get_data_summary",
]


def preprocess_for_training(ticker: str, config: dict | None = None) -> dict:
    """Preprocess stock data for XGBoost training.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA").
        config: Optional config override.

    Returns:
        Dictionary with X_train, X_test, y_train, y_test, feature_columns.
    """
    from src.ingestion import get_stock_data

    # Get raw data from ingestion
    df = get_stock_data(ticker)

    # Create and run pipeline
    pipeline = PreprocessingPipeline(ticker, config)
    result = pipeline.fit_transform(df)

    return result


def preprocess_for_prediction(ticker: str, days: int = 60) -> dict:
    """Preprocess recent data for prediction.

    Args:
        ticker: Stock ticker symbol.
        days: Number of recent days to fetch.

    Returns:
        Dictionary with recent data and features.
    """
    from src.ingestion import get_stock_data

    # Get recent data
    df = get_stock_data(ticker)

    # Take only last `days` rows
    df_recent = df.tail(days).copy()

    # Create features
    from .feature_engineering import create_all_features

    df_features = create_all_features(df_recent)

    return {
        "data": df_features,
        "feature_columns": [col for col in df_features.columns if col not in ["date", "target"]],
    }