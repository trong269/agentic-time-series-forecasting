"""Stock data preprocessing module for XGBoost.

This module provides preprocessing functionality to prepare stock data
for XGBoost training and inference.
"""

from .calendar import create_calendar_features
from .feature_functions import create_features, split_train_test, trim_dataframe
from .pipeline import preprocess_data, preprocess_for_prediction
from .technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
)
from .validator import get_data_summary, handle_missing_values, validate_data

__all__ = [
    # Pipeline functions (agents use these)
    "preprocess_data",
    "preprocess_for_prediction",
    # Feature functions (extracted for agent use)
    "create_features",
    "split_train_test",
    "trim_dataframe",
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


def preprocess_for_training(ticker: str | None = None, config: dict | None = None) -> dict:
    """Preprocess stock data for XGBoost training.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA").
        config: Optional config override.

    Returns:
        Dictionary with X_train, X_test, y_train, y_test, feature_columns.
    """
    from .pipeline import preprocess_for_training as _preprocess_for_training_impl
    return _preprocess_for_training_impl(ticker, config)
