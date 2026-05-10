"""Main preprocessing pipeline for XGBoost."""

from typing import Any

import pandas as pd

from src.utils.config_manager import config_manager

from .feature_engineering import create_all_features
from .validator import handle_missing_values, validate_data


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for XGBoost training."""

    def __init__(self, ticker: str, config: dict | None = None):
        """Initialize preprocessing pipeline.

        Args:
            ticker: Stock ticker symbol (e.g., "NVDA").
            config: Optional config override. If None, uses config_manager.
        """
        self.ticker = ticker
        self.config = config or config_manager.preprocessing

        # Feature columns will be set during fit
        self.feature_columns: list[str] = []
        self.target_column = "target"

    def _get_feature_config(self) -> dict[str, Any]:
        """Get feature engineering configuration."""
        features = self.config.get("features", {})
        return {
            "price_lags": features.get("price_lags", [1, 7, 30]),
            "ma_windows": features.get("ma_windows", [7, 21, 50]),
            "return_periods": features.get("return_periods", [21]),
            "volatility_windows": features.get("volatility_windows", [21]),
            "rsi_period": features.get("rsi_period"),
            "macd_fast": features.get("macd_fast", 12),
            "macd_slow": features.get("macd_slow", 26),
            "macd_signal": features.get("macd_signal", 9),
            "macd_include_histogram": features.get("macd_include_histogram", False),
            "bb_window": features.get("bb_window", 20),
            "bb_std": features.get("bb_std", 2.0),
            "bb_include_bands": features.get("bb_include_bands", False),
            "include_calendar": features.get("include_calendar", True),
            "calendar_features": features.get("calendar_features", ["quarter", "month", "week_of_year"]),
            "include_close_to_ma": features.get("include_close_to_ma", True),
            "close_to_ma_windows": features.get("close_to_ma_windows", [50]),
            "include_high_low_ratio": features.get("include_high_low_ratio", False),
            # Note: volume_ma_windows and include_atr removed - cannot be used in recursive prediction
        }

    def _get_target_config(self) -> dict[str, Any]:
        """Get target variable configuration."""
        target = self.config.get("target", {})
        return {
            "horizon": target.get("horizon", 7),
            "type": target.get("type", "return"),
        }

    def _get_split_config(self) -> dict[str, Any]:
        """Get train/test split configuration."""
        split = self.config.get("split", {})
        return {
            "test_days": split.get("test_days", 60),
            "gap": split.get("gap", 0),
        }

    def fit_transform(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit on training data and transform.

        Args:
            df: Input DataFrame with OHLCV columns and date.

        Returns:
            Dictionary with keys:
            - X_train, X_test: Feature DataFrames
            - y_train, y_test: Target Series
            - feature_columns: List of feature column names
            - train_dates, test_dates: Date ranges for each split
        """
        # Validate data
        is_valid, errors = validate_data(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")

        # Handle missing values
        df = handle_missing_values(df, fill_method="ffill")

        # Get configs
        feature_config = self._get_feature_config()
        target_config = self._get_target_config()
        split_config = self._get_split_config()

        # Determine max lag and horizon for trimming
        max_price_lag = max(feature_config["price_lags"])
        # Volume has no lags now, only MA windows
        max_lag = max_price_lag
        target_horizon = target_config["horizon"]

        # Create all features
        df_features = create_all_features(
            df,
            price_lags=feature_config["price_lags"],
            ma_windows=feature_config["ma_windows"],
            return_periods=feature_config["return_periods"],
            volatility_windows=feature_config["volatility_windows"],
            rsi_period=feature_config["rsi_period"],
            macd_fast=feature_config["macd_fast"],
            macd_slow=feature_config["macd_slow"],
            macd_signal=feature_config["macd_signal"],
            macd_include_histogram=feature_config["macd_include_histogram"],
            bb_window=feature_config["bb_window"],
            bb_std=feature_config["bb_std"],
            bb_include_bands=feature_config["bb_include_bands"],
            include_calendar=feature_config["include_calendar"],
            calendar_features=feature_config["calendar_features"],
            include_close_to_ma=feature_config["include_close_to_ma"],
            close_to_ma_windows=feature_config["close_to_ma_windows"],
            include_high_low_ratio=feature_config["include_high_low_ratio"],
            target_horizon=target_horizon,
            target_type=target_config["type"],
        )

        # Store feature columns, excluding columns that are all NaN
        exclude_cols = ["date", "target"]
        all_feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        # Remove columns that are all NaN (like adj_close for recent data)
        self.feature_columns = [col for col in all_feature_cols if df_features[col].notna().any()]

        # Trim initial rows with NaN from lags and final rows with NaN from target
        # Keep rows from index max_lag onwards, and exclude last target_horizon rows
        trim_start = max_lag
        trim_end = target_horizon

        df_trimmed = df_features.iloc[trim_start:-trim_end] if trim_end > 0 else df_features.iloc[trim_start:]
        df_trimmed = df_trimmed.dropna(subset=self.feature_columns)

        # Time-based split
        test_days = split_config["test_days"]
        gap = split_config["gap"]

        # Split data chronologically
        if gap > 0:
            train_end = len(df_trimmed) - test_days - gap
            X_train = df_trimmed.iloc[:train_end][self.feature_columns]
            y_train = df_trimmed.iloc[:train_end][self.target_column]
            X_test = df_trimmed.iloc[train_end + gap :][self.feature_columns]
            y_test = df_trimmed.iloc[train_end + gap :][self.target_column]
        else:
            train_end = len(df_trimmed) - test_days
            X_train = df_trimmed.iloc[:train_end][self.feature_columns]
            y_train = df_trimmed.iloc[:train_end][self.target_column]
            X_test = df_trimmed.iloc[train_end:][self.feature_columns]
            y_test = df_trimmed.iloc[train_end:][self.target_column]

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_columns": self.feature_columns,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline.

        Args:
            df: Input DataFrame with same structure as fit().

        Returns:
            DataFrame with only feature columns.
        """
        if not self.feature_columns:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")

        # Handle missing values
        df = handle_missing_values(df, fill_method="ffill")

        # Get configs
        feature_config = self._get_feature_config()
        target_config = self._get_target_config()

        # Create features
        df_features = create_all_features(
            df,
            price_lags=feature_config["price_lags"],
            ma_windows=feature_config["ma_windows"],
            return_periods=feature_config["return_periods"],
            volatility_windows=feature_config["volatility_windows"],
            rsi_period=feature_config["rsi_period"],
            macd_fast=feature_config["macd_fast"],
            macd_slow=feature_config["macd_slow"],
            macd_signal=feature_config["macd_signal"],
            macd_include_histogram=feature_config["macd_include_histogram"],
            bb_window=feature_config["bb_window"],
            bb_std=feature_config["bb_std"],
            bb_include_bands=feature_config["bb_include_bands"],
            include_calendar=feature_config["include_calendar"],
            calendar_features=feature_config["calendar_features"],
            include_close_to_ma=feature_config["include_close_to_ma"],
            close_to_ma_windows=feature_config["close_to_ma_windows"],
            include_high_low_ratio=feature_config["include_high_low_ratio"],
            target_horizon=target_config["horizon"],
            target_type=target_config["type"],
        )

        # Return only feature columns
        return df_features[self.feature_columns]