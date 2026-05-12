"""XGBoost forecaster trainer with holdout evaluation."""

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import re
from datetime import datetime

from src.utils.config_manager import config_manager
from src.utils.logger import logger
from src.forecasting.models.xgboost_model import train_xgb_quantile, save_model


def get_latest_model_version(base_dir: Path) -> int | None:
    """Find the latest model version number.

    Args:
        base_dir: Base directory containing versioned model folders.

    Returns:
        Latest version number (int) or None if no versions exist.
    """
    if not base_dir.exists():
        return None
    versions = []
    for d in base_dir.iterdir():
        if d.is_dir() and re.match(r"^ver_\d+$", d.name):
            try:
                versions.append(int(d.name[4:]))
            except ValueError:
                pass
    return max(versions) if versions else None


def get_next_model_version(base_dir: Path) -> int:
    """Get the next version number for a new model.

    Args:
        base_dir: Base directory containing versioned model folders.

    Returns:
        Next version number (1 if no previous versions exist).
    """
    latest = get_latest_model_version(base_dir)
    return (latest + 1) if latest is not None else 1


def _format_quantile(q: float) -> str:
    """Format quantile for model filename (e.g., 0.025 -> '0_025')."""
    return str(q).replace(".", "_")


def _select_point_quantile(models: dict[float, Any]) -> float:
    """Select the quantile used as the point forecast."""
    if 0.50 in models:
        return 0.50
    return min(models, key=lambda q: abs(q - 0.50))


# =============================================================================
# EXTRACTED STANDALONE FUNCTIONS (for agent use)
# =============================================================================


def compute_metrics(y_pred: np.ndarray, y_actual: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE from predictions and actuals.

    Pure function - no config reading, no data fetching.

    Args:
        y_pred: Predicted values.
        y_actual: Actual target values.

    Returns:
        Dict with MAE, RMSE, MAPE.
    """
    mae = float(np.mean(np.abs(y_pred - y_actual)))
    rmse = float(np.sqrt(np.mean((y_pred - y_actual) ** 2)))
    denominator = np.where(np.abs(y_actual) == 0, np.nan, np.abs(y_actual))
    mape = float(np.nanmean(np.abs(y_pred - y_actual) / denominator))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: list[float],
    xgb_params: dict,
) -> dict[float, Any]:
    """Train XGBoost quantile regression models.

    Pure function - no config reading, no data fetching.
    Agents can call this directly with explicit X_train, y_train.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        quantiles: List of quantile values to train (e.g., [0.025, 0.10, 0.50, 0.90, 0.975]).
        xgb_params: XGBoost parameters dict (e.g., {"n_estimators": 100, "max_depth": 6}).

    Returns:
        Dict mapping each quantile to its fitted XGBRegressor model.
    """
    models = {}
    for q in quantiles:
        model = train_xgb_quantile(X_train, y_train, q, xgb_params)
        models[q] = model
    return models


def save_quantile_models(
    models: dict,
    ticker: str,
    version_dir: Path,
) -> dict[float, Path]:
    """Save trained quantile models to disk.

    Args:
        models: Dict mapping quantiles to fitted models.
        ticker: Stock ticker symbol for filename.
        version_dir: Directory to save models (will be created if needed).

    Returns:
        Dict of saved model paths {quantile: path}.
    """
    version_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for q, model in models.items():
        model_path = version_dir / f"{ticker}_q{_format_quantile(q)}.pkl"
        save_model(model, model_path)
        saved_paths[q] = model_path
    return saved_paths


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def train_xgboost_forecaster(
    ticker: str,
    df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Train 5 XGBoost quantile models and evaluate on test set.

    This function accepts explicit parameters - no hidden config reading or data fetching
    (except for fallback cases where df is None).

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA").
        df: Pre-fetched stock DataFrame. If None, fetches from DB via preprocess_for_training.
        config: Full config dict. If None, uses config_manager.preprocessing + config_manager.model.

    Returns:
        Dictionary containing:
        - ticker: The stock ticker
        - models: Dict mapping quantiles to fitted models
        - test_metrics: Dict with MAE, RMSE, MAPE for median predictions on test set
        - feature_columns: List of feature column names used
        - version: Model version number
        - version_dir: Path to versioned model directory
    """
    # Load configs
    if config is None:
        preprocessing_cfg = config_manager.preprocessing
        model_cfg = config_manager.model
    else:
        preprocessing_cfg = config
        model_cfg = config.get("model", config_manager.model)

    xgb_params = model_cfg.get("xgb_params", {})
    artifacts_dir = Path(model_cfg.get("artifacts_dir", "artifacts/models/"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    quantiles = model_cfg.get("quantiles", [0.025, 0.10, 0.50, 0.90, 0.975])

    # Determine version for this training run
    version = get_next_model_version(artifacts_dir)
    version_dir = artifacts_dir / f"ver_{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess data - either use provided df or fetch from DB
    if df is None:
        from src.preprocessing import preprocess_for_training
        result = preprocess_for_training(ticker, preprocessing_cfg)
    else:
        from src.preprocessing import preprocess_data
        result = preprocess_data(df, preprocessing_cfg)

    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]

    logger.info(f"Training model for {ticker} | Train: {len(X_train)}, Test: {len(X_test)}")

    # Train quantile models
    models = train_quantile_models(X_train, y_train, quantiles, xgb_params)

    # Save models to versioned directory
    save_quantile_models(models, ticker, version_dir)

    # Evaluate on test set
    point_quantile = _select_point_quantile(models)
    y_test_pred = models[point_quantile].predict(X_test)
    test_metrics = compute_metrics(y_test_pred, y_test.values)

    logger.info(
        f"Training complete | Version: {version} | "
        f"MAE: {test_metrics['MAE']:.2f}, RMSE: {test_metrics['RMSE']:.2f}, MAPE: {test_metrics['MAPE']:.2%}"
    )

    # Compute metrics for all quantile models (for drift detection reference)
    all_quantile_metrics = {}
    for q, model in models.items():
        y_pred_q = model.predict(X_test)
        all_quantile_metrics[str(q)] = compute_metrics(y_pred_q, y_test.values)

    # Get feature importance from median model
    median_model = models[point_quantile]
    feature_importance = {}
    if hasattr(median_model, "feature_importances_"):
        for fname, fimp in zip(median_model.feature_names_in_, median_model.feature_importances_):
            feature_importance[fname] = float(fimp)

    # Save metadata.json
    metadata = {
        "version": version,
        "ticker": ticker,
        "trained_at": datetime.now().isoformat(),
        "test_metrics": {
            "MAE": test_metrics["MAE"],
            "RMSE": test_metrics["RMSE"],
            "MAPE": test_metrics["MAPE"],
        },
        "all_quantile_metrics": all_quantile_metrics,
        "feature_importance": feature_importance,
        "feature_columns": result["feature_columns"],
        "preprocessing_config": preprocessing_cfg,
        "model_config": model_cfg,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "point_quantile": point_quantile,
    }

    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "ticker": ticker,
        "models": models,
        "test_metrics": test_metrics,
        "feature_columns": result["feature_columns"],
        "version": version,
        "version_dir": version_dir,
    }
