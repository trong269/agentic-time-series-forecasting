"""XGBoost forecaster trainer with holdout evaluation."""
from typing import Any
import json
import numpy as np
import pandas as pd
from pathlib import Path
import re
from datetime import datetime

from src.preprocessing import preprocess_for_training
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


def compute_metrics(y_pred: np.ndarray, y_actual: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE from predictions and actuals."""
    mae = float(np.mean(np.abs(y_pred - y_actual)))
    rmse = float(np.sqrt(np.mean((y_pred - y_actual) ** 2)))
    mape = float(np.mean(np.abs(y_pred - y_actual) / np.abs(y_actual)))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def train_xgboost_forecaster(ticker: str) -> dict[str, Any]:
    """Train 5 XGBoost quantile models and evaluate on full test set.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA").

    Returns:
        Dictionary containing:
        - ticker: The stock ticker
        - models: Dict mapping quantiles to fitted models
        - test_metrics: Dict with MAE, RMSE, MAPE for median predictions on full test set
        - feature_columns: List of feature column names used
        - version: Model version number
        - version_dir: Path to versioned model directory
    """
    # Load config
    model_cfg = config_manager.model
    xgb_params = model_cfg.get("xgb_params", {})
    artifacts_dir = Path(model_cfg.get("artifacts_dir", "artifacts/models/"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Determine version for this training run
    version = get_next_model_version(artifacts_dir)
    version_dir = artifacts_dir / f"ver_{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    quantiles = model_cfg.get("quantiles", [0.025, 0.10, 0.50, 0.90, 0.975])

    # Preprocess data
    result = preprocess_for_training(ticker)
    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]

    logger.info(f"Training model for {ticker} | Train: {len(X_train)}, Test: {len(X_test)}")

    # Train on X_train only (proper holdout - X_test is truly unseen)
    # Evaluate on full X_test for comparison with other versions
    models = {}
    for q in quantiles:
        model = train_xgb_quantile(X_train, y_train, q, xgb_params)
        models[q] = model
        # Save each model to versioned directory
        model_path = version_dir / f"{ticker}_q{_format_quantile(q)}.pkl"
        save_model(model, model_path)

    # Evaluate on full test set (non-recursive direct prediction)
    y_test_pred = models[0.50].predict(X_test)
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
    median_model = models[0.50]
    feature_importance = {}
    if hasattr(median_model, "feature_importances_"):
        for fname, fimp in zip(median_model.feature_names_in_, median_model.feature_importances_):
            feature_importance[fname] = float(fimp)

    # Save metadata.json - only drift-detection relevant info, not redundant with config
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
        "train_size": len(X_train),
        "test_size": len(X_test),
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