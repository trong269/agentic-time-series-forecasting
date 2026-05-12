"""XGBoost-based forecasting module."""
from src.forecasting.trainer import (
    train_xgboost_forecaster,
    train_quantile_models,
    compute_metrics,
    save_quantile_models,
)
from src.forecasting.predictor import (
    predict_with_intervals,
    predict_single_day,
    build_prediction_features,
    compute_holdout_metrics,
    load_models,
)

__all__ = [
    # Trainer
    "train_xgboost_forecaster",
    "train_quantile_models",
    "compute_metrics",
    "save_quantile_models",
    # Predictor
    "predict_with_intervals",
    "predict_single_day",
    "build_prediction_features",
    "compute_holdout_metrics",
    "load_models",
]
