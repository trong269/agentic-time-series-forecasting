"""XGBoost-based forecasting module."""
from src.forecasting.trainer import train_xgboost_forecaster, compute_metrics
from src.forecasting.predictor import predict_with_intervals

__all__ = ["train_xgboost_forecaster", "predict_with_intervals", "compute_metrics"]
