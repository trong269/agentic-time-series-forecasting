from xgboost import XGBRegressor
import joblib
from pathlib import Path


def train_xgb_quantile(X_train, y_train, quantile, xgb_params):
    """Train XGBRegressor with quantile regression objective.

    Args:
        X_train: Training features (pandas DataFrame or numpy array)
        y_train: Training target values
        quantile: Quantile value between 0 and 1 (e.g., 0.5 for median)
        xgb_params: Dictionary of XGBoost parameters

    Returns:
        Fitted XGBRegressor model
    """
    params = xgb_params.copy()
    params["objective"] = "reg:quantileerror"
    params["quantile_alpha"] = quantile

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    return model


def save_model(model, path):
    """Save model to a .pkl file using joblib.

    Args:
        model: Fitted model to save
        path: Path to save the model file (should end in .pkl)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load model from a .pkl file.

    Args:
        path: Path to the model file

    Returns:
        Loaded model
    """
    return joblib.load(path)