"""Manual model retraining entry point."""
import json
import sys
from src.forecasting import train_xgboost_forecaster


def main(ticker: str = "NVDA"):
    result = train_xgboost_forecaster(ticker)
    output = {
        "ticker": result["ticker"],
        "test_metrics": result["test_metrics"],
        "version": result["version"],
    }
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    main(ticker)
