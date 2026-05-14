"""Run the daily multi-agent forecasting pipeline."""

from __future__ import annotations

import argparse
import json

from src.workflow import DailyForecastingWorkflow


def main() -> dict:
    parser = argparse.ArgumentParser(description="Run daily forecasting pipeline")
    parser.add_argument("ticker", nargs="?", default="NVDA")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--fetch-latest", action="store_true")
    args = parser.parse_args()

    workflow = DailyForecastingWorkflow()
    state = workflow.run(args.ticker, horizon=args.horizon, fetch_latest=args.fetch_latest)
    summary = {
        "ticker": state["ticker"],
        "run_id": state["run_id"],
        "run_date": state["run_date"],
        "model_version": state["model_version"],
        "model_path": state["model_path"],
        "trust_score": state["evaluation_output"]["trust_score"],
        "decision_band": state["evaluation_output"]["decision_band"],
        "action": state["reporter_output"]["action"],
        "composite_score": state["reporter_output"]["composite_score"],
        "report_paths": state["reporter_output"]["report_paths"],
        "improvement": state.get("improvement_output"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
