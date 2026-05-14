"""Report serialization helpers for daily forecasting runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json_report(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return str(path)


def write_markdown_report(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    forecasting = payload.get("forecasting", {})
    evaluation = payload.get("evaluation", {})
    reporter = payload.get("reporter", {})
    predictions = forecasting.get("predictions", [])
    risks = evaluation.get("risk_breakdown", {})

    lines = [
        f"# {payload.get('ticker', forecasting.get('ticker', 'UNKNOWN'))} Forecast Report",
        "",
        f"- Run date: {payload.get('run_date', '')}",
        f"- Model version: {forecasting.get('model_version', '')}",
        f"- Action: {reporter.get('action', '')}",
        f"- Trust score: {evaluation.get('trust_score', 0):.2f}",
        f"- Composite score: {reporter.get('composite_score', 0):.2f}",
        f"- Decision: {evaluation.get('decision_band', '')}",
        f"- Reason: {reporter.get('reason') or evaluation.get('reason', '')}",
        "",
        "## Holdout Metrics",
        "",
    ]
    metrics = forecasting.get("holdout_metrics", {})
    lines.extend([
        f"- MAE: {metrics.get('MAE', 0):.4f}",
        f"- RMSE: {metrics.get('RMSE', 0):.4f}",
        f"- MAPE: {metrics.get('MAPE', 0):.4%}",
        "",
        "## Forecast",
        "",
        "| Date | Point | 80% Low | 80% High | 95% Low | 95% High |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in predictions:
        c80 = row.get("confidence_80", {})
        c95 = row.get("confidence_95", {})
        lines.append(
            f"| {row.get('date', '')} | {row.get('point_forecast', 0):.2f} | "
            f"{c80.get('lower', 0):.2f} | {c80.get('upper', 0):.2f} | "
            f"{c95.get('lower', 0):.2f} | {c95.get('upper', 0):.2f} |"
        )

    lines.extend(["", "## Risk Breakdown", ""])
    for key, value in risks.items():
        lines.append(f"- {key}: {float(value):.2f}")
    lines.extend(["", "## News Context", "", evaluation.get("news_context", {}).get("summary", "")])
    
    insight_summary = reporter.get("insight_summary")
    if insight_summary:
        lines.extend(["", "## AI Insights", "", insight_summary])

    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return str(path)
