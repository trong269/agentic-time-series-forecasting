import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.agents.evaluator_agent import (
    build_news_context,
    calculate_risk_breakdown,
    heuristic_enrich_article,
)
from src.agents.factory import AgentFactory
from src.agents.reporter_agent import ReporterAgent, assess_history_trend
from src.utils.scoring import composite_score
from src.forecasting.predictor import load_models
from src.workflow import DailyForecastingWorkflow
from src.workflow.base import BaseWorkflow


def _forecasting_output():
    return {
        "ticker": "NVDA",
        "model_version": 1,
        "model_path": "artifacts/models/ver_1",
        "holdout_metrics": {"MAE": 1.0, "RMSE": 2.0, "MAPE": 0.02},
        "predictions": [
            {
                "date": "2026-05-13",
                "point_forecast": 104.0,
                "confidence_80": {"lower": 100.0, "upper": 108.0},
                "confidence_95": {"lower": 96.0, "upper": 112.0},
            }
        ],
        "forecast_diagnostics": {
            "avg_80_width_pct": 8.0,
            "avg_95_width_pct": 16.0,
            "forecast_7d_return_pct": 4.0,
            "latest_close": 100.0,
        },
    }


class AgentTests(unittest.TestCase):
    def test_risk_breakdown_is_deterministic(self):
        df = pd.DataFrame({"close": [90, 91, 92, 93, 94, 95, 96, 97, 98, 100]})
        news = build_news_context("query", [])

        risks = calculate_risk_breakdown(_forecasting_output(), df, news)

        self.assertEqual(risks["holdout_mape_risk"], 20.0)
        self.assertEqual(risks["holdout_rmse_pct_risk"], 20.0)
        self.assertEqual(risks["news_risk"], 0.0)
        self.assertGreaterEqual(composite_score(80.0, risks), 0.0)
        self.assertLessEqual(composite_score(80.0, risks), 100.0)

    def test_news_enrichment_fallback_normalizes_article(self):
        article = heuristic_enrich_article({
            "title": "NVDA faces probe risk",
            "url": "https://example.com",
        })

        self.assertLess(article["sentiment"], 0)
        self.assertGreaterEqual(article["event_severity"], 0.0)
        self.assertLessEqual(article["event_severity"], 1.0)

    def test_reporter_routes_retrain_and_writes_reports(self):
        agent = ReporterAgent()
        evaluation = {
            "trust_score": 40.0,
            "decision_band": "retrain",
            "reason": "low score",
            "risk_breakdown": {
                "holdout_mape_risk": 60.0,
                "holdout_rmse_pct_risk": 50.0,
                "interval_width_risk": 40.0,
                "recent_volatility_risk": 30.0,
                "trend_alignment_risk": 20.0,
                "news_risk": 10.0,
            },
            "news_context": build_news_context("query", []),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent.invoke({
                "ticker": "NVDA",
                "run_id": "test",
                "run_date": "2026-05-12",
                "is_retrain": False,
                "forecasting_output": _forecasting_output(),
                "evaluation_output": evaluation,
                "previous_reports": [],
                "agent_config": {
                    "evaluator": {"accept_threshold": 70, "retrain_threshold": 50},
                    "reporter": {"reports_dir": tmpdir, "history_n": 7},
                },
            })
            output = result["reporter_output"]

            self.assertEqual(output["action"], "retrain")
            self.assertTrue(Path(output["report_paths"]["json"]).exists())
            self.assertTrue(Path(output["report_paths"]["markdown"]).exists())

    def test_history_trend_detects_degradation(self):
        previous = [{"reporter": {"composite_score": score}} for score in [80, 82, 78]]

        self.assertEqual(assess_history_trend(60, previous, 8.0), "degrading")

    def test_load_models_accepts_explicit_version_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            version_dir = Path(tmpdir) / "ver_1"
            version_dir.mkdir()

            self.assertEqual(
                load_models("NVDA", artifacts_dir=version_dir, version_dir=version_dir, quantiles=[0.5]),
                {},
            )

    def test_daily_workflow_uses_base_workflow_graph_pattern(self):
        workflow = DailyForecastingWorkflow()

        self.assertIsInstance(workflow, BaseWorkflow)
        self.assertTrue(AgentFactory.is_ready())
        self.assertIs(workflow.forecasting_agent, AgentFactory.get_agent("forecasting_agent"))
        self.assertIs(workflow.evaluator_agent, AgentFactory.get_agent("evaluator_agent"))
        self.assertIs(workflow.reporter_agent, AgentFactory.get_agent("reporter_agent"))
        self.assertIs(workflow.improvement_agent, AgentFactory.get_agent("improvement_agent"))
        graph = workflow.build_graph()
        self.assertIn("load_inputs", graph.nodes)
        self.assertIn("forecasting", graph.nodes)
        self.assertIn("evaluation", graph.nodes)
        self.assertIn("reporting", graph.nodes)
        self.assertIn("improvement", graph.nodes)


if __name__ == "__main__":
    unittest.main()
