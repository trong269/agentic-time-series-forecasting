# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

An end-to-end automated pipeline for time series forecasting focused on **NVDA stock** using **yfinance** for live data ingestion. The system generates 7-day price predictions via ML models, then uses an AI agent (LangGraph) to evaluate, interpret, and act on those predictions.

## Architecture

```
[Live Data Ingestion] → [Preprocessing] → [Forecasting Model] → [AI Agent Evaluator] → [Report Output]
                                                                              ↑
                                                                        [Improvement Agent]
```

### Directory Structure

- `src/ingestion/` — Fetches daily NVDA stock data via **yfinance**. Writes to `data/`.
- `src/preprocessing/` — Cleans ingested data (missing values, duplicates), derives lag features.
- `src/forecasting/` — Generates 7-day point forecasts with 80%/95% confidence intervals. Supports ARIMA, Prophet, XGBoost, LSTM.
- `src/agents/` — AI agent that evaluates prediction quality, diagnoses model issues, and recommends adjustments.
- `src/evaluation/` — Computes holdout metrics (MAE, RMSE, MAPE).
- `src/reporting/` — Produces structured JSON reports and human-readable Markdown.
- `src/workflow/` — LangGraph orchestration (states, nodes, edges).
- `src/tools/` — Agent tools (web search, data analysis, model control).
- `src/llm/` — LLM provider abstraction (Anthropic, OpenAI).
- `scripts/` — Entry points: `run_daily_pipeline.py` (full pipeline), `train_model.py` (manual retraining).
- `configs/` — YAML configuration for data sources, model params, and agent settings.
- `artifacts/models/` — Serialized forecast models.
- `artifacts/reports/` — Generated daily reports.
- `data/` — Raw ingested NVDA time series data stored in **SQLite** (`data/stocks.db`).

## Common Commands

```bash
# Run the full daily pipeline (ingest → forecast → evaluate → report)
python scripts/run_daily_pipeline.py

# Retrain forecast model manually
python scripts/train_model.py

# Run tests
pytest tests/

# Install dependencies
pip install -r requirements.txt
```

## Configuration

All settings managed via `configs/app.yaml`. Environment variables (API keys, DB paths) stored in `.env` (not committed).

## Data Flow

1. **Ingestion** pulls latest NVDA data via yfinance and stores in SQLite (`data/stocks.db`)
2. **Preprocessing** handles missing values, outliers, feature engineering
3. **Forecasting** outputs predictions with confidence intervals to `artifacts/reports/`
4. **Agent Evaluator** reads predictions, diagnoses issues, outputs structured feedback
5. **Improvement Agent** receives feedback and adjusts model parameters or triggers retraining

---

## Development Guidelines

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
