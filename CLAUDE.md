# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

An end-to-end automated pipeline for time series forecasting focused on **NVDA stock** using **yfinance** for live data ingestion. The forecasting core is a **one-step-ahead quantile XGBoost model** that is rolled forward recursively to produce a 7-day forecast with confidence intervals.

The current project also includes a LangGraph-based multi-agent daily workflow:

1. Load local/latest stock inputs.
2. Forecast with the latest usable model.
3. Evaluate forecast risk with technical metrics and news context.
4. Generate JSON/Markdown reports.
5. Optionally retrain candidate models.
6. Re-forecast and re-evaluate the selected retrain candidate.
7. Promote or reject the candidate in evaluation, then write a final report.

## Architecture

```
[Load Inputs]
      ↓
[Forecasting Agent]
      ↓
[Evaluator Agent]
      ↓
[Reporter Agent] ── action=retrain ──→ [Improvement Agent]
      ↑                                      │
      └──── rerun with selected tmp model ───┘
```

Promotion contract:

- `ImprovementAgent` trains candidate models and selects the best temporary candidate only.
- `EvaluatorAgent` is the owner of final promote/reject after the selected candidate is forecast and evaluated.
- A promoted candidate is moved from `artifacts/models/tmp_*` to `artifacts/models/ver_N/`.
- A rejected candidate is deleted and workflow state is restored to the original model/output before final reporting.

### Directory Structure

- `src/ingestion/` — Fetches daily stock data via **yfinance**. Writes to `data/`.
- `src/preprocessing/` — Feature engineering and train/test splitting for XGBoost.
- `src/forecasting/` — One-step quantile XGBoost training and recursive multi-day prediction.
- `src/agents/` — LangGraph agents for forecasting, evaluation, reporting, and improvement.
- `src/workflow/` — Daily LangGraph orchestration (states, nodes, edges, promotion/rejection lifecycle).
- `src/tools/` — External tool wrappers such as Tavily news search.
- `src/llm/` — LLM provider abstraction (OpenAI/Gemini via LangChain).
- `scripts/` — Entry points: `run_daily_pipeline.py` (full pipeline), `train_model.py` (manual retraining).
- `configs/` — YAML configuration for data sources, model params, and agent settings.
- `artifacts/models/` — Serialized forecast models (versioned: `ver_N/`).
- `artifacts/reports/` — Generated daily reports including prediction JSON.
- `data/` — Raw ingested stock time series data stored in **SQLite** (`data/stocks.db`).

## Common Commands

```bash
# Fetch stock data
python -c "from src.ingestion import fetch_stock_data; print(fetch_stock_data('NVDA'))"

# Get stock data
python -c "from src.ingestion import get_stock_data; df = get_stock_data('NVDA')"

# Preprocess for training using project config
python -c "from src.preprocessing import preprocess_for_training; result = preprocess_for_training('NVDA')"

# Preprocess explicit DataFrame/config for agent use
python -c "from src.ingestion import get_stock_data; from src.preprocessing import preprocess_data; from src.utils.config_manager import config_manager; df = get_stock_data('NVDA'); result = preprocess_data(df, config_manager.preprocessing)"

# Run the full daily workflow (forecast → evaluate → report → optional retrain/evaluate/report)
python scripts/run_daily_pipeline.py

# Run daily workflow for an explicit ticker
python scripts/run_daily_pipeline.py NVDA

# Run workflow and fetch latest yfinance data first
python scripts/run_daily_pipeline.py NVDA --fetch-latest

# Retrain forecast model manually
python scripts/train_model.py

# Run prediction and save to JSON
python -c "
import json
from src.forecasting.predictor import predict_with_intervals
result = predict_with_intervals('NVDA', horizon=7)
with open('artifacts/reports/prediction_result.json', 'w') as f:
    json.dump(result, f, indent=2)
"

# Run tests without pytest
python -m unittest tests.test_agents -v

# Install dependencies
pip install -r requirements.txt
```

## Configuration

All settings managed via YAML configs in `configs/`. Environment variables (API keys, DB paths) stored in `.env` (not committed).

### Config Files

| File | Purpose |
|------|---------|
| `configs/app.yaml` | General app settings |
| `configs/ingestion.yaml` | Data ingestion settings (ticker, lookback, retry) |
| `configs/preprocessing.yaml` | Feature engineering settings (lags, technical indicators) |
| `configs/model.yaml` | XGBoost model parameters, quantiles, artifacts directory |
| `configs/agent.yaml` | Agent settings |

### Agent Workflow Config

**`configs/agent.yaml`**

- `llm`: provider/model credentials for LLM-backed enrichment/report text.
- `evaluator.accept_threshold`: score threshold for accept.
- `evaluator.retrain_threshold`: score threshold below which reporting may request retrain.
- `evaluator.news`: Tavily search settings.
- `reporter.reports_dir`: JSON/Markdown output directory.
- `reporter.history_n`: number of prior reports used for trend context.
- `improvement.candidates`: XGBoost parameter candidates trained during retrain.

### Key Configuration Keys

**`configs/preprocessing.yaml`** (May 2026 - updated)
```yaml
features:
  price_lags: [1, 7, 30]
  ma_windows: [7, 21, 50]
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_window: 20
  bb_std: 2
  volatility_windows: [21]
  include_calendar: true
  calendar_features: [quarter, month, week_of_year]
  # Note: volume_ma_windows and include_atr REMOVED - cannot be used in recursive prediction
  return_periods: [21]
  include_close_to_ma: true
  close_to_ma_windows: [50]

target:
  horizon: 1
  type: "close"

split:
  test_days: 60
  gap: 0
```

**`configs/model.yaml`**
```yaml
artifacts_dir: "artifacts/models/"
quantiles: [0.025, 0.10, 0.50, 0.90, 0.975]
xgb_params:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  random_state: 42
```

---

## Session Progress (May 2026)

### Phase 1: Live Data Ingestion ✅ COMPLETED

- [x] **ConfigManager** (`src/utils/config_manager.py`)
  - Central singleton that loads all YAML configs from `configs/`
  - Properties: `config_manager.ingestion`, `config_manager.app`, `config_manager.preprocessing`, `config_manager.agent`, `config_manager.model`

- [x] **Ingestion Module** (`src/ingestion/`)
  - `fetch_stock_data(ticker)` — Fetch and store stock data via yfinance
  - `get_stock_data(ticker, start_date, end_date)` — Retrieve data for downstream
  - Incremental fetch: only fetches new data since last stored date
  - Exponential backoff retry for API resilience
  - Dynamic ticker support: any stock ticker works via parameter or config
  - Returns structured error payloads instead of mixing raise/return behavior
  - Current runtime limitation: network/DNS failures to Yahoo are surfaced cleanly, but not masked

- [x] **SQLite Storage** (`src/ingestion/storage.py`)
  - `init_db(db_path, table_name)` — Create table with dynamic name
  - `upsert_data(db_path, df, ticker, table_name)` — Insert or replace
  - `get_data(db_path, ticker, start_date, end_date, table_name)` — Query
  - `get_latest_date(db_path, ticker, table_name)` — For incremental fetch
  - Validates SQLite identifiers before interpolating table names

### Phase 2: Preprocessing ✅ COMPLETED

- [x] **Preprocessing Pipeline** (`src/preprocessing/pipeline.py`)
  - `preprocess_data(df, config)` — Explicit preprocessing entrypoint for agents
  - `preprocess_for_training(ticker)` — Convenience function
  - `preprocess_for_prediction(ticker)` — Returns `df_raw`, `last_features`, `last_date`, `close_list`, `feature_columns`
  - Time-based train/test split (last 60 days for test)
  - Auto-excludes columns with all NaN values (e.g., `adj_close`)

- [x] **Feature Engineering** (`src/preprocessing/feature_functions.py`)
  - Price lag features: close_lag_1, close_lag_7, close_lag_30
  - Return features: return_21d (only)
  - Moving averages: MA_7, MA_21, MA_50
  - Close-to-MA ratio: close_to_MA_50
  - Technical: MACD_signal, BB_position, volatility_21d
  - Calendar: quarter, month, week_of_year
  - Training and prediction feature builders are now aligned for recursive inference
  - **Note: ATR and volume features REMOVED** - cannot be computed in recursive prediction without simulation

- [x] **Technical Indicators** (`src/preprocessing/technical.py`)
  - `calculate_rsi()` — Relative Strength Index (not used in final model)
  - `calculate_macd()` — MACD line, signal, histogram (only signal used)
  - `calculate_bollinger_bands()` — Upper, middle, lower bands (only position used)
  - `calculate_atr()` — Average True Range (removed from features May 2026)
  - `calculate_volatility()` — Rolling standard deviation

- [x] **Calendar Features** (`src/preprocessing/calendar.py`)
  - quarter, month, week_of_year (filtered for importance)

- [x] **Data Validation** (`src/preprocessing/validator.py`)
  - `validate_data()` — Check required columns, date monotonicity
  - `handle_missing_values()` — Forward fill for stock data
  - `get_data_summary()` — Statistics overview

- [x] **Feature Selection** (May 2026)
  - Analyzed correlations and feature importance
  - Final: 14 features (reduced from 17)
  - **Dropped features:** ATR, volume_MA_7, volume_MA_21 (require "current moment" data or simulation)
  - **Dropped earlier:** open, high, low, close, volume, RSI, volume_lags, return_1d/5d/7d, is_* flags, day_of_week, day_of_month

### Phase 3: Forecasting ✅ COMPLETED

- [x] **XGBoost Model** (`src/forecasting/models/xgboost_model.py`)
  - `train_xgb_quantile(X_train, y_train, quantile, xgb_params)` — Train quantile regression model
  - `save_model(model, path)` / `load_model(path)` — Persist to disk via joblib

- [x] **Trainer** (`src/forecasting/trainer.py`)
  - `train_xgboost_forecaster(ticker)` — Trains 5 XGBoost quantile models (0.025, 0.10, 0.50, 0.90, 0.975)
  - Versioned model storage: `artifacts/models/ver_N/`
  - Metadata saved: `metadata.json` with metrics, feature importance, feature columns, preprocessing/model config
  - **Fix (May 2026):** Model trains on `X_train` only and evaluates on `X_test` holdout
  - **Current contract:** training target is one-step-ahead close (`target.horizon: 1`)

- [x] **Predictor** (`src/forecasting/predictor.py`)
  - `predict_with_intervals(ticker, horizon=7)` — 7-day recursive prediction with confidence intervals
  - `build_prediction_features(close_list, feature_config, feature_date)` — Agent-ready explicit feature builder
  - `predict_single_day(models, close_list, feature_config, feature_date)` — Agent-ready explicit one-step predictor
  - `_compute_holdout_metrics()` — Computes MAE, RMSE, MAPE on last 7 days of test set
  - Quantile outputs are monotonized before returning intervals
  - Output: JSON with predictions and holdout_metrics

- [x] **Scripts** (`scripts/train_model.py`)
  - Standalone model retraining script

**Model Metrics Note:**

Model metrics are version-specific and stored in each `artifacts/models/ver_N/metadata.json`.
Do not hard-code a "current best" model version in docs; use `resolve_latest_usable_model()` or inspect model metadata.

### Phase 4: Agent Workflow ✅ IMPLEMENTED

- [x] **BaseAgent / BaseWorkflow**
  - Shared LangGraph compile/invoke pattern.
  - Runnable config is forwarded so tracing/callbacks can flow into nested agent graphs.

- [x] **ForecastingAgent**
  - Loads explicit model version directory.
  - Generates 7-day recursive forecast with 80%/95% intervals.
  - Adds diagnostics such as average interval width, 7-day return, latest close, and optional live MAPE from prior reports.

- [x] **EvaluatorAgent**
  - Gathers Tavily news context when available.
  - Computes normalized technical/news risk breakdown.
  - Produces `trust_score`, `decision_band`, and structured `evaluation_output`.
  - Reuses cached news context during post-retrain evaluation to avoid repeated Tavily calls.

- [x] **ReporterAgent**
  - Computes composite score and trend factors from prior reports.
  - Decides `accept`, `retrain`, `accept_after_retrain`, or forced workflow actions.
  - Writes canonical daily JSON and Markdown reports to `artifacts/reports/YYYY-MM-DD_TICKER_report.*`.

- [x] **ImprovementAgent**
  - Diagnoses whether retrain is warranted.
  - Trains configured XGBoost candidates into temporary `artifacts/models/tmp_*` directories.
  - Selects the best temporary candidate by training holdout score.
  - Does **not** promote. Promotion is deliberately deferred to workflow evaluation after candidate rerun.

- [x] **DailyForecastingWorkflow**
  - Graph: `load_inputs -> forecasting -> evaluation -> reporting -> optional improvement -> forecasting -> evaluation -> reporting`.
  - Retrain budget is capped at one cycle per run.
  - Candidate promotion/rejection happens after post-retrain evaluation.
  - Reject path deletes the temporary candidate, restores the original model/output, and final report uses `reject_retrained_keep_old`.

### Current Known Issues / Follow-Ups

- Tavily/network failures currently fail evaluator unless the caller injects/offlines news context; production workflow should degrade to `news_unavailable` instead of crashing.
- Temporary retrain directories are named `tmp_0`, `tmp_1`, `tmp_2`; make them run-scoped to avoid stale artifact collisions after crashes.
- Skip-retrain due to external market/news conditions should use a distinct action such as `skip_retrain_keep_old` rather than `reject_retrained_keep_old`.
- Promotion threshold is currently `decision_band != "retrain"` and `new_composite_score > old_composite_score`; consider a configurable minimum delta and/or requiring `accept`.
- `pytest` may be absent in some local conda envs; `python -m unittest tests.test_agents -v` is the current reliable test command.

### Agent-Ready Functions

Prefer these explicit functions in future agents:

- `src.ingestion.get_stock_data()`
- `src.preprocessing.preprocess_data()`
- `src.preprocessing.create_features()`
- `src.preprocessing.split_train_test()`
- `src.preprocessing.trim_dataframe()`
- `src.forecasting.train_quantile_models()`
- `src.forecasting.compute_metrics()`
- `src.forecasting.build_prediction_features()`
- `src.forecasting.predict_single_day()`
- `src.forecasting.compute_holdout_metrics()`

Convenience wrappers that still touch project config / DB / artifacts:

- `src.ingestion.fetch_stock_data()`
- `src.preprocessing.preprocess_for_training()`
- `src.preprocessing.preprocess_for_prediction()`
- `src.forecasting.train_xgboost_forecaster()`
- `src.forecasting.load_models()`
- `src.forecasting.predict_with_intervals()`

---

## Data Flow

1. **Ingestion** pulls stock data via yfinance and stores in SQLite (`data/stocks.db`)
2. **Preprocessing** creates 14 features: price lags, returns, MAs, technical indicators, calendar
3. **Forecasting** trains a one-step quantile model and rolls it forward recursively for 7 trading days
4. **Evaluator Agent** reads predictions and news context, then outputs trust/risk decisions
5. **Reporter Agent** writes structured JSON/Markdown and may request retrain
6. **Improvement Agent** trains candidate models only when requested
7. **Post-retrain Evaluation** decides whether to promote the selected candidate or keep the original model
8. **Final Reporting** writes the final accept/reject outcome

## Retrain / Promotion Flow

When `ReporterAgent` returns `action == "retrain"` and the retrain budget is available:

1. `ImprovementAgent` trains configured candidate parameter sets into temporary model directories.
2. `ImprovementAgent` selects the best temporary candidate by training holdout score.
3. Workflow stores the original model path/version and original forecast/evaluation/report outputs.
4. Workflow reruns `ForecastingAgent` using the selected temporary model.
5. Workflow reruns `EvaluatorAgent` on the candidate forecast.
6. `evaluation_workflow_node` finalizes promotion:
   - Promote when candidate decision band is not `retrain` and candidate composite score beats the original composite score.
   - Reject otherwise.
7. Promote path moves the temporary model to `ver_N` and updates metadata.
8. Reject path deletes the temporary model and restores original state for final reporting.

---

## Feature Validation Rule (Critical)

**Before adding any feature, verify it can be computed without "current moment" data:**

| Feature Type | Example | Valid? |
|--------------|---------|--------|
| Lag-based | close_lag_1, close_lag_7 | ✅ Uses only past data |
| Derived (past) | MA_7, return_21d, BB_position | ✅ Uses only past data |
| Volume MA | volume_MA_7 | ❌ Stale in recursive prediction - cannot update without actual volume |
| ATR | ATR | ❌ Requires future high/low (must simulate) |
| Current day OHLCV | close, volume | ❌ Not known at prediction time |
| Calendar | quarter, month | ✅ Available for forecast date |

**Any feature requiring simulation or using "current moment" data MUST be excluded.**

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

### 5. Post-Task File Audit

**After completing any task, list all modified files with change summaries.**

Every completed task must include:
1. **List of modified files** — exact file paths
2. **Change summary per file** — what was changed, why, and how the logic differs from the old code
3. **Output this list in the final response** so it is easy to review

Format:
```
Modified files:
- `src/foo/bar.py` — Added X to handle Y; old code did Z but failed when W.
- `src/baz/qux.py` — Refactored X into a separate function to fix Y.
```

**Why:** This makes code reviews faster, keeps git history meaningful, and forces the author to verify each change is intentional rather than accidental.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
