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

- `src/ingestion/` — Fetches daily stock data via **yfinance**. Writes to `data/`.
- `src/preprocessing/` — Feature engineering for XGBoost: lag features, technical indicators, calendar features.
- `src/forecasting/` — XGBoost model for 7-day price predictions with quantile regression for confidence intervals.
- `src/agents/` — AI agent that evaluates prediction quality, diagnoses model issues, and recommends adjustments.
- `src/evaluation/` — Computes holdout metrics (MAE, RMSE, MAPE).
- `src/reporting/` — Produces structured JSON reports and human-readable Markdown.
- `src/workflow/` — LangGraph orchestration (states, nodes, edges).
- `src/tools/` — Agent tools (web search, data analysis, model control).
- `src/llm/` — LLM provider abstraction (Anthropic, OpenAI).
- `scripts/` — Entry points: `run_daily_pipeline.py` (full pipeline), `train_model.py` (manual retraining).
- `configs/` — YAML configuration for data sources, model params, and agent settings.
- `artifacts/models/` — Serialized forecast models (versioned: `ver_N/`).
- `artifacts/reports/` — Generated daily reports including prediction JSON.
- `data/` — Raw ingested stock time series data stored in **SQLite** (`data/stocks.db`).

## Common Commands

```bash
# Fetch stock data
python -c "from src.ingestion import fetch_stock_data; fetch_stock_data('NVDA')"

# Get stock data
python -c "from src.ingestion import get_stock_data; df = get_stock_data('NVDA')"

# Preprocess for training
python -c "from src.preprocessing import preprocess_for_training; result = preprocess_for_training('NVDA')"

# Run the full daily pipeline (ingest → forecast → evaluate → report)
python scripts/run_daily_pipeline.py

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

# Run tests
pytest tests/

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
  horizon: 7
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

- [x] **SQLite Storage** (`src/ingestion/storage.py`)
  - `init_db(db_path, table_name)` — Create table with dynamic name
  - `upsert_data(db_path, df, ticker, table_name)` — Insert or replace
  - `get_data(db_path, ticker, start_date, end_date, table_name)` — Query
  - `get_latest_date(db_path, ticker, table_name)` — For incremental fetch

### Phase 2: Preprocessing ✅ COMPLETED

- [x] **Preprocessing Pipeline** (`src/preprocessing/pipeline.py`)
  - `PreprocessingPipeline` class with `fit_transform()` and `transform()`
  - `preprocess_for_training(ticker)` — Convenience function
  - Time-based train/test split (last 60 days for test)
  - Auto-excludes columns with all NaN values (e.g., `adj_close`)

- [x] **Feature Engineering** (`src/preprocessing/feature_engineering.py`)
  - Price lag features: close_lag_1, close_lag_7, close_lag_30
  - Return features: return_21d (only)
  - Moving averages: MA_7, MA_21, MA_50
  - Close-to-MA ratio: close_to_MA_50
  - Technical: MACD_signal, BB_position, volatility_21d
  - Calendar: quarter, month, week_of_year
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
  - Metadata saved: `metadata.json` with metrics, feature importance, train/test sizes
  - **Fix (May 2026):** Removed data leakage - model now trains only on X_train, evaluates on X_test (proper holdout)

- [x] **Predictor** (`src/forecasting/predictor.py`)
  - `predict_with_intervals(ticker, horizon=7)` — 7-day recursive prediction with confidence intervals
  - `_build_daily_features()` — Builds features from close_list only (no simulation needed)
  - `_compute_holdout_metrics()` — Computes MAE, RMSE, MAPE on last 7 days of test set
  - Output: JSON with predictions and holdout_metrics

- [x] **Scripts** (`scripts/train_model.py`)
  - Standalone model retraining script

**Current Model Metrics (Version 4):**
| Metric | Value |
|--------|-------|
| Train samples | 386 |
| Test samples | 60 |
| Test MAE | ~10.44 |
| Test RMSE | ~13.20 |
| Holdout MAE | ~19.74 (~9.6% MAPE) |

### Pending

- [ ] Agent evaluation logic (`src/agents/`)
- [ ] Reporting module (`src/reporting/`)
- [ ] Scripts implementation (`run_daily_pipeline.py`)
- [ ] Test coverage

---

## Data Flow

1. **Ingestion** pulls stock data via yfinance and stores in SQLite (`data/stocks.db`)
2. **Preprocessing** creates 14 features: price lags, returns, MAs, technical indicators, calendar
3. **Forecasting** outputs 7-day price predictions with quantile-based confidence intervals
4. **Agent Evaluator** reads predictions, diagnoses issues, outputs structured feedback
5. **Improvement Agent** receives feedback and adjusts model parameters or triggers retraining

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

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.