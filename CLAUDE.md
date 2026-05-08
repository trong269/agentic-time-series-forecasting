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
- `src/forecasting/` — XGBoost model for 7-day price predictions.
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
| `configs/model.yaml` | Model parameters |
| `configs/agent.yaml` | Agent settings |

### Key Configuration Keys

**`configs/ingestion.yaml`**
```yaml
ticker: "NVDA"           # Default ticker (can override in code)
lookback_days: 730       # Days of historical data to fetch
retry_attempts: 3
retry_delay: 1.0
db:
  path: "data/stocks.db"
  table_name: "stock_daily"
```

**`configs/preprocessing.yaml`**
```yaml
features:
  price_lags: [1, 2, 3, 5, 7, 14, 21, 30]
  volume_lags: [1, 5, 21]
  ma_windows: [7, 21, 50]
  rsi_period: 14
  macd_fast: 12, macd_slow: 26, macd_signal: 9
  bb_window: 20, bb_std: 2
  volatility_windows: [7, 21]
  include_calendar: true

target:
  horizon: 7                # Predict 7 days ahead
  type: "return"             # "return" or "direction"

split:
  test_days: 60             # Last 60 days for testing
  gap: 0
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
  - Price lag features: close_lag_1, _2, _3, _5, _7, _14, _21, _30
  - Return features: return_1d, return_5d, return_7d, return_21d
  - Moving averages: MA_7, MA_21, MA_50, close_to_MA ratios
  - Volume features: volume_lag, volume_MA, volume_ratio

- [x] **Technical Indicators** (`src/preprocessing/technical.py`)
  - `calculate_rsi()` — Relative Strength Index
  - `calculate_macd()` — MACD line, signal, histogram
  - `calculate_bollinger_bands()` — Upper, middle, lower bands
  - `calculate_atr()` — Average True Range
  - `calculate_volatility()` — Rolling standard deviation

- [x] **Calendar Features** (`src/preprocessing/calendar.py`)
  - day_of_week, day_of_month, month, quarter
  - is_month_start, is_month_end, is_quarter_start, is_year_start

- [x] **Data Validation** (`src/preprocessing/validator.py`)
  - `validate_data()` — Check required columns, date monotonicity
  - `handle_missing_values()` — Forward fill for stock data
  - `get_data_summary()` — Statistics overview

**Features Generated (49 features):**
- OHLCV: open, high, low, close, volume
- Price lags: 8 features
- Returns: 4 features
- Moving averages & ratios: 6 features
- Technical: RSI, MACD (3), Bollinger position, ATR, volatility (2)
- Volume: 5 features
- Calendar: 8 features

### Phase 3: Forecasting 🚧 IN PROGRESS

**Next: XGBoost Model Implementation**

- [ ] `src/forecasting/trainer.py` — XGBoost training
- [ ] `src/forecasting/predictor.py` — 7-day prediction
- [ ] `configs/model.yaml` — XGBoost hyperparameters

### Pending

- [ ] Agent evaluation logic (`src/agents/`)
- [ ] Reporting module (`src/reporting/`)
- [ ] Scripts implementation (`run_daily_pipeline.py`, `train_model.py`)
- [ ] Evaluation metrics (`src/evaluation/`)
- [ ] Test coverage

---

## Data Flow

1. **Ingestion** pulls stock data via yfinance and stores in SQLite (`data/stocks.db`)
2. **Preprocessing** creates 49 features: lags, returns, technical indicators, calendar
3. **Forecasting** outputs 7-day predictions with confidence intervals
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