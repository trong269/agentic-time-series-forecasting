"""Stock data ingestion module.

Fetches daily stock data via yfinance and stores in SQLite.
"""

from datetime import date, timedelta

import pandas as pd

from src.utils.config_manager import config_manager

from .exceptions import FetchError, IngestionError, StorageError
from .fetcher import fetch_with_retry
from .storage import get_latest_date, get_data as _get_data, init_db, upsert_data

__all__ = [
    "fetch_stock_data",
    "get_stock_data",
    "IngestionError",
    "FetchError",
    "StorageError",
]


def fetch_stock_data(ticker: str | None = None) -> dict:
    """Fetch latest stock data and store in database.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL"). If None, uses config.

    Uses incremental fetch strategy:
    - If DB is empty: fetch last lookback_days from config
    - If DB has data: fetch from (latest_date + 1 day) to today

    Returns:
        dict with status, rows_fetched, date_range keys.
        e.g., {"status": "success", "rows_fetched": 5, "date_range": ("2024-05-01", "2024-05-05")}
        or {"status": "error", "error": "error message"}
    """
    cfg = config_manager.ingestion
    ticker = ticker or cfg.get("ticker", "NVDA")
    db_path = cfg.get("db", {}).get("path", "data/stocks.db")
    table_name = cfg.get("db", {}).get("table_name", "stock_daily")
    lookback_days = cfg.get("lookback_days", 730)
    retry_attempts = cfg.get("retry_attempts", 3)
    retry_delay = cfg.get("retry_delay", 1.0)

    try:
        init_db(db_path, table_name)

        today = date.today()
        latest_date = get_latest_date(db_path, ticker, table_name)

        if latest_date is None:
            start_date = today - timedelta(days=lookback_days)
        else:
            start_date = date.fromisoformat(latest_date) + timedelta(days=1)

        if start_date > today:
            return {
                "status": "success",
                "rows_fetched": 0,
                "date_range": (str(start_date), str(today)),
                "message": "No new data to fetch",
            }

        df = fetch_with_retry(
            ticker=ticker,
            start_date=start_date,
            end_date=today + timedelta(days=1),
            max_retries=retry_attempts,
            base_delay=retry_delay,
        )

        rows_inserted = upsert_data(db_path, df, ticker, table_name)

        fetched_start = df.index.min().strftime("%Y-%m-%d") if not df.empty else str(start_date)
        fetched_end = df.index.max().strftime("%Y-%m-%d") if not df.empty else str(today)

        return {
            "status": "success",
            "rows_fetched": rows_inserted,
            "date_range": (fetched_start, fetched_end),
        }

    except IngestionError:
        raise
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_stock_data(
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Retrieve stock data for downstream modules (preprocessing, forecasting).

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL"). If None, uses config.
        start_date: Optional start date in ISO format (YYYY-MM-DD).
        end_date: Optional end date in ISO format (YYYY-MM-DD).

    Returns:
        DataFrame with columns: date, open, high, low, close, adj_close, volume.
    """
    cfg = config_manager.ingestion
    db_path = cfg.get("db", {}).get("path", "data/stocks.db")
    ticker = ticker or cfg.get("ticker", "NVDA")
    table_name = cfg.get("db", {}).get("table_name", "stock_daily")

    return _get_data(db_path, ticker, start_date, end_date, table_name)