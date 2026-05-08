"""SQLite storage operations for stock data."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import StorageError


def init_db(db_path: str | Path, table_name: str = "stock_daily") -> None:
    """Create the database and stock_daily table if they don't exist.

    Args:
        db_path: Path to the SQLite database file.
        table_name: Name of the table to store daily stock data.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume INTEGER,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (ticker, date)
    );
    """

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(create_table_sql)
            conn.commit()
    except sqlite3.Error as e:
        raise StorageError(f"Failed to initialize database: {e}") from e


def upsert_data(db_path: str | Path, df: pd.DataFrame, ticker: str, table_name: str = "stock_daily") -> int:
    """Insert or replace stock rows into the database.

    Args:
        db_path: Path to the SQLite database file.
        df: DataFrame with columns: date, open, high, low, close, adj_close, volume.
        ticker: Stock ticker symbol (e.g., "NVDA").
        table_name: Name of the table to insert into.

    Returns:
        Number of rows inserted/updated.

    Raises:
        StorageError: If database operation fails.
    """
    if df.empty:
        return 0

    fetched_at = datetime.now(timezone.utc).isoformat()

    insert_sql = f"""
    INSERT OR REPLACE INTO {table_name}
        (ticker, date, open, high, low, close, adj_close, volume, fetched_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    try:
        with sqlite3.connect(db_path) as conn:
            # Normalize column names from yfinance (Title case) to snake_case
            df_normalized = df.copy()
            column_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
            df_normalized.columns = [column_map.get(c, c) for c in df_normalized.columns]

            # yfinance returns date as DatetimeIndex, reset to get it as a column
            df_normalized = df_normalized.reset_index()
            if "Date" in df_normalized.columns:
                df_normalized = df_normalized.rename(columns={"Date": "date"})
            elif df_normalized.index.name == "Date" or df_normalized.index.name is None:
                df_normalized["date"] = df_normalized.index.strftime("%Y-%m-%d")
                df_normalized = df_normalized.reset_index(drop=True)

            # Ensure date is string ISO format
            if "date" in df_normalized.columns:
                df_normalized["date"] = pd.to_datetime(df_normalized["date"]).dt.strftime("%Y-%m-%d")

            rows = []
            for _, row in df_normalized.iterrows():
                rows.append((
                    ticker,
                    row["date"],
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("adj_close"),
                    int(row["volume"]) if pd.notna(row.get("volume")) else None,
                    fetched_at,
                ))

            conn.executemany(insert_sql, rows)
            conn.commit()
            return len(rows)
    except sqlite3.Error as e:
        raise StorageError(f"Failed to upsert data: {e}") from e
    except Exception as e:
        raise StorageError(f"Unexpected error during upsert: {e}") from e


def get_data(
    db_path: str | Path,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    table_name: str = "stock_daily",
) -> pd.DataFrame:
    """Retrieve stock data within a date range.

    Args:
        db_path: Path to the SQLite database file.
        ticker: Stock ticker symbol (e.g., "NVDA").
        start_date: Start date in ISO format (YYYY-MM-DD), inclusive.
        end_date: End date in ISO format (YYYY-MM-DD), inclusive.
        table_name: Name of the table to query.

    Returns:
        DataFrame with columns: date, open, high, low, close, adj_close, volume.

    Raises:
        StorageError: If database query fails.
    """
    query = f"SELECT date, open, high, low, close, adj_close, volume FROM {table_name} WHERE ticker = ?"
    params: list[Any] = [ticker]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date ASC"

    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            return df
    except sqlite3.Error as e:
        raise StorageError(f"Failed to query data: {e}") from e


def get_latest_date(db_path: str | Path, ticker: str, table_name: str = "stock_daily") -> str | None:
    """Get the latest date stored for a ticker.

    Args:
        db_path: Path to the SQLite database file.
        ticker: Stock ticker symbol (e.g., "NVDA").
        table_name: Name of the table to query.

    Returns:
        Latest date as ISO string (YYYY-MM-DD), or None if no data exists.

    Raises:
        StorageError: If database query fails.
    """
    query = f"SELECT MAX(date) FROM {table_name} WHERE ticker = ?"

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(query, (ticker,))
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
    except sqlite3.Error as e:
        raise StorageError(f"Failed to get latest date: {e}") from e