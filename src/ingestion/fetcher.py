"""yfinance data fetcher with retry logic."""

import time
from datetime import date
from typing import Any

import yfinance as yf
from yfinance.exceptions import YFException

from .exceptions import FetchError


def fetch_with_retry(
    ticker: str,
    start_date: date,
    end_date: date,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """Fetch stock data from yfinance with exponential backoff retry.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA").
        start_date: Start date for historical data.
        end_date: End date for historical data.
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Base delay in seconds for exponential backoff (default: 1.0).

    Returns:
        pandas.DataFrame with historical stock data.

    Raises:
        FetchError: If all retries fail or ticker is invalid.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                raise FetchError(f"No data returned for ticker {ticker}")

            return df

        except YFException as e:
            last_error = e
            if "404" in str(e) or "Not Found" in str(e):
                raise FetchError(f"Ticker '{ticker}' not found") from e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            continue

        except (ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            continue

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            continue

    raise FetchError(
        f"Failed to fetch data for {ticker} after {max_retries} attempts. "
        f"Last error: {last_error}"
    ) from last_error
