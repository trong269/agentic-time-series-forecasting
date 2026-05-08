"""Custom exceptions for the ingestion module."""


class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class FetchError(IngestionError):
    """Raised when yfinance data fetch fails after retries."""
    pass


class StorageError(IngestionError):
    """Raised when database operations fail."""
    pass