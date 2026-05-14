"""Langfuse tracing helpers."""

from __future__ import annotations

from typing import Any

from src.utils.config_manager import config_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _langfuse_config() -> dict[str, Any]:
    return (config_manager.app.get("observability", {}) or {}).get("langfuse", {}) or {}


def is_langfuse_enabled() -> bool:
    """Return whether Langfuse tracing is enabled in app config."""
    return _as_bool(_langfuse_config().get("enabled"), default=False)


def build_langfuse_runnable_config(
    *,
    workflow_name: str,
    input_data: dict[str, Any],
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build LangChain RunnableConfig with Langfuse callback and metadata."""
    if not is_langfuse_enabled():
        return base_config or {}

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning("Langfuse is enabled but package 'langfuse' is not installed.")
        return base_config or {}

    cfg = _langfuse_config()
    callbacks = list((base_config or {}).get("callbacks", []))
    callbacks.append(CallbackHandler())

    metadata = dict((base_config or {}).get("metadata", {}))
    metadata.update({
        "langfuse_session_id": input_data.get("run_id"),
        "ticker": input_data.get("ticker"),
        "workflow_name": workflow_name,
        "run_date": input_data.get("run_date"),
    })
    user_id = cfg.get("user_id")
    if user_id:
        metadata["langfuse_user_id"] = user_id

    tags = list((base_config or {}).get("tags", []))
    configured_tags = cfg.get("tags", [])
    if isinstance(configured_tags, str):
        configured_tags = [configured_tags]
    tags.extend(configured_tags)
    tags.extend([workflow_name, str(input_data.get("ticker", "")).upper()])
    tags = [tag for tag in dict.fromkeys(tags) if tag]

    runnable_config = dict(base_config or {})
    runnable_config.update({
        "callbacks": callbacks,
        "metadata": metadata,
        "tags": tags,
        "run_name": workflow_name,
    })
    return runnable_config


def flush_langfuse() -> None:
    """Flush queued Langfuse events for short-lived CLI runs."""
    if not is_langfuse_enabled() or not _as_bool(_langfuse_config().get("flush_on_exit"), default=True):
        return
    try:
        from langfuse import get_client
    except ImportError:
        return
    get_client().flush()
