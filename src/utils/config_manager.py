"""Centralized configuration manager for the project."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file at module import
load_dotenv()


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR:default} patterns in config values."""
    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        def replacer(m):
            var_name, default = m.group(1), m.group(2) or ''
            return os.environ.get(var_name, default)
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


class ConfigManager:
    """Loads and provides access to all YAML configuration files."""

    def __init__(self, configs_dir: str | Path = "configs"):
        self.configs_dir = Path(configs_dir)
        self._configs: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all YAML config files from configs directory."""
        for yaml_file in self.configs_dir.glob("*.yaml"):
            config_name = yaml_file.stem
            config = self._load_yaml(yaml_file)
            self._configs[config_name] = _expand_env_vars(config)

    def _load_yaml(self, path: Path) -> dict:
        """Load a single YAML file, return empty dict if file is empty or missing."""
        if not path.exists() or path.stat().st_size == 0:
            return {}
        with open(path) as f:
            content = yaml.safe_load(f)
            return content if content else {}

    def get(self, config_name: str, key: str | None = None, default: Any = None) -> Any:
        """Get a config value by name and optional key.

        Args:
            config_name: Name of config file (e.g., "ingestion", "model")
            key: Optional dot-notation key (e.g., "db.path")
            default: Default value if key not found

        Returns:
            Config value or default
        """
        config = self._configs.get(config_name, {})
        if key is None:
            return config

        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def ingestion(self) -> dict:
        """Get ingestion config."""
        return self._configs.get("ingestion", {})

    @property
    def app(self) -> dict:
        """Get app config."""
        return self._configs.get("app", {})

    @property
    def agent(self) -> dict:
        """Get agent config."""
        return self._configs.get("agent", {})

    @property
    def model(self) -> dict:
        """Get model config."""
        return self._configs.get("model", {})

    @property
    def preprocessing(self) -> dict:
        """Get preprocessing config."""
        return self._configs.get("preprocessing", {})


# Global singleton instance
config_manager = ConfigManager()