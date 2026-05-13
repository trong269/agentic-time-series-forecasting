"""Prompt factory to load and format markdown prompts."""

import os
from pathlib import Path
from string import Template
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptFactory:
    """Factory to load and manage prompt templates."""

    _prompts_dir = Path(__file__).parent
    _cache: dict[str, Template] = {}

    @classmethod
    def get_prompt(cls, name: str, **kwargs: Any) -> str:
        """
        Load a prompt template from <name>.md and format it.

        Args:
            name: The name of the prompt file (without .md).
            **kwargs: Variables to inject into the template using $var syntax.

        Returns:
            The formatted prompt string.
        """
        if name not in cls._cache:
            prompt_path = cls._prompts_dir / f"{name}.md"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template {prompt_path} not found.")

            with open(prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
            cls._cache[name] = Template(content)

        template = cls._cache[name]
        try:
            return template.safe_substitute(**kwargs)
        except Exception as e:
            logger.error("Error formatting prompt %s: %s", name, e)
            return template.template

prompt_factory = PromptFactory()
