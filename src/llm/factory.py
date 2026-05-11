from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config_manager import config_manager



LLM_REGISTRY = {
    "openai": ChatOpenAI,
    "gemini": ChatGoogleGenerativeAI,
}


def _load_llm_config() -> dict[str, Any]:
    """Load LLM config from agent.yaml via config_manager."""
    agent_config = config_manager.agent
    return agent_config.get("llm", {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.0
    })


class LlmFactory:
    _instances = {}

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}

    def get_llm(self, name: str = "default"):
        if name in self._instances:
            return self._instances[name]

        llm_cfg = self._config.get(name, self._config.get("default", {}))
        provider = llm_cfg.get("provider", "openai")
        model = llm_cfg.get("model", "gpt-4o" if provider == "openai" else "gemini-2.0-flash")
        temperature = llm_cfg.get("temperature", 0.0)
        api_key = llm_cfg.get("api_key")
        base_url = llm_cfg.get("base_url")

        if provider not in LLM_REGISTRY:
            raise ValueError(f"Unsupported provider: {provider}. Available: {list(LLM_REGISTRY.keys())}")

        if provider == "openai":
            llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url, extra_body={"reasoning_split": True})
        else:
            llm = LLM_REGISTRY[provider](model=model, temperature=temperature, api_key=api_key)
        self._instances[name] = llm
        return llm

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LlmFactory":
        return cls(config)

    def reset(self):
        self._instances.clear()


# Global singleton instance pre-loaded from config
_llm_config = _load_llm_config()
llm_factory = LlmFactory.from_config({"default": _llm_config})


def get_llm():
    """Get the default LLM instance."""
    return llm_factory.get_llm("default")