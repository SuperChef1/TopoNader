from __future__ import annotations

import os
import time
from typing import Protocol

import requests

from ..config import TopoNaderAgentConfig


def resolve_toponader_llm_runtime(config: TopoNaderAgentConfig) -> tuple[str, str]:
    """Resolve non-secret runtime settings for execution and provenance."""
    model = os.getenv("TOPONADER_LLM_MODEL_NAME", config.model).strip()
    if model.startswith("deepseek"):
        base_url = os.getenv("TOPONADER_LLM_BASE_URL", "https://api.deepseek.com")
    else:
        base_url = (
            os.getenv("TOPONADER_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or config.base_url
        )
    return model, base_url.rstrip("/")


class TopoNaderLLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> dict: ...


class TopoNaderOpenAICompatibleClient:
    """Project-prefixed adaptation of NADER's LLM client."""

    def __init__(self, config: TopoNaderAgentConfig):
        self.config = config
        self.model, self.base_url = resolve_toponader_llm_runtime(config)
        self.api_key = os.getenv("TOPONADER_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if self.model.startswith("deepseek"):
            self.api_key = os.getenv("TOPONADER_LLM_API_KEY") or os.getenv("API_KEY_DEEPSEEK")

    def complete(self, messages: list[dict[str, str]]) -> dict:
        if not self.api_key:
            raise RuntimeError("TOPONADER_LLM_API_KEY is not set; architecture audit cannot call the agents")
        error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                if self.model.startswith("deepseek"):
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={"model": self.model, "messages": messages},
                        timeout=self.config.timeout_seconds,
                    )
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage") or {}
                    return {
                        "content": data["choices"][0]["message"]["content"],
                        "model": data.get("model", self.model),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    }
                response = requests.post(
                    f"{self.base_url}/responses",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "input": messages},
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                content = "".join(
                    part.get("text", "")
                    for item in data.get("output", [])
                    if item.get("type") == "message"
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                )
                usage = data.get("usage") or {}
                return {
                    "content": content,
                    "model": data.get("model", self.model),
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                }
            except Exception as exc:
                error = exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"TopoNader LLM request failed: {error}")
