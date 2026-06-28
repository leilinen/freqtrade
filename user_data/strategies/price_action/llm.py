"""OpenAI SDK-compatible JSON LLM client."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


class LlmConfigurationError(RuntimeError):
    """Raised when LLM credentials or SDK are not available."""


@dataclass
class OpenAIJsonClient:
    """Small wrapper around OpenAI-compatible chat completions."""

    base_url: str
    api_key: str | None
    model: str
    temperature: float = 0.1
    timeout: float = 60.0
    client: Any | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAIJsonClient":
        api_key = (
            config.get("pa_llm_api_key")
            or os.environ.get(str(config.get("pa_llm_api_key_env") or "DEEPSEEK_API_KEY"))
            or os.environ.get("OPENAI_API_KEY")
        )
        return cls(
            base_url=config.get("pa_llm_base_url") or os.environ.get("PA_LLM_BASE_URL")
            or "https://api.deepseek.com",
            api_key=api_key,
            model=config.get("pa_llm_model") or os.environ.get("PA_LLM_MODEL") or "deepseek-chat",
            temperature=float(config.get("pa_llm_temperature", 0.1)),
            timeout=float(config.get("pa_llm_timeout", 60)),
        )

    def complete_json(self, messages: list[dict[str, str]], *, stage: str) -> str:
        """Call chat.completions with JSON-object response_format."""
        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            timeout=self.timeout,
        )
        try:
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"LLM response missing content for {stage}") from exc

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise LlmConfigurationError(
                "pa_llm_api_key is not configured; set it in config or DEEPSEEK_API_KEY"
            )
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LlmConfigurationError("openai SDK is not installed") from exc
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self.client
