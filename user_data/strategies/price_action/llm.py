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
    last_response: dict[str, Any] | None = None

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
            message = response.choices[0].message
            content = message.content or ""
        except Exception as exc:
            raise RuntimeError(f"LLM response missing content for {stage}") from exc
        self.last_response = _serialize_chat_response(
            response,
            stage=stage,
            model=self.model,
            content=content,
        )
        return content

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


def _serialize_chat_response(
    response: Any,
    *,
    stage: str,
    model: str,
    content: str,
) -> dict[str, Any]:
    """Return a compact PA_Agent-style raw response record."""
    message = None
    try:
        message = response.choices[0].message
    except Exception:
        message = None
    usage = _object_to_dict(getattr(response, "usage", None))
    raw = {
        "stage": stage,
        "model": getattr(response, "model", None) or model,
        "content": content,
        "usage": usage,
    }
    if message is not None:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is not None:
            raw["reasoning_content"] = reasoning
        role = getattr(message, "role", None)
        if role is not None:
            raw["role"] = role
    response_id = getattr(response, "id", None)
    if response_id is not None:
        raw["id"] = response_id
    return raw


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, dict) else {}
    if hasattr(value, "__dict__"):
        result: dict[str, Any] = {}
        for key, item in vars(value).items():
            if key.startswith("_"):
                continue
            if isinstance(item, (str, int, float, bool)) or item is None:
                result[key] = item
            else:
                nested = _object_to_dict(item)
                if nested:
                    result[key] = nested
        details = result.get("prompt_tokens_details")
        if isinstance(details, dict) and "cached_prompt_tokens" not in result:
            cached = details.get("cached_tokens")
            if cached is not None:
                result["cached_prompt_tokens"] = cached
        return result
    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        item = getattr(value, key, None)
        if item is not None:
            result[key] = item
    details = getattr(value, "prompt_tokens_details", None)
    details_dict = _object_to_dict(details)
    if details_dict:
        result["prompt_tokens_details"] = details_dict
        cached = details_dict.get("cached_tokens")
        if cached is not None and "cached_prompt_tokens" not in result:
            result["cached_prompt_tokens"] = cached
    return result
