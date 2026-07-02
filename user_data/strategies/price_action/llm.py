"""LLM clients with a shared JSON-completion interface.

Two protocol backends are supported:

* :class:`OpenAIJsonClient` – OpenAI ``/v1/chat/completions`` protocol.
* :class:`AnthropicJsonClient` – Anthropic ``/v1/messages`` protocol.

Both implement the :class:`LlmClient` protocol so the orchestrator can use
either transparently. :func:`build_llm_client` routes by the
``pa_llm_protocol`` config key (or ``PA_LLM_PROTOCOL`` env var).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol, runtime_checkable


class LlmConfigurationError(RuntimeError):
    """Raised when LLM credentials or SDK are not available."""


@runtime_checkable
class LlmClient(Protocol):
    """Minimal contract shared by all LLM backends.

    The orchestrator and monitor only depend on these attributes/methods.
    """

    base_url: str
    model: str
    api_key: str | None
    last_response: dict[str, Any] | None

    def complete_json(self, messages: list[dict[str, str]], *, stage: str) -> str: ...


def _read_common_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract shared LLM config keys from config dict + environment.

    Both backends use the same keys so users can switch protocols by
    changing only ``pa_llm_protocol``.
    """
    api_key = (
        config.get("pa_llm_api_key")
        or os.environ.get(str(config.get("pa_llm_api_key_env") or "DEEPSEEK_API_KEY"))
        or os.environ.get("OPENAI_API_KEY")
    )
    return {
        "api_key": api_key,
        "base_url": (
            config.get("pa_llm_base_url")
            or os.environ.get("PA_LLM_BASE_URL")
            or "https://api.deepseek.com"
        ),
        "model": (
            config.get("pa_llm_model")
            or os.environ.get("PA_LLM_MODEL")
            or "deepseek-chat"
        ),
        "temperature": float(
            config.get("pa_llm_temperature")
            or os.environ.get("PA_LLM_TEMPERATURE")
            or 0.1
        ),
        "timeout": float(
            config.get("pa_llm_timeout") or os.environ.get("PA_LLM_TIMEOUT") or 60
        ),
        "max_tokens": _optional_int(
            config.get("pa_llm_max_tokens") or os.environ.get("PA_LLM_MAX_TOKENS")
        ),
    }


@dataclass
class OpenAIJsonClient:
    """Small wrapper around OpenAI-compatible chat completions."""

    base_url: str
    api_key: str | None
    model: str
    temperature: float = 0.1
    timeout: float = 60.0
    max_tokens: int | None = None
    client: Any | None = None
    last_response: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAIJsonClient":
        cfg = _read_common_config(config)
        return cls(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            temperature=cfg["temperature"],
            timeout=cfg["timeout"],
            max_tokens=cfg["max_tokens"],
        )

    def complete_json(self, messages: list[dict[str, str]], *, stage: str) -> str:
        """Call chat.completions with JSON-object response_format."""
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "timeout": self.timeout,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = client.chat.completions.create(
            **kwargs,
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
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return self.client


@dataclass
class AnthropicJsonClient:
    """Wrapper around the Anthropic ``/v1/messages`` protocol.

    The Anthropic API takes ``system`` as a separate top-level parameter
    rather than as a message. This client extracts the first ``role: system``
    message (the orchestrator/prompts always supply exactly one at index 0)
    and forwards it via the ``system`` parameter, passing the remaining
    user/assistant turns as ``messages``.
    """

    base_url: str
    api_key: str | None
    model: str
    temperature: float = 0.1
    timeout: float = 60.0
    max_tokens: int | None = None
    client: Any | None = None
    last_response: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AnthropicJsonClient":
        cfg = _read_common_config(config)
        return cls(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            temperature=cfg["temperature"],
            timeout=cfg["timeout"],
            max_tokens=cfg["max_tokens"],
        )

    def complete_json(self, messages: list[dict[str, str]], *, stage: str) -> str:
        """Call messages.create and return the text content.

        JSON-object enforcement relies on prompt instructions and the
        ``parse_json_object`` post-processor in validation.py (Anthropic has
        no native ``response_format`` equivalent).
        """
        client = self._client()
        system_text, conversation = _split_system_message(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if system_text:
            kwargs["system"] = system_text
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = client.messages.create(**kwargs)
        content = _extract_anthropic_text(response)
        self.last_response = _serialize_anthropic_response(
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
            from anthropic import Anthropic
        except Exception as exc:
            raise LlmConfigurationError("anthropic SDK is not installed") from exc
        # Anthropic SDK builds the endpoint as {base_url}/v1/messages. The shared
        # config commonly stores ``https://host/v1`` (OpenAI-style); strip the
        # trailing ``/v1`` so we don't hit ``/v1/v1/messages`` on the gateway.
        anthropic_base_url = self.base_url.rstrip("/")
        if anthropic_base_url.endswith("/v1"):
            anthropic_base_url = anthropic_base_url[: -len("/v1")]
        self.client = Anthropic(
            base_url=anthropic_base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return self.client


def build_llm_client(config: dict[str, Any]) -> LlmClient:
    """Factory that selects a backend by ``pa_llm_protocol`` / ``PA_LLM_PROTOCOL``.

    Accepted values: ``"anthropic"``/``"claude"`` → Anthropic protocol;
    ``"openai"``/``None`` → OpenAI protocol (default).
    """
    protocol = (
        config.get("pa_llm_protocol")
        or os.environ.get("PA_LLM_PROTOCOL")
        or "openai"
    ).strip().lower()
    if protocol in ("anthropic", "claude"):
        return AnthropicJsonClient.from_config(config)
    return OpenAIJsonClient.from_config(config)


def _split_system_message(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    """Pull the first ``role: system`` message out; return (system, rest).

    The rest preserves original order of non-system messages. If multiple
    system messages occur (not currently produced by prompts.py), only the
    first is extracted and the others are dropped, matching Anthropic's
    single-system-parameter contract.
    """
    system_text: str | None = None
    rest: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") == "system" and system_text is None:
            system_text = message.get("content", "")
            continue
        rest.append(message)
    return system_text, rest


def _extract_anthropic_text(response: Any) -> str:
    """Concatenate text blocks from an Anthropic message response."""
    try:
        blocks = getattr(response, "content", None) or []
        parts: list[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    except Exception as exc:
        raise RuntimeError(f"LLM response missing content: {exc}") from exc


def _serialize_anthropic_response(
    response: Any,
    *,
    stage: str,
    model: str,
    content: str,
) -> dict[str, Any]:
    """Map an Anthropic response to the OpenAI-shaped raw-record dict.

    This lets orchestrator's ``_capture_llm_response`` / persistence
    consume both backends uniformly.
    """
    usage_obj = getattr(response, "usage", None)
    usage = _anthropic_usage_to_dict(usage_obj)
    raw: dict[str, Any] = {
        "stage": stage,
        "model": getattr(response, "model", None) or model,
        "content": content,
        "usage": usage,
    }
    response_id = getattr(response, "id", None)
    if response_id is not None:
        raw["id"] = response_id
    role = getattr(response, "role", None)
    if role is not None:
        raw["role"] = role
    return raw


def _anthropic_usage_to_dict(usage: Any) -> dict[str, Any]:
    """Convert Anthropic usage object to OpenAI-shaped dict.

    ``input_tokens`` → ``prompt_tokens``
    ``output_tokens`` → ``completion_tokens``
    ``cache_read_input_tokens`` → ``cached_prompt_tokens``
    ``total_tokens`` is computed for parity with OpenAI.
    """
    if usage is None:
        return {}
    if isinstance(usage, dict):
        result = dict(usage)
    elif hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        result = dict(dumped) if isinstance(dumped, dict) else {}
    elif hasattr(usage, "__dict__"):
        result = {
            key: value
            for key, value in vars(usage).items()
            if not key.startswith("_")
        }
    else:
        result = {}
    input_tokens = result.get("input_tokens")
    output_tokens = result.get("output_tokens")
    normalized: dict[str, Any] = {}
    if isinstance(input_tokens, (int, float)):
        normalized["prompt_tokens"] = int(input_tokens)
    if isinstance(output_tokens, (int, float)):
        normalized["completion_tokens"] = int(output_tokens)
    if (
        isinstance(input_tokens, (int, float))
        and isinstance(output_tokens, (int, float))
    ):
        normalized["total_tokens"] = int(input_tokens + output_tokens)
    cache_read = result.get("cache_read_input_tokens")
    if isinstance(cache_read, (int, float)):
        normalized["cached_prompt_tokens"] = int(cache_read)
    return normalized


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


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
