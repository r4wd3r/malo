"""Ollama client wrapper for chat completions."""

from __future__ import annotations

import time
from typing import Any, Iterable

from MaLO.utils.openrouter_client import ChatResponse
from MaLO.utils.weave_utils import get_weave
from openai import OpenAI
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion


class OllamaClient:
    """Ollama chat client with tool support."""

    def __init__(self, base_url: str | None = None) -> None:
        """Initializes the Ollama client.

        Args:
            base_url: Optional base URL override.
        """
        self._client = OpenAI(
            api_key="ollama",
            base_url=base_url or "http://localhost:11434/v1",
        )

    @get_weave().op()
    def generate(
        self,
        model: str,
        messages: Iterable[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        """Generates a chat completion with optional tool support.

        Args:
            model: Model identifier.
            messages: Chat messages.
            temperature: Sampling temperature.
            tools: Optional list of tool schemas.
            timeout: Optional timeout in seconds for the HTTP request.

        Returns:
            Structured chat response.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if timeout is not None:
            kwargs["timeout"] = timeout
        if seed is not None:
            kwargs["seed"] = seed

        max_attempts = 3
        last_response: ChatCompletion | None = None

        for attempt in range(max_attempts):
            try:
                response: ChatCompletion = self._client.chat.completions.create(
                    **kwargs
                )
            except (
                APIError,
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
            ) as e:
                raise RuntimeError(
                    f"Ollama API error for model '{model}': "
                    f"{type(e).__name__}: {str(e)}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error calling Ollama API for model "
                    f"'{model}': {type(e).__name__}: {str(e)}"
                ) from e

            if response is None:
                raise ValueError(
                    f"Ollama API returned None response for model '{model}'"
                )

            if response.choices is not None and response.choices:
                break

            last_response = response
            if attempt < max_attempts - 1:
                time.sleep(2)
        else:
            resp_id = getattr(last_response, "id", None) if last_response else None
            raise ValueError(
                f"Ollama API returned no choices for model '{model}' after "
                f"{max_attempts} attempt(s). Response id: {resp_id}. May indicate "
                f"API error, rate limiting, or malformed response."
            )

        choice = response.choices[0]
        message = choice.message

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        return ChatResponse(
            content=message.content,
            tool_calls=list(message.tool_calls) if message.tool_calls else [],
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
