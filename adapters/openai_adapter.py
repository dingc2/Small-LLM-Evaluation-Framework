"""
OpenAI-compatible adapter.

Works with:
  - OpenAI API (gpt-4o, gpt-4-turbo, …)
  - Any OpenAI-compatible server: vLLM, Ollama, LM Studio, Together AI, etc.
    Just pass ``base_url`` pointing at the server's /v1 endpoint.

Example
-------
>>> adapter = OpenAIAdapter(model="gpt-4o", api_key="sk-...")
>>> response = await adapter.generate("What is 2+2?")
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from .base import AdapterError, ModelAdapter, ModelResponse, ToolCall, ToolDefinition


class OpenAIAdapter(ModelAdapter):
    """
    Adapter for OpenAI-compatible chat completion APIs.

    Parameters
    ----------
    model:
        Model identifier string (e.g. ``"gpt-4o"``).
    api_key:
        API key.  Falls back to ``OPENAI_API_KEY`` env var when ``None``.
    base_url:
        Override the API base URL for self-hosted / alternative providers.
    default_kwargs:
        Default generation parameters forwarded to every call
        (e.g. ``{"temperature": 0.0, "max_tokens": 1024}``).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIAdapter.  "
                "Install it with: pip install openai"
            ) from exc

        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._default_kwargs: dict[str, Any] = default_kwargs or {}

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        tools: Optional[list[ToolDefinition]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: dict[str, Any] = {**self._default_kwargs, **kwargs}

        # GPT-5.x and reasoning-tier models (o1/o3/o4) renamed max_tokens → max_completion_tokens
        if self._model.startswith(("gpt-5", "o1", "o3", "o4")) and "max_tokens" in call_kwargs:
            call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")

        openai_tools: Optional[list[dict[str, Any]]] = None
        if tools:
            openai_tools = [t.to_openai_spec() for t in tools]

        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=openai_tools,
                **call_kwargs,
            )
        except Exception as exc:
            raise AdapterError(f"OpenAI API call failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = response.choices[0]
        content: str = choice.message.content or ""

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = response.usage
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model_name=self._model,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )
