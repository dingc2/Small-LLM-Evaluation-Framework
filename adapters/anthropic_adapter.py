"""
Anthropic adapter.

Works with:
  - Anthropic API (claude-haiku-4-5-20251001, claude-sonnet-*, …)
    Pass ``api_key`` or set the ``ANTHROPIC_API_KEY`` environment variable.
  - Optionally override ``base_url`` for proxy / custom endpoints.

Tool-format note
----------------
The framework passes tools as ``ToolDefinition`` objects (see ``base.py``).
``ToolDefinition.to_openai_spec()`` produces OpenAI-style dicts::

    {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}

This adapter converts those to Anthropic-style at the boundary via
``_convert_tools()``.  The no-tool path is fully supported and is the
primary use-case for the Claude Haiku baseline run.

Example
-------
>>> adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001", api_key="sk-ant-...")
>>> response = await adapter.generate("What is 2+2?")
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .base import AdapterError, ModelAdapter, ModelResponse, ToolCall, ToolDefinition


def _convert_tools(
    tools: list[ToolDefinition],
) -> list[dict[str, Any]]:
    """
    Convert a list of ``ToolDefinition`` objects to Anthropic tool dicts.

    Anthropic tool schema::

        {
            "name": "<name>",
            "description": "<description>",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...],
            },
        }
    """
    result: list[dict[str, Any]] = []
    for td in tools:
        result.append(
            {
                "name": td.name,
                "description": td.description,
                "input_schema": {
                    "type": "object",
                    "properties": td.parameters,
                    "required": td.required,
                },
            }
        )
    return result


class AnthropicAdapter(ModelAdapter):
    """
    Adapter for the Anthropic Messages API.

    Parameters
    ----------
    model:
        Model identifier string (e.g. ``"claude-haiku-4-5-20251001"``).
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var when ``None``.
    base_url:
        Override the API base URL (e.g. for proxies or Bedrock-compatible endpoints).
    timeout:
        HTTP request timeout in seconds (default 300).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter.  "
                "Install it with: pip install anthropic"
            ) from exc

        self._model = model

        client_kwargs: dict[str, Any] = {"timeout": timeout}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self._client = AsyncAnthropic(**client_kwargs)

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
        """
        Generate a response via the Anthropic Messages API.

        Parameters
        ----------
        prompt:
            The user-facing input text.
        tools:
            Optional list of ``ToolDefinition`` objects.  Converted to
            Anthropic-style tool dicts at the boundary.  Tool-calling is
            supported; the no-tool path is the primary use-case.
        system_prompt:
            Optional system-level instruction passed as Anthropic's top-level
            ``system`` parameter.
        **kwargs:
            Extra generation parameters forwarded to ``messages.create``
            (e.g. ``temperature``, ``max_tokens``).
        """
        from anthropic import NOT_GIVEN  # type: ignore

        temperature: float = kwargs.pop("temperature", 0.0)
        max_tokens: int = kwargs.pop("max_tokens", 1024)

        anthropic_tools = _convert_tools(tools) if tools else NOT_GIVEN

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": anthropic_tools,
            **kwargs,
        }
        # Only pass system if provided — Anthropic rejects an explicit None
        if system_prompt:
            create_kwargs["system"] = system_prompt

        start = time.perf_counter()
        try:
            response = await self._client.messages.create(**create_kwargs)
        except Exception as exc:
            raise AdapterError(f"Anthropic API call failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Anthropic returns a list of content blocks; flatten text + collect tool_use
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        content = "".join(text_parts)

        usage = response.usage
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            latency_ms=latency_ms,
            model_name=self._model,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )
