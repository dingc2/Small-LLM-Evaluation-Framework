"""
Base adapter interface.

All model adapters must subclass ``ModelAdapter`` and implement
``generate()``.  The rest of the framework only ever touches this
interface — swapping models is done by swapping adapters.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """Schema of a callable tool exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

    def to_openai_spec(self) -> dict[str, Any]:
        """Convert to the OpenAI tool-call JSON schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


class ToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Unified response from any model adapter."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class AdapterError(Exception):
    """Raised when a model adapter encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ModelAdapter(ABC):
    """
    Abstract base class for all model adapters.

    Subclasses must implement:
      - ``model_name`` property
      - ``generate()`` async method

    Nothing in the core framework depends on *which* adapter is used;
    only this interface matters.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """A unique, human-readable identifier for this model/adapter."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        tools: Optional[list[ToolDefinition]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a response for *prompt*.

        Parameters
        ----------
        prompt:
            The user-facing input text.
        tools:
            Optional list of tools the model may call.
        system_prompt:
            Optional system-level instruction.
        **kwargs:
            Adapter-specific generation parameters (temperature, max_tokens, …).

        Returns
        -------
        ModelResponse
            Unified response object with content, optional tool calls,
            token counts, and latency.

        Raises
        ------
        AdapterError
            On any unrecoverable failure (auth, timeout, parsing, …).
        """

    # ------------------------------------------------------------------
    # Convenience helpers (concrete, not required to override)
    # ------------------------------------------------------------------

    async def generate_with_retry(
        self,
        prompt: str,
        tools: Optional[list[ToolDefinition]] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> ModelResponse:
        """Call ``generate`` with exponential-backoff retries."""
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(max_retries):
            try:
                return await self.generate(
                    prompt, tools=tools, system_prompt=system_prompt, **kwargs
                )
            except AdapterError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
        raise AdapterError(
            f"Failed after {max_retries} retries: {last_exc}"
        ) from last_exc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
