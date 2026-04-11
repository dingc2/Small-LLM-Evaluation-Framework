"""
Unit tests for the adapter layer.

Tests cover:
  - ModelResponse / ToolDefinition data models
  - The ABC interface (cannot instantiate directly)
  - A MockAdapter that fulfils the interface
  - Retry logic in generate_with_retry
  - ToolDefinition.to_openai_spec output shape
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, Optional

from eval_framework.adapters.base import (
    AdapterError,
    ModelAdapter,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class MockAdapter(ModelAdapter):
    """Minimal concrete adapter for testing — always returns a canned response."""

    def __init__(
        self,
        response_text: str = "42",
        fail_times: int = 0,
    ) -> None:
        self._response_text = response_text
        self._fail_times = fail_times
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-model-v1"

    async def generate(
        self,
        prompt: str,
        tools: Optional[list[ToolDefinition]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise AdapterError(f"Simulated failure #{self._call_count}")
        return ModelResponse(
            content=self._response_text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self._response_text.split()),
            latency_ms=12.5,
            model_name=self.model_name,
        )


class ToolCallingMockAdapter(ModelAdapter):
    """Mock that returns a tool call for any prompt containing 'calculate'."""

    @property
    def model_name(self) -> str:
        return "tool-calling-mock"

    async def generate(
        self,
        prompt: str,
        tools: Optional[list[ToolDefinition]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if "calculate" in prompt.lower() and tools:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_001",
                        name=tools[0].name,
                        arguments={"query": prompt},
                    )
                ],
                model_name=self.model_name,
                latency_ms=5.0,
            )
        return ModelResponse(
            content="I don't know.",
            model_name=self.model_name,
            latency_ms=5.0,
        )


# ---------------------------------------------------------------------------
# Tests: ModelAdapter ABC
# ---------------------------------------------------------------------------


def test_cannot_instantiate_abstract_adapter():
    with pytest.raises(TypeError):
        ModelAdapter()  # type: ignore[abstract]


def test_mock_adapter_is_valid_subclass():
    adapter = MockAdapter()
    assert isinstance(adapter, ModelAdapter)


def test_repr():
    adapter = MockAdapter()
    assert "MockAdapter" in repr(adapter)
    assert "mock-model-v1" in repr(adapter)


# ---------------------------------------------------------------------------
# Tests: generate()
# ---------------------------------------------------------------------------


def test_generate_returns_model_response():
    adapter = MockAdapter(response_text="Hello world")
    result = asyncio.run(adapter.generate("Say hello"))
    assert isinstance(result, ModelResponse)
    assert result.content == "Hello world"
    assert result.model_name == "mock-model-v1"


def test_generate_records_latency():
    adapter = MockAdapter()
    result = asyncio.run(adapter.generate("test"))
    assert result.latency_ms == 12.5


def test_generate_token_counts():
    adapter = MockAdapter(response_text="forty two")
    result = asyncio.run(adapter.generate("What is the answer?"))
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens


# ---------------------------------------------------------------------------
# Tests: ModelResponse properties
# ---------------------------------------------------------------------------


def test_model_response_has_tool_calls_false_when_empty():
    r = ModelResponse(content="hi")
    assert r.has_tool_calls is False


def test_model_response_has_tool_calls_true():
    r = ModelResponse(
        content="",
        tool_calls=[ToolCall(name="calc", arguments={"query": "2+2"})],
    )
    assert r.has_tool_calls is True


def test_model_response_total_tokens():
    r = ModelResponse(prompt_tokens=10, completion_tokens=5)
    assert r.total_tokens == 15


# ---------------------------------------------------------------------------
# Tests: ToolDefinition
# ---------------------------------------------------------------------------


def test_tool_definition_to_openai_spec():
    td = ToolDefinition(
        name="calculator",
        description="Evaluates math expressions.",
        parameters={"query": {"type": "string", "description": "Expression"}},
        required=["query"],
    )
    spec = td.to_openai_spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "calculator"
    assert "query" in spec["function"]["parameters"]["properties"]
    assert spec["function"]["parameters"]["required"] == ["query"]


def test_tool_definition_empty_parameters():
    td = ToolDefinition(name="ping", description="Ping")
    spec = td.to_openai_spec()
    assert spec["function"]["parameters"]["properties"] == {}


# ---------------------------------------------------------------------------
# Tests: generate_with_retry()
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_transient_failures():
    adapter = MockAdapter(response_text="OK", fail_times=2)
    result = asyncio.run(
        adapter.generate_with_retry("prompt", max_retries=3, retry_delay=0.0)
    )
    assert result.content == "OK"
    assert adapter._call_count == 3


def test_retry_raises_after_max_retries_exceeded():
    adapter = MockAdapter(fail_times=10)
    with pytest.raises(AdapterError):
        asyncio.run(
            adapter.generate_with_retry("prompt", max_retries=2, retry_delay=0.0)
        )


def test_retry_succeeds_on_first_try():
    adapter = MockAdapter(response_text="immediate")
    result = asyncio.run(
        adapter.generate_with_retry("p", max_retries=3, retry_delay=0.0)
    )
    assert result.content == "immediate"
    assert adapter._call_count == 1


# ---------------------------------------------------------------------------
# Tests: tool-calling adapter
# ---------------------------------------------------------------------------


def test_tool_calling_adapter_returns_tool_call():
    adapter = ToolCallingMockAdapter()
    tools = [ToolDefinition(name="calculator", description="Math")]
    result = asyncio.run(adapter.generate("Please calculate 2+2", tools=tools))
    assert result.has_tool_calls
    assert result.tool_calls[0].name == "calculator"


def test_tool_calling_adapter_no_tool_without_keyword():
    adapter = ToolCallingMockAdapter()
    result = asyncio.run(adapter.generate("What is the weather?"))
    assert not result.has_tool_calls
