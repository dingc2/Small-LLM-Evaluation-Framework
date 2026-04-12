"""
Ollama adapter — optimised for local model inference via Ollama.

Ollama exposes an OpenAI-compatible API at ``http://localhost:11434/v1``,
but this adapter talks to Ollama's *native* ``/api/chat`` endpoint which
provides better tool-calling support and model management.

Prerequisites
-------------
Install Ollama from https://ollama.com and pull the model you want:

    ollama pull gemma3:4b
    ollama pull llama3.2:3b
    ollama pull qwen3:8b

Example
-------
>>> adapter = OllamaAdapter(model="gemma3:4b")
>>> response = await adapter.generate("What is 2+2?")
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from .base import AdapterError, ModelAdapter, ModelResponse, ToolCall, ToolDefinition


class OllamaAdapter(ModelAdapter):
    """
    Adapter for Ollama's native chat API.

    Parameters
    ----------
    model:
        Ollama model tag (e.g. ``"gemma3:4b"``, ``"llama3.2:3b"``).
    host:
        Ollama server URL.  Defaults to ``http://localhost:11434``.
    default_kwargs:
        Default generation parameters (``temperature``, ``num_predict``, etc.).
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        host: str = "http://localhost:11434",
        default_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._default_kwargs: dict[str, Any] = default_kwargs or {}

        # Lazy-import httpx for async HTTP
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError(
                "httpx is required for OllamaAdapter.  "
                "Install it with: pip install httpx"
            )

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
        import httpx

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request body
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {**self._default_kwargs, **kwargs},
        }

        # Add tools if provided (Ollama native tool format)
        if tools:
            body["tools"] = [self._tool_to_ollama_spec(t) for t in tools]

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self._host}/api/chat",
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise AdapterError(
                f"Ollama API returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.ConnectError as exc:
            raise AdapterError(
                f"Cannot connect to Ollama at {self._host}. "
                "Is Ollama running?  Start it with: ollama serve"
            ) from exc
        except Exception as exc:
            raise AdapterError(f"Ollama API call failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Parse response
        message = data.get("message", {})
        content: str = message.get("content", "")

        tool_calls: list[ToolCall] = []
        if message.get("tool_calls"):
            for i, tc in enumerate(message["tool_calls"]):
                fn = tc.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", {}),
                    )
                )

        # Token counts from Ollama response
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model_name=self._model,
            raw=data,
        )

    @staticmethod
    def _tool_to_ollama_spec(tool: ToolDefinition) -> dict[str, Any]:
        """Convert a ToolDefinition to Ollama's native tool format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": tool.required,
                },
            },
        }

    async def list_local_models(self) -> list[dict[str, Any]]:
        """List models available in the local Ollama instance."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return data.get("models", [])
        except Exception:
            return []

    async def check_health(self) -> bool:
        """Check if Ollama server is reachable."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self._host)
                return resp.status_code == 200
        except Exception:
            return False
