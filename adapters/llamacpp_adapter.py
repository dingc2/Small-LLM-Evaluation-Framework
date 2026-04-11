"""
llama.cpp adapter (GGUF models).

Uses ``llama-cpp-python`` to run quantised GGUF models locally.
Supports GPU offloading via ``n_gpu_layers``.

Example
-------
>>> adapter = LlamaCppAdapter(
...     model_path="/models/llama-3-8b-instruct.Q4_K_M.gguf",
...     n_gpu_layers=32,
... )
>>> response = await adapter.generate("What is 2+2?")

Install
-------
    pip install llama-cpp-python
    # GPU build (CUDA):
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from .base import AdapterError, ModelAdapter, ModelResponse, ToolCall, ToolDefinition

_EXECUTOR = ThreadPoolExecutor(max_workers=1)  # llama.cpp is not thread-safe by default

_TOOL_SYSTEM_PREFIX = """\
You have access to the following tools. To use a tool, output ONLY a JSON
object on a single line (no markdown fences, no extra text):
  {{"tool": "<tool_name>", "arguments": {{<key: value>}}}}

Tools available:
{tool_specs}

If no tool is needed, respond in plain text.
"""


class LlamaCppAdapter(ModelAdapter):
    """
    Adapter wrapping ``llama_cpp.Llama`` for GGUF model inference.

    Parameters
    ----------
    model_path:
        Absolute path to the ``.gguf`` model file.
    n_ctx:
        Context window size (tokens).
    n_gpu_layers:
        Number of transformer layers to offload to GPU (0 = CPU only).
    chat_format:
        Chat template format string (e.g. ``"chatml"``, ``"llama-2"``).
        Pass ``None`` to let llama-cpp-python auto-detect.
    verbose:
        Whether to print llama.cpp progress messages.
    model_kwargs:
        Extra keyword arguments forwarded to ``Llama()``.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        chat_format: Optional[str] = None,
        verbose: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._chat_format = chat_format
        self._verbose = verbose
        self._extra_kwargs: dict[str, Any] = model_kwargs or {}
        self._llm: Any = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppAdapter.  "
                "Install with: pip install llama-cpp-python"
            ) from exc

        init_kwargs: dict[str, Any] = {
            "model_path": self._model_path,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "verbose": self._verbose,
            **self._extra_kwargs,
        }
        if self._chat_format:
            init_kwargs["chat_format"] = self._chat_format

        self._llm = Llama(**init_kwargs)

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        import os
        return os.path.basename(self._model_path)

    async def generate(
        self,
        prompt: str,
        tools: Optional[list[ToolDefinition]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                _EXECUTOR,
                lambda: self._generate_sync(prompt, tools, system_prompt, **kwargs),
            )
        except Exception as exc:
            raise AdapterError(f"llama.cpp generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Sync internals
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        prompt: str,
        tools: Optional[list[ToolDefinition]],
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> ModelResponse:
        self._load()

        full_system = system_prompt or ""
        if tools:
            specs = json.dumps(
                [t.to_openai_spec()["function"] for t in tools], indent=2
            )
            full_system = _TOOL_SYSTEM_PREFIX.format(tool_specs=specs) + full_system

        messages: list[dict[str, str]] = []
        if full_system:
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": prompt})

        gen_kwargs: dict[str, Any] = {
            "max_tokens": kwargs.pop("max_tokens", 512),
            "temperature": kwargs.pop("temperature", 0.1),
            **kwargs,
        }

        start = time.perf_counter()
        response = self._llm.create_chat_completion(messages=messages, **gen_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = response["choices"][0]
        raw_text: str = choice["message"].get("content") or ""

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        tool_calls, content = self._parse_tool_calls(raw_text)

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model_name=self.model_name,
            raw=response,
        )

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
        """Extract JSON tool-call from model output, same strategy as HF adapter."""
        text = text.strip()
        for pattern in [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*\"tool\"\s*:.*?\})",
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if "tool" in data:
                        tc = ToolCall(
                            name=data["tool"],
                            arguments=data.get("arguments", {}),
                        )
                        remaining = text[: match.start()] + text[match.end() :]
                        return [tc], remaining.strip()
                except json.JSONDecodeError:
                    pass
        return [], text
