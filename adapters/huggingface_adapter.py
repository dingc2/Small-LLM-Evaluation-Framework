"""
HuggingFace Transformers local adapter.

Runs a model entirely on-device using the ``transformers`` pipeline API.
Token counts are estimated from the tokenizer when available.

Example
-------
>>> adapter = HuggingFaceAdapter(model_id="microsoft/Phi-3-mini-4k-instruct")
>>> response = await adapter.generate("What is 2+2?")

Notes
-----
* Tool calling is implemented via a JSON-in-prompt technique: tool definitions
  are serialised into the system prompt and the model is asked to output a
  JSON object when it wants to call a tool.
* Generation runs in a ThreadPoolExecutor to stay non-blocking.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from .base import AdapterError, ModelAdapter, ModelResponse, ToolCall, ToolDefinition

_EXECUTOR = ThreadPoolExecutor(max_workers=2)

_TOOL_SYSTEM_PREFIX = """\
You have access to the following tools. To call a tool, respond ONLY with a
JSON object of this form (no extra text):
  {"tool": "<tool_name>", "arguments": {<key: value, ...>}}

Available tools:
{tool_specs}

If you do not need a tool, respond normally in plain text.
"""


class HuggingFaceAdapter(ModelAdapter):
    """
    Adapter wrapping a local HuggingFace Transformers text-generation pipeline.

    Parameters
    ----------
    model_id:
        HuggingFace Hub model identifier or local path.
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"``, or an integer device index.
    torch_dtype:
        Torch dtype string (``"float16"``, ``"bfloat16"``, ``"float32"``).
    max_new_tokens:
        Maximum tokens the model may generate per call.
    pipeline_kwargs:
        Extra keyword arguments forwarded to ``transformers.pipeline()``.
    """

    def __init__(
        self,
        model_id: str,
        device: str | int = "cpu",
        torch_dtype: str = "float32",
        max_new_tokens: int = 512,
        pipeline_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._pipe: Any = None  # lazy-initialised
        self._tokenizer: Any = None
        self._init_kwargs: dict[str, Any] = {
            "device": device,
            "torch_dtype": torch_dtype,
            **(pipeline_kwargs or {}),
        }

    # ------------------------------------------------------------------
    # Lazy initialisation (avoids loading the model until first call)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for HuggingFaceAdapter.  "
                "Install with: pip install transformers torch"
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(
            self._init_kwargs.pop("torch_dtype", "float32"), torch.float32
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._pipe = pipeline(
            "text-generation",
            model=self._model_id,
            tokenizer=self._tokenizer,
            torch_dtype=torch_dtype,
            **self._init_kwargs,
        )

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_id

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
            raise AdapterError(f"HuggingFace generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Synchronous internals (run in thread pool)
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
            tool_specs = json.dumps(
                [t.to_openai_spec()["function"] for t in tools], indent=2
            )
            full_system = _TOOL_SYSTEM_PREFIX.format(tool_specs=tool_specs) + full_system

        # Build chat messages
        messages: list[dict[str, str]] = []
        if full_system:
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": prompt})

        # Try chat template; fall back to plain concatenation
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = "\n".join(m["content"] for m in messages)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "return_full_text": False,
            **kwargs,
        }

        start = time.perf_counter()
        outputs = self._pipe(formatted, **gen_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        raw_text: str = outputs[0]["generated_text"] if outputs else ""

        # Token counts (best-effort)
        prompt_tokens = (
            len(self._tokenizer.encode(formatted)) if self._tokenizer else 0
        )
        completion_tokens = (
            len(self._tokenizer.encode(raw_text)) if self._tokenizer else 0
        )

        # Try to parse tool calls from JSON-in-text response
        tool_calls, content = self._parse_tool_calls(raw_text)

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model_name=self._model_id,
        )

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
        """
        Try to extract a JSON tool-call block from the model output.
        Returns ``(tool_calls, remaining_text)``.
        """
        text = text.strip()
        # Look for ```json ... ``` fences or bare JSON objects
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*\"tool\"\s*:.*?\})",
        ]
        for pattern in patterns:
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
