"""Model adapter layer — import concrete adapters from here."""

from .base import ModelAdapter, ModelResponse, ToolDefinition, AdapterError
from .openai_adapter import OpenAIAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .llamacpp_adapter import LlamaCppAdapter

__all__ = [
    "ModelAdapter",
    "ModelResponse",
    "ToolDefinition",
    "AdapterError",
    "OpenAIAdapter",
    "HuggingFaceAdapter",
    "LlamaCppAdapter",
]
