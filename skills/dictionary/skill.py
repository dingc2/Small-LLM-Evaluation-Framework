"""
Dictionary / definition lookup skill — returns definitions of words and terms.

Uses a curated built-in dictionary for deterministic, reproducible benchmarks.
This simulates a real dictionary API tool that the model must learn to invoke.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Skill metadata
# ---------------------------------------------------------------------------

SKILL_META = {
    "name": "dictionary",
    "description": (
        "Use for looking up what a word or term means. "
        "Returns the definition, part of speech, and example usage. "
        "Use this whenever the user asks about the meaning of a word."
    ),
    "trigger_patterns": [
        r"\bdefine\b",
        r"\bdefinition\b",
        r"\bmeaning\b",
        r"\bwhat\s+does\s+.+\s+mean\b",
        r"\bwhat\s+is\s+(?:a|an|the)\s+\w+\b",
        r"\blook\s*up\b.*\bword\b",
        r"\bdictionary\b",
    ],
    "version": "1.0.0",
    "author": "eval_framework",
    "examples": [
        {"input": "define ephemeral", "expected": "lasting for a very short time"},
        {"input": "what does 'ubiquitous' mean", "expected": "present, appearing, or found everywhere"},
        {"input": "definition of algorithm", "expected": "a process or set of rules to be followed in calculations or other problem-solving operations"},
    ],
}

# ---------------------------------------------------------------------------
# Built-in dictionary (for reproducible evaluation)
# ---------------------------------------------------------------------------

_DICTIONARY: dict[str, dict[str, str]] = {
    "ephemeral": {
        "definition": "lasting for a very short time",
        "part_of_speech": "adjective",
        "example": "The ephemeral beauty of cherry blossoms.",
    },
    "ubiquitous": {
        "definition": "present, appearing, or found everywhere",
        "part_of_speech": "adjective",
        "example": "Smartphones have become ubiquitous in modern life.",
    },
    "algorithm": {
        "definition": "a process or set of rules to be followed in calculations or other problem-solving operations",
        "part_of_speech": "noun",
        "example": "The sorting algorithm runs in O(n log n) time.",
    },
    "paradigm": {
        "definition": "a typical example or pattern of something; a model",
        "part_of_speech": "noun",
        "example": "The shift in programming paradigm from procedural to object-oriented.",
    },
    "optimize": {
        "definition": "to make the best or most effective use of a situation or resource",
        "part_of_speech": "verb",
        "example": "We need to optimize the code for better performance.",
    },
    "latency": {
        "definition": "the delay before a transfer of data begins following an instruction for its transfer",
        "part_of_speech": "noun",
        "example": "Network latency affects the user experience of real-time applications.",
    },
    "inference": {
        "definition": "a conclusion reached on the basis of evidence and reasoning",
        "part_of_speech": "noun",
        "example": "Model inference time is critical for production deployment.",
    },
    "hallucination": {
        "definition": "an experience involving the apparent perception of something not present",
        "part_of_speech": "noun",
        "example": "LLM hallucinations can produce plausible but incorrect information.",
    },
    "benchmark": {
        "definition": "a standard or point of reference against which things may be compared or assessed",
        "part_of_speech": "noun",
        "example": "We used MMLU as a benchmark for language model evaluation.",
    },
    "perplexity": {
        "definition": "a measurement of how well a probability model predicts a sample; lower is better",
        "part_of_speech": "noun",
        "example": "The model achieved a perplexity of 15.2 on the test set.",
    },
    "tokenization": {
        "definition": "the process of breaking text into smaller units called tokens for processing",
        "part_of_speech": "noun",
        "example": "Different tokenization strategies affect model vocabulary size.",
    },
    "quantization": {
        "definition": "the process of reducing the precision of a model's weights to decrease memory usage and increase speed",
        "part_of_speech": "noun",
        "example": "4-bit quantization reduced the model size by 75% with minimal quality loss.",
    },
    "transformer": {
        "definition": "a deep learning model architecture based on self-attention mechanisms",
        "part_of_speech": "noun",
        "example": "The transformer architecture revolutionized natural language processing.",
    },
    "gradient": {
        "definition": "a vector of partial derivatives that indicates the direction of steepest ascent",
        "part_of_speech": "noun",
        "example": "Gradient descent is the most common optimization algorithm in deep learning.",
    },
    "entropy": {
        "definition": "a measure of the uncertainty or randomness in a system or dataset",
        "part_of_speech": "noun",
        "example": "Higher entropy in the output distribution indicates less confident predictions.",
    },
}


def _extract_word(query: str) -> str:
    """Extract the word to look up from a natural language query."""
    query = query.strip().lower()

    # Remove common prefixes
    for prefix in [
        "define ", "definition of ", "what does ", "what is a ",
        "what is an ", "what is the ", "what is ",
        "look up ", "meaning of ", "the meaning of ",
    ]:
        if query.startswith(prefix):
            query = query[len(prefix):]
            break

    # Remove trailing punctuation and quotes
    query = re.sub(r"['\"\?\.\!]+$", "", query)
    query = re.sub(r"^['\"]|['\"]$", "", query)

    # Remove trailing "mean" from "what does X mean"
    query = re.sub(r"\s+mean$", "", query)

    # Take just the first word if multiple remain (after cleanup)
    words = query.strip().split()
    return words[0] if words else ""


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(input: Any) -> Any:
    """Look up a word definition."""
    if hasattr(input, "query"):
        query: str = input.query
        params: dict = getattr(input, "parameters", {})
    else:
        query = input.get("query", "")
        params = input.get("parameters", {})

    try:
        from eval_framework.skills.registry import SkillOutput
    except ImportError:
        SkillOutput = None

    try:
        word = params.get("word", _extract_word(query)).lower().strip()

        if word in _DICTIONARY:
            entry = _DICTIONARY[word]
            result = entry["definition"]
            meta = {
                "word": word,
                "part_of_speech": entry["part_of_speech"],
                "example": entry["example"],
                "full_entry": entry,
            }
            if SkillOutput:
                return SkillOutput(result=result, success=True, metadata=meta)
            return {"result": result, "success": True, "error": None, "metadata": meta}
        else:
            error = f"Word '{word}' not found in dictionary. Available: {', '.join(sorted(_DICTIONARY.keys()))}"
            if SkillOutput:
                return SkillOutput.failure(error, metadata={"word": word})
            return {"result": None, "success": False, "error": error, "metadata": {"word": word}}

    except Exception as exc:
        if SkillOutput:
            return SkillOutput.failure(str(exc))
        return {"result": None, "success": False, "error": str(exc), "metadata": {}}
