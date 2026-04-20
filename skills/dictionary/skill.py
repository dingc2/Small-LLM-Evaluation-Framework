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
    "author": "sLLM_eval_framework",
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


# Words we strip when hunting for the target word inside a noisy query.
# Anything *functional* for the user's intent (define / mean / word / …)
# plus a small set of polite fillers. Kept as a set for O(1) membership.
_STOPWORDS: frozenset[str] = frozenset({
    "define", "definition", "of", "what", "whats", "does", "is", "mean",
    "means", "meaning", "the", "a", "an", "look", "up", "can", "you",
    "please", "tell", "me", "word", "term", "about", "for",
    "in", "dictionary", "like", "to",
})


def _extract_word(query: str) -> str:
    """Extract the best candidate word from a natural-language query.

    Priority order — we try each strategy and return on the first that
    yields a non-empty candidate. This generalises past the old
    "strip-prefixes-at-start-only" approach which broke on polite wrappers
    ("Can you please define 'latency'") and double-quoted words.

    1. First quoted token (single, double, curly, or back-quotes) — the
       model's most explicit signal of which word matters.
    2. First non-stopword after stripping punctuation — handles "define X",
       "what does X mean", "the meaning of X", "X" (bare) alike.
    """
    q = query.strip()

    # 1. Quoted word: 'X' or "X" or ‘X’ / “X” / `X`
    m = re.search(r"['\"\u2018\u201c`]([^'\"\u2019\u201d`]+)['\"\u2019\u201d`]", q)
    if m:
        candidate = m.group(1).strip().lower()
        # Strip any internal punctuation that might have snuck in.
        candidate = re.sub(r"[^a-z0-9\-]", "", candidate)
        if candidate:
            return candidate

    # 2. First non-stopword token. Lower-case, drop all non-letters, then
    #    walk left-to-right ignoring stopwords.
    lowered = q.lower()
    tokens = re.findall(r"[a-z][a-z0-9\-]*", lowered)
    for tok in tokens:
        if tok not in _STOPWORDS:
            return tok

    # Nothing usable — fall back to the raw stripped string so the caller
    # surfaces a "word not found" error with the user's input visible.
    return re.sub(r"[^a-z0-9\-]", "", lowered)


# Simple English-plural → singular reductions. Not a full morphology
# library, just enough to catch the common case where a model says
# "algorithms" when it means "algorithm". Applied only as a fallback when
# the direct lookup fails.
_PLURAL_REDUCTIONS: list[tuple[str, str]] = [
    ("ies", "y"),   # babies → baby
    ("ses", "s"),   # bases → base (lossy but rarely wrong for our vocab)
    ("es", ""),     # boxes → box
    ("s", ""),      # cats → cat
]


def _singularize(word: str) -> str | None:
    """Return a plausible singular form of *word* if it looks plural.

    Returns None when no reduction applies (word is ≤2 chars or already
    singular-looking). The caller should try the reduced form against the
    dictionary and fall back to the original on miss.
    """
    if len(word) <= 2:
        return None
    for suffix, replacement in _PLURAL_REDUCTIONS:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: len(word) - len(suffix)] + replacement
    return None


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
        from sLLM_eval_framework.skills.registry import SkillOutput
    except ImportError:
        SkillOutput = None

    try:
        word = params.get("word", _extract_word(query)).lower().strip()

        # Try direct lookup, then a singular-form fallback for plurals
        # ("algorithms" → "algorithm"). The fallback only runs when the
        # direct lookup misses and the reduction yields a real word, so
        # we never silently mangle a successful lookup.
        matched = word if word in _DICTIONARY else None
        if matched is None:
            singular = _singularize(word)
            if singular and singular in _DICTIONARY:
                matched = singular

        if matched is not None:
            entry = _DICTIONARY[matched]
            result = entry["definition"]
            meta = {
                "word": matched,
                "original_query_word": word,
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
