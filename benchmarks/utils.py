"""Shared utilities for benchmark implementations."""

from __future__ import annotations

import re


def strip_think_tags(text: str) -> str:
    """Strip <think>/<thinking> blocks from model output.

    Performs two passes:
    1. Remove properly closed blocks: <think>...</think>
    2. Remove unclosed opening tags to end of string (model hit token limit)

    Returns the stripped text (may be empty if all content was inside tags).
    Does NOT fall back to the original — callers decide what to do with empty output.
    """
    # Pass 1: closed blocks
    cleaned = re.sub(
        r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    # Pass 2: unclosed opening tag (model forgot to close or hit token limit)
    cleaned = re.sub(
        r"<think(?:ing)?>.*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    return cleaned


def recover_answer_from_think_block(
    raw_output: str,
    known_tokens: list[str],
) -> str | None:
    """Scan raw model output for a known answer token when stripping produces empty.

    Used when the entire response is inside a think block (the model either forgot
    to close the tag or hit the token limit before emitting a final answer).
    Models almost always state their decision somewhere in the reasoning trace.

    Returns the last-occurring known token (the model's final decision), or None
    if no known token is found anywhere in the output.
    """
    raw_lower = raw_output.lower()
    best: str | None = None
    best_pos = -1
    for tok in known_tokens:
        pos = raw_lower.rfind(tok.lower())
        if pos > best_pos:
            best_pos = pos
            best = tok
    return best
