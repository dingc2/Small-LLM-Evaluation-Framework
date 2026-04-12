"""
End-to-end task-completion benchmark.

Measures: does the model, when given access to skills as tools, produce the
correct final answer?

Protocol
--------
1. Present each test case as a user query with all enabled skills injected as
   tool definitions.
2. If the model returns a tool call, execute it via the SkillRegistry and
   feed the result back (single tool-call round trip; multi-turn is
   configurable via ``max_turns``).
3. Parse the final textual answer and compare it to the expected value.
4. Scoring: exact numeric match (with tolerance for floats), or a custom
   scorer provided per-test-case.

The benchmark includes built-in calculator tasks pulled from
``SKILL_META["examples"]``.  Custom cases can be injected via ``extra_cases``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Any, Callable, Optional

from .base import Benchmark, BenchmarkResult, TestCase, TestResult

logger = logging.getLogger(__name__)

# How to ask the model to produce a final numeric/text answer
_SYSTEM_PROMPT = """\
You are a helpful assistant. You have access to tools that can help you
answer the user's question. Use them when appropriate.

After using a tool (if needed), provide your final answer as a plain number
or short string — nothing else. Do not include units or extra explanation.
Do not explain your reasoning. Output ONLY the answer.
"""

# Built-in test cases — organised by skill
_BUILTIN_CASES: list[dict[str, Any]] = [
    # --- Calculator ---
    {"id": "e2e_calc_01", "prompt": "What is 17 * 23?",
     "expected": 391, "skill": "calculator", "expression": "17 * 23"},
    {"id": "e2e_calc_02", "prompt": "Calculate sqrt(625)",
     "expected": 25.0, "skill": "calculator", "expression": "sqrt(625)"},
    {"id": "e2e_calc_03", "prompt": "What is 2 ** 10?",
     "expected": 1024, "skill": "calculator", "expression": "2 ** 10"},
    {"id": "e2e_calc_04", "prompt": "Compute (55 + 45) / 4",
     "expected": 25.0, "skill": "calculator", "expression": "(55 + 45) / 4"},
    {"id": "e2e_calc_05", "prompt": "What is sin(0)?",
     "expected": 0.0, "skill": "calculator", "expression": "sin(0)"},
    {"id": "e2e_calc_06", "prompt": "Round pi to 4 decimal places (compute pi)",
     "expected": math.pi, "skill": "calculator", "expression": "pi",
     "tolerance": 0.0001},

    # --- Unit Converter ---
    {"id": "e2e_conv_01", "prompt": "Convert 5 km to miles",
     "expected": 3.10686, "skill": "unit_converter", "tolerance": 0.001},
    {"id": "e2e_conv_02", "prompt": "Convert 100 F to C",
     "expected": 37.7778, "skill": "unit_converter", "tolerance": 0.01},
    {"id": "e2e_conv_03", "prompt": "Convert 1 kg to lb",
     "expected": 2.20462, "skill": "unit_converter", "tolerance": 0.001},
    {"id": "e2e_conv_04", "prompt": "Convert 1 gal to L",
     "expected": 3.78541, "skill": "unit_converter", "tolerance": 0.001},
    {"id": "e2e_conv_05", "prompt": "Convert 12 inches to cm",
     "expected": 30.48, "skill": "unit_converter", "tolerance": 0.01},

    # --- Dictionary ---
    {"id": "e2e_dict_01", "prompt": "Define the word ephemeral",
     "expected": "lasting for a very short time", "skill": "dictionary"},
    {"id": "e2e_dict_02", "prompt": "What does algorithm mean?",
     "expected": "a process or set of rules to be followed in calculations or other problem-solving operations",
     "skill": "dictionary"},
    {"id": "e2e_dict_03", "prompt": "Define quantization",
     "expected": "the process of reducing the precision of a model's weights to decrease memory usage and increase speed",
     "skill": "dictionary"},

    # --- Date/Time Calculator ---
    {"id": "e2e_date_01", "prompt": "How many days between 2024-01-01 and 2024-12-31?",
     "expected": 365, "skill": "datetime_calc"},
    {"id": "e2e_date_02", "prompt": "What day of the week is 2024-07-04?",
     "expected": "Thursday", "skill": "datetime_calc"},
    {"id": "e2e_date_03", "prompt": "Add 30 days to 2024-01-15",
     "expected": "2024-02-14", "skill": "datetime_calc"},
    {"id": "e2e_date_04", "prompt": "How many days between 2024-03-01 and 2024-03-15?",
     "expected": 14, "skill": "datetime_calc"},

    # --- No-tool baselines ---
    {"id": "e2e_no_tool_01", "prompt": "Say exactly: hello",
     "expected": "hello", "skill": None},
    {"id": "e2e_no_tool_02", "prompt": "What is 2 + 2? Reply with just the number.",
     "expected": 4, "skill": None},
]

# Default float comparison tolerance
_DEFAULT_TOLERANCE = 1e-6


def _default_scorer(actual: Any, expected: Any, tolerance: float = _DEFAULT_TOLERANCE) -> float:
    """
    Score a model answer vs. expected value.
    Returns 1.0 for match, 0.0 otherwise.
    Handles int/float with tolerance, and string keyword-overlap for long strings.
    """
    if actual is None:
        return 0.0

    # Numeric comparison
    try:
        actual_f = float(actual)
        expected_f = float(expected)
        return 1.0 if math.isclose(actual_f, expected_f, abs_tol=tolerance, rel_tol=tolerance) else 0.0
    except (TypeError, ValueError):
        pass

    actual_s = str(actual).strip().lower()
    expected_s = str(expected).strip().lower()

    # Exact string match (for short expected values like "hello", "Thursday")
    if actual_s == expected_s:
        return 1.0

    # Keyword overlap for longer expected strings (e.g. dictionary definitions)
    # A model won't output a definition verbatim, so check if key terms are present
    expected_words = set(re.findall(r"\w{3,}", expected_s))  # words with 3+ chars
    if len(expected_words) >= 3:
        actual_words = set(re.findall(r"\w{3,}", actual_s))
        overlap = len(expected_words & actual_words)
        ratio = overlap / len(expected_words)
        return 1.0 if ratio >= 0.6 else 0.0

    # Fallback: exact match for short strings
    return 1.0 if actual_s == expected_s else 0.0


def _extract_number(text: str) -> Optional[float]:
    """Pull the first number out of a string (handles negatives and decimals)."""
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    return float(match.group()) if match else None


class EndToEndBenchmark(Benchmark):
    """
    Benchmark: full pipeline from query → (optional tool call) → final answer.

    Parameters
    ----------
    extra_cases:
        Additional test-case dicts to include.
    max_turns:
        Maximum tool-call round trips per test case.
    temperature:
        Generation temperature.
    float_tolerance:
        Absolute tolerance for numeric comparisons.
    custom_scorer:
        Optional callable ``(actual, expected) -> float`` for all cases
        that don't define their own scorer.
    """

    name = "end_to_end_task_completion"
    description = (
        "Measures whether the model correctly completes tasks end-to-end, "
        "optionally using skill tools to produce the right final answer."
    )

    def __init__(
        self,
        extra_cases: Optional[list[dict[str, Any]]] = None,
        max_turns: int = 3,
        temperature: float = 0.0,
        float_tolerance: float = _DEFAULT_TOLERANCE,
        custom_scorer: Optional[Callable[[Any, Any], float]] = None,
    ) -> None:
        self._extra_cases = extra_cases or []
        self._max_turns = max_turns
        self._temperature = temperature
        self._float_tolerance = float_tolerance
        self._scorer = custom_scorer or (
            lambda a, e: _default_scorer(a, e, float_tolerance)
        )

    # ------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------

    async def run(
        self,
        model: Any,
        skills: Optional[Any] = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        skill_names = skills.names if skills else []
        result, start_time = self._make_result(
            model_name=model.model_name,
            benchmark_name=self.name,
            skill_config=skill_names,
        )

        test_cases = self._build_test_cases(skills)
        if not test_cases:
            logger.warning("EndToEndBenchmark: no test cases to run.")
            return self._close_result(result, start_time)

        tasks = [
            self._run_single(tc, model, skills, **kwargs)
            for tc in test_cases
        ]
        test_results = await asyncio.gather(*tasks, return_exceptions=True)

        for tc, tr in zip(test_cases, test_results):
            if isinstance(tr, Exception):
                result.test_results.append(
                    TestResult(
                        test_id=tc.id,
                        passed=False,
                        score=0.0,
                        error=str(tr),
                        expected=tc.expected,
                    )
                )
            else:
                result.test_results.append(tr)

        return self._close_result(result, start_time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_test_cases(self, skills: Optional[Any]) -> list[TestCase]:
        cases = []
        for c in _BUILTIN_CASES:
            # Skip skill-dependent cases when no skills are available
            # (otherwise we'd be testing raw reasoning, not tool-use uplift)
            if c.get("skill") and (skills is None or c["skill"] not in skills):
                continue
            cases.append(
                TestCase(
                    id=c["id"],
                    prompt=c["prompt"],
                    expected=c["expected"],
                    metadata={k: v for k, v in c.items()
                               if k not in ("id", "prompt", "expected")},
                )
            )
        for c in self._extra_cases:
            cases.append(
                TestCase(
                    id=c["id"],
                    prompt=c["prompt"],
                    expected=c["expected"],
                    metadata=c.get("metadata", {}),
                    weight=c.get("weight", 1.0),
                )
            )
        return cases

    async def _run_single(
        self,
        tc: TestCase,
        model: Any,
        skills: Optional[Any],
        **kwargs: Any,
    ) -> TestResult:
        from eval_framework.adapters.base import AdapterError
        from eval_framework.skills.registry import SkillInput

        tool_defs = []
        if skills:
            tool_defs = [s.to_tool_definition() for s in skills.all()]

        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        final_content = ""
        actual_value: Any = None
        error_msg: Optional[str] = None

        prompt = tc.prompt
        turns = 0

        while turns < self._max_turns:
            turns += 1
            try:
                response = await model.generate(
                    prompt=prompt,
                    tools=tool_defs if tool_defs else None,
                    system_prompt=_SYSTEM_PROMPT,
                    temperature=kwargs.get("temperature", self._temperature),
                )
            except AdapterError as exc:
                error_msg = str(exc)
                break

            total_latency += response.latency_ms
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens
            final_content = response.content

            # No tool call → final answer is in content
            if not response.has_tool_calls:
                break

            # Execute tool calls
            tool_outputs: list[str] = []
            for tc_call in response.tool_calls:
                if skills and tc_call.name in skills:
                    skill = skills.get(tc_call.name)
                    args = tc_call.arguments
                    query = args.get("query", prompt)
                    expression = args.get("expression", args.get("query", prompt))
                    si = SkillInput(
                        query=query,
                        parameters={"expression": expression},
                    )
                    skill_out = await skill.execute(si)
                    if skill_out.success:
                        tool_outputs.append(
                            f"Tool '{tc_call.name}' returned: {skill_out.result}"
                        )
                        actual_value = skill_out.result
                    else:
                        tool_outputs.append(
                            f"Tool '{tc_call.name}' error: {skill_out.error}"
                        )
                else:
                    tool_outputs.append(
                        f"Tool '{tc_call.name}' is not available."
                    )

            # Feed tool output back as the new prompt for the next turn
            prompt = "\n".join(tool_outputs) + "\n\nNow provide your final answer."

        # Strip thinking blocks before parsing final content
        # Handles both closed (<think>...</think>) and unclosed (<think>...) tags
        if final_content:
            import re as _re
            # Remove properly closed blocks
            final_content_clean = _re.sub(
                r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>", "",
                final_content, flags=_re.DOTALL | _re.IGNORECASE
            ).strip()
            # Remove unclosed opening tags (model forgot to close)
            final_content_clean = _re.sub(
                r"<think(?:ing)?>.*$", "",
                final_content_clean, flags=_re.DOTALL | _re.IGNORECASE
            ).strip()
            if not final_content_clean:
                final_content_clean = final_content
        else:
            final_content_clean = final_content or ""

        # If no direct tool result, try to parse a number from the model's text
        if actual_value is None and final_content_clean:
            num = _extract_number(final_content_clean)
            if num is not None:
                actual_value = num
            else:
                actual_value = final_content_clean.strip()

        tolerance = tc.metadata.get("tolerance", self._float_tolerance)
        score = _default_scorer(actual_value, tc.expected, tolerance)
        passed = score >= 1.0

        return TestResult(
            test_id=tc.id,
            passed=passed,
            score=score,
            model_output=final_content,
            expected=tc.expected,
            actual=actual_value,
            latency_ms=total_latency,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            error=error_msg,
            metadata={"turns": turns},
        )
