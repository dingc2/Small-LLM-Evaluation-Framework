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

# Built-in test cases — organised by skill.
# Cases use non-trivial values that small LLMs are unlikely to have memorised
# (no perfect squares, no round conversions, no famous holidays, etc.).
_BUILTIN_CASES: list[dict[str, Any]] = [
    # --- Calculator ---
    {"id": "e2e_calc_01", "prompt": "What is 347 * 829?",
     "expected": 287663, "skill": "calculator", "expression": "347 * 829"},
    {"id": "e2e_calc_02", "prompt": "Compute 7 ** 5 - 9384",
     "expected": 7423, "skill": "calculator", "expression": "7 ** 5 - 9384"},
    {"id": "e2e_calc_03", "prompt": "What is sin(1.37) + cos(2.84)?",
     "expected": 0.02504344501898781, "skill": "calculator",
     "expression": "sin(1.37) + cos(2.84)", "tolerance": 0.001},
    {"id": "e2e_calc_04", "prompt": "What's the square root of 7291?",
     "expected": 85.38735269347563, "skill": "calculator",
     "expression": "sqrt(7291)", "tolerance": 0.01},
    {"id": "e2e_calc_05", "prompt": "Calculate (1247 + 3891) / 17.3",
     "expected": 296.9942196531792, "skill": "calculator",
     "expression": "(1247 + 3891) / 17.3", "tolerance": 0.01},
    {"id": "e2e_calc_06", "prompt": "Compute log2(4096) * 3.7 - sqrt(841)",
     "expected": 15.4, "skill": "calculator",
     "expression": "log2(4096) * 3.7 - sqrt(841)", "tolerance": 0.01},

    # --- Unit Converter ---
    {"id": "e2e_conv_01", "prompt": "Convert 38471 km to miles",
     "expected": 23904.771, "skill": "unit_converter", "tolerance": 1.0},
    {"id": "e2e_conv_02", "prompt": "Convert 41 F to C",
     "expected": 5.0, "skill": "unit_converter", "tolerance": 0.1},
    {"id": "e2e_conv_03", "prompt": "Convert 237 grams to ounces",
     "expected": 8.3599, "skill": "unit_converter", "tolerance": 0.01},
    {"id": "e2e_conv_04", "prompt": "Convert 3.7 liters to gallons",
     "expected": 0.9774, "skill": "unit_converter", "tolerance": 0.001},
    {"id": "e2e_conv_05", "prompt": "Convert 91.4 cm to inches",
     "expected": 35.984, "skill": "unit_converter", "tolerance": 0.01},

    # --- Dictionary ---
    {"id": "e2e_dict_01", "prompt": "What does the term 'perplexity' mean?",
     "expected": "a measurement of how well a probability model predicts a sample; lower is better",
     "skill": "dictionary"},
    {"id": "e2e_dict_02", "prompt": "Explain what 'tokenization' refers to",
     "expected": "the process of breaking text into smaller units called tokens for processing",
     "skill": "dictionary"},
    {"id": "e2e_dict_03", "prompt": "What is 'entropy' in the technical sense?",
     "expected": "a measure of the uncertainty or randomness in a system or dataset",
     "skill": "dictionary"},

    # --- Date/Time Calculator ---
    {"id": "e2e_date_01", "prompt": "How many days between 2024-02-17 and 2025-11-03?",
     "expected": 625, "skill": "datetime_calc"},
    {"id": "e2e_date_02", "prompt": "What day of the week was 1997-08-23?",
     "expected": "Saturday", "skill": "datetime_calc"},
    {"id": "e2e_date_03", "prompt": "Add 90 days to 2025-03-11",
     "expected": "2025-06-09", "skill": "datetime_calc"},
    {"id": "e2e_date_04", "prompt": "How many days between 2024-03-17 and 2024-10-09?",
     "expected": 206, "skill": "datetime_calc"},

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

    Comparison order:
    1. Both sides parse as floats → numeric comparison with tolerance.
       If ``actual`` is a string that won't parse directly, attempt to
       extract the *last* number it contains (handles outputs like
       "5 km is approximately 3.10686 miles.").
    2. Exact string match (covers short expected values like "hello",
       "Thursday", "2024-02-14").
    3. Keyword-overlap for longer definitions (60 % threshold on words ≥ 3
       chars) — only when expected has 3+ such words.
    """
    if actual is None:
        return 0.0

    # --- Numeric comparison ---
    # Try to get a float for both sides.  When ``actual`` is a string we
    # first attempt a direct conversion; if that fails we fall back to
    # extracting the *last* number from the string.  This handles model
    # outputs that include units or explanatory prose around the answer.
    try:
        expected_f = float(expected)
        # Try direct float conversion of actual first
        try:
            actual_f = float(actual)
        except (TypeError, ValueError):
            # Fall back: pull the last number embedded in the string
            num = _extract_last_number(str(actual))
            if num is None:
                raise ValueError("no number in actual")
            actual_f = num
        return 1.0 if math.isclose(actual_f, expected_f, abs_tol=tolerance, rel_tol=tolerance) else 0.0
    except (TypeError, ValueError):
        pass

    actual_s = str(actual).strip().lower()
    expected_s = str(expected).strip().lower()

    # --- Exact string match (covers short strings, dates, day-names …) ---
    if actual_s == expected_s:
        return 1.0

    # --- Keyword overlap for longer expected strings (e.g. definitions) ---
    # A model won't output a definition verbatim; check if key terms appear.
    expected_words = set(re.findall(r"\w{3,}", expected_s))  # words with 3+ chars
    if len(expected_words) >= 3:
        actual_words = set(re.findall(r"\w{3,}", actual_s))
        overlap = len(expected_words & actual_words)
        ratio = overlap / len(expected_words)
        return 1.0 if ratio >= 0.6 else 0.0

    # Fallback: exact match for short strings
    return 1.0 if actual_s == expected_s else 0.0


def _extract_last_number(text: str) -> Optional[float]:
    """
    Pull the *last* number out of a string (handles negatives and decimals).

    Using the last rather than the first number is more robust for model
    outputs like "5 km is approximately 3.10686 miles." where the answer
    appears at the end of an explanatory sentence.
    """
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text))
    return float(matches[-1].group()) if matches else None


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
            # Skip cases that require a skill not present in the registry.
            # When skills=None (no_skills baseline), this condition is always False
            # so ALL 20 cases run — the same population as all_skills, but without
            # tool definitions injected. This makes skill_delta a fair comparison.
            if c.get("skill") and skills is not None and c["skill"] not in skills:
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

        # Failure mode flags — tracked across turns for observability.
        # None = not applicable (e.g. no tool was expected / no tool was called).
        valid_tool_call_format: bool = False       # model produced a parseable tool call
        skill_selected_correctly: Optional[bool] = None  # named the right skill
        tool_executed_successfully: Optional[bool] = None  # skill ran without error

        expected_skill = tc.metadata.get("skill")  # None for no-tool baseline cases

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

            valid_tool_call_format = True

            # Execute tool calls
            tool_outputs: list[str] = []
            for tc_call in response.tool_calls:
                # Track whether the model selected the expected skill
                if expected_skill is not None and skill_selected_correctly is None:
                    skill_selected_correctly = tc_call.name == expected_skill

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
                        tool_executed_successfully = True
                        tool_outputs.append(
                            f"Tool '{tc_call.name}' returned: {skill_out.result}"
                        )
                        actual_value = skill_out.result
                    else:
                        tool_executed_successfully = False
                        tool_outputs.append(
                            f"Tool '{tc_call.name}' error: {skill_out.error}"
                        )
                else:
                    tool_executed_successfully = False
                    tool_outputs.append(
                        f"Tool '{tc_call.name}' is not available."
                    )

            # Feed tool output back as the new prompt for the next turn
            prompt = "\n".join(tool_outputs) + "\n\nNow provide your final answer."

        # Strip thinking blocks before parsing final content.
        # Falls back to original if stripping produces empty (model put everything in tags).
        if final_content:
            from benchmarks.utils import strip_think_tags
            final_content_clean = strip_think_tags(final_content) or final_content
        else:
            final_content_clean = final_content or ""

        # If no direct tool result, preserve the full cleaned model output as a
        # string.  _default_scorer will attempt numeric extraction internally
        # when needed (using the *last* number in the string, which is more
        # reliable than grabbing the first).  Preserving the string also allows
        # exact-match scoring for date strings like "2024-02-14" that would be
        # wrongly converted to a float by an eager _extract_number call.
        if actual_value is None and final_content_clean:
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
            metadata={
                "turns": turns,
                "valid_tool_call_format": valid_tool_call_format,
                "skill_selected_correctly": skill_selected_correctly,
                "tool_executed_successfully": tool_executed_successfully,
                "final_answer_correct": passed,
            },
        )
