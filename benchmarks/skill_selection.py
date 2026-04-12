"""
Skill-selection accuracy benchmark.

Measures: given a natural-language query, does the model select the correct
skill from the available set?

Protocol
--------
1. Build a system prompt listing all enabled skills (names + descriptions).
2. Ask the model to respond with ONLY the name of the skill it would use,
   or "none" if no skill applies.
3. Compare the model's answer (after normalisation) to the expected skill name.
4. Score: 1.0 for exact match, 0.0 otherwise (binary).

Test cases are auto-generated from ``SKILL_META["examples"]`` when found in
the skill, plus a built-in no-skill set.  Custom cases can be injected via
``extra_cases``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from .base import Benchmark, BenchmarkResult, ScoringStrategy, TestCase, TestResult

logger = logging.getLogger(__name__)

# System prompt template
_SYSTEM_PROMPT = """\
You are a skill-routing assistant. Given a user query, you must decide which
skill (if any) to invoke.

Available skills:
{skill_list}

Respond with EXACTLY one of the skill names listed above, or the word "none"
if no skill is appropriate. Output ONLY the skill name — no explanation, no
punctuation, nothing else.
"""

# Built-in test cases: queries that should map to "none"
_NO_SKILL_CASES: list[dict[str, Any]] = [
    {"id": "no_skill_01", "prompt": "Tell me a joke.", "expected": "none"},
    {"id": "no_skill_02", "prompt": "What is the capital of France?", "expected": "none"},
    {"id": "no_skill_03", "prompt": "Write a haiku about spring.", "expected": "none"},
    {"id": "no_skill_04", "prompt": "Explain how photosynthesis works.", "expected": "none"},
    {"id": "no_skill_05", "prompt": "Summarize the plot of Romeo and Juliet.", "expected": "none"},
]

# Built-in calculator cases
_CALCULATOR_CASES: list[dict[str, Any]] = [
    {"id": "calc_sel_01", "prompt": "Calculate 15 * 7 + 3", "expected": "calculator"},
    {"id": "calc_sel_02", "prompt": "What is sqrt(256)?", "expected": "calculator"},
    {"id": "calc_sel_03", "prompt": "Compute 2 ** 16", "expected": "calculator"},
    {"id": "calc_sel_04", "prompt": "Evaluate sin(0) + cos(0)", "expected": "calculator"},
    {"id": "calc_sel_05", "prompt": "Solve (100 / 4) - 7", "expected": "calculator"},
]

# Unit converter cases
_UNIT_CONVERTER_CASES: list[dict[str, Any]] = [
    {"id": "conv_sel_01", "prompt": "Convert 5 kilometers to miles", "expected": "unit_converter"},
    {"id": "conv_sel_02", "prompt": "How many pounds in 10 kg?", "expected": "unit_converter"},
    {"id": "conv_sel_03", "prompt": "Convert 100 fahrenheit to celsius", "expected": "unit_converter"},
    {"id": "conv_sel_04", "prompt": "Convert 2 gallons to liters", "expected": "unit_converter"},
    {"id": "conv_sel_05", "prompt": "How many inches in 3 feet?", "expected": "unit_converter"},
]

# Dictionary cases
_DICTIONARY_CASES: list[dict[str, Any]] = [
    {"id": "dict_sel_01", "prompt": "Define the word 'ephemeral'", "expected": "dictionary"},
    {"id": "dict_sel_02", "prompt": "What does ubiquitous mean?", "expected": "dictionary"},
    {"id": "dict_sel_03", "prompt": "Give me the definition of algorithm", "expected": "dictionary"},
    {"id": "dict_sel_04", "prompt": "Look up the word 'paradigm' in the dictionary", "expected": "dictionary"},
    {"id": "dict_sel_05", "prompt": "What is the meaning of quantization?", "expected": "dictionary"},
]

# Date/time calculator cases
_DATETIME_CASES: list[dict[str, Any]] = [
    {"id": "date_sel_01", "prompt": "How many days between 2024-01-01 and 2024-12-31?", "expected": "datetime_calc"},
    {"id": "date_sel_02", "prompt": "What day of the week is 2024-07-04?", "expected": "datetime_calc"},
    {"id": "date_sel_03", "prompt": "Add 30 days to 2024-01-15", "expected": "datetime_calc"},
    {"id": "date_sel_04", "prompt": "How many days from 2024-03-01 to 2024-06-15?", "expected": "datetime_calc"},
    {"id": "date_sel_05", "prompt": "What day is 2024-12-25?", "expected": "datetime_calc"},
]


class SkillSelectionBenchmark(Benchmark):
    """
    Benchmark: does the model correctly choose which skill to call?

    Parameters
    ----------
    extra_cases:
        Additional ``TestCase``-compatible dicts (keys: id, prompt, expected).
    include_no_skill_cases:
        Whether to include the built-in "no skill needed" negative examples.
    temperature:
        Generation temperature (default 0 for deterministic routing).
    """

    name = "skill_selection_accuracy"
    description = (
        "Measures how accurately the model selects the correct skill "
        "given a user query from a menu of available skills."
    )

    def __init__(
        self,
        extra_cases: Optional[list[dict[str, Any]]] = None,
        include_no_skill_cases: bool = True,
        temperature: float = 0.0,
    ) -> None:
        self._extra_cases = extra_cases or []
        self._include_no_skill = include_no_skill_cases
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------

    async def run(
        self,
        model: Any,
        skills: Optional[Any] = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        skill_names: list[str] = []
        if skills is not None:
            skill_names = skills.names

        result, start_time = self._make_result(
            model_name=model.model_name,
            benchmark_name=self.name,
            skill_config=skill_names,
        )

        test_cases = self._build_test_cases(skills)
        if not test_cases:
            logger.warning("SkillSelectionBenchmark: no test cases — nothing to run.")
            return self._close_result(result, start_time)

        system_prompt = self._build_system_prompt(skills)

        # Run all test cases concurrently
        tasks = [
            self._run_single(tc, model, system_prompt, **kwargs)
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
                    )
                )
            else:
                result.test_results.append(tr)

        return self._close_result(result, start_time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_test_cases(self, skills: Optional[Any]) -> list[TestCase]:
        cases: list[dict[str, Any]] = []

        # Include skill-specific cases only when that skill is registered
        if skills and "calculator" in skills:
            cases.extend(_CALCULATOR_CASES)
        if skills and "unit_converter" in skills:
            cases.extend(_UNIT_CONVERTER_CASES)
        if skills and "dictionary" in skills:
            cases.extend(_DICTIONARY_CASES)
        if skills and "datetime_calc" in skills:
            cases.extend(_DATETIME_CASES)

        if self._include_no_skill:
            cases.extend(_NO_SKILL_CASES)

        cases.extend(self._extra_cases)

        return [
            TestCase(
                id=c["id"],
                prompt=c["prompt"],
                expected=c["expected"],
                metadata=c.get("metadata", {}),
            )
            for c in cases
        ]

    @staticmethod
    def _build_system_prompt(skills: Optional[Any]) -> str:
        if skills is None or not skills.names:
            skill_list = "  (no skills available — answer 'none' for every query)"
        else:
            lines = [
                f"  - {s.name}: {s.description}"
                for s in skills.all()
            ]
            skill_list = "\n".join(lines)
        return _SYSTEM_PROMPT.format(skill_list=skill_list)

    async def _run_single(
        self,
        tc: TestCase,
        model: Any,
        system_prompt: str,
        **kwargs: Any,
    ) -> TestResult:
        from eval_framework.adapters.base import AdapterError

        try:
            response = await model.generate(
                prompt=tc.prompt,
                system_prompt=system_prompt,
                temperature=kwargs.get("temperature", self._temperature),
            )
        except AdapterError as exc:
            return TestResult(
                test_id=tc.id,
                passed=False,
                score=0.0,
                error=str(exc),
                expected=tc.expected,
            )

        raw_output = response.content.strip().lower()
        # Normalise: take first non-empty word/line
        first_word = raw_output.split()[0] if raw_output.split() else ""
        # Strip punctuation
        import re
        normalised = re.sub(r"[^\w]", "", first_word)

        expected = str(tc.expected).strip().lower()
        passed = normalised == expected
        score = 1.0 if passed else 0.0

        return TestResult(
            test_id=tc.id,
            passed=passed,
            score=score,
            model_output=response.content,
            expected=tc.expected,
            actual=normalised,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata={"normalised_output": normalised},
        )
