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

# System prompt template — structured for small LLMs with few-shot examples.
# Research shows few-shot examples are the single biggest driver of routing
# accuracy for models <20B (see LangChain few-shot tool-calling blog, 2024).
_SYSTEM_PROMPT = """\
You are a skill-routing assistant. Read the user's query and pick which skill to use.

## Available skills (pick ONE or say "none"):
{skill_list}

## Examples:
User: "What is 123 * 456?"
Answer: calculator

User: "How many liters in 5 gallons?"
Answer: unit_converter

User: "What does the word 'inference' mean?"
Answer: dictionary

User: "How many days between 2023-06-01 and 2023-09-15?"
Answer: datetime_calc

User: "Dots score for male 80kg 600kg total?"
Answer: powerlifting

User: "Write me a poem about the ocean."
Answer: none

## Rules:
- Output ONLY the skill name (e.g. calculator) or the word none.
- Do NOT add any explanation, punctuation, or extra text.
- Do NOT wrap your answer in tags or quotes.
- Pick "none" if the query is general knowledge, creative writing, or opinion.
"""

# Built-in test cases: queries that should map to "none"
# These are intentionally tricky — they mention numbers, units, dates, or
# word meanings but should NOT trigger any skill.
_NO_SKILL_CASES: list[dict[str, Any]] = [
    {"id": "no_skill_01", "prompt": "Why did the number 13 become associated with bad luck?", "expected": "none"},
    {"id": "no_skill_02", "prompt": "Explain the concept of absolute zero in physics.", "expected": "none"},
    {"id": "no_skill_03", "prompt": "What were the historical origins of the metric system?", "expected": "none"},
    {"id": "no_skill_04", "prompt": "Describe the cultural significance of the calendar reform of 1582.", "expected": "none"},
    {"id": "no_skill_05", "prompt": "Why do some languages have untranslatable words?", "expected": "none"},
]

# Built-in calculator cases — phrased naturally so routing isn't trivially
# keyword-triggered by words like "calculate" or "compute".
_CALCULATOR_CASES: list[dict[str, Any]] = [
    {"id": "calc_sel_01", "prompt": "I need to figure out what 347 times 829 is", "expected": "calculator"},
    {"id": "calc_sel_02", "prompt": "If I raise 7 to the 5th power then subtract 9384, what do I get?", "expected": "calculator"},
    {"id": "calc_sel_03", "prompt": "What's the sine of 1.37 radians plus the cosine of 2.84?", "expected": "calculator"},
    {"id": "calc_sel_04", "prompt": "Take the square root of 7291 and round to 2 decimals", "expected": "calculator"},
    {"id": "calc_sel_05", "prompt": "How much is (1247 + 3891) divided by 17.3?", "expected": "calculator"},
]

# Unit converter cases — use non-round values and natural phrasing.
# Last 3 cases cover the clinical-lab extension (Step 1 of v2 plan).
_UNIT_CONVERTER_CASES: list[dict[str, Any]] = [
    {"id": "conv_sel_01", "prompt": "My car's odometer reads 38,471 km — what's that in miles?", "expected": "unit_converter"},
    {"id": "conv_sel_02", "prompt": "A recipe calls for 237 grams of flour, how many ounces is that?", "expected": "unit_converter"},
    {"id": "conv_sel_03", "prompt": "It's 41 degrees Fahrenheit outside — what is that in Celsius?", "expected": "unit_converter"},
    {"id": "conv_sel_04", "prompt": "I have a 3.7 liter engine — how many gallons is that?", "expected": "unit_converter"},
    {"id": "conv_sel_05", "prompt": "The shelf is 91.4 centimeters wide, I need that in inches", "expected": "unit_converter"},
    {"id": "conv_sel_06", "prompt": "Convert 1.5 mg/dL of serum creatinine to µmol/L", "expected": "unit_converter"},
    {"id": "conv_sel_07", "prompt": "What's 126 mg/dL glucose in mmol/L?", "expected": "unit_converter"},
    {"id": "conv_sel_08", "prompt": "Hemoglobin 14.5 g/dL — what's that in g/L?", "expected": "unit_converter"},
]

# Dictionary cases — avoid blatant keywords like "define" or "dictionary"
_DICTIONARY_CASES: list[dict[str, Any]] = [
    {"id": "dict_sel_01", "prompt": "I keep hearing the word 'perplexity' in ML contexts — what does it actually mean?", "expected": "dictionary"},
    {"id": "dict_sel_02", "prompt": "Can you tell me what 'tokenization' refers to?", "expected": "dictionary"},
    {"id": "dict_sel_03", "prompt": "Someone described a problem as having high 'entropy' — what's that?", "expected": "dictionary"},
    {"id": "dict_sel_04", "prompt": "What exactly is a 'gradient' in the technical sense?", "expected": "dictionary"},
    {"id": "dict_sel_05", "prompt": "I need to understand what 'hallucination' means when talking about LLMs", "expected": "dictionary"},
]

# Date/time calculator cases — use obscure dates, not holidays or Jan 1
_DATETIME_CASES: list[dict[str, Any]] = [
    {"id": "date_sel_01", "prompt": "My lease started 2024-02-17 and ends 2025-11-03 — how many days is that?", "expected": "datetime_calc"},
    {"id": "date_sel_02", "prompt": "I was born on 1997-08-23 — what day of the week was that?", "expected": "datetime_calc"},
    {"id": "date_sel_03", "prompt": "If I start a 90-day challenge on 2025-03-11, when does it end?", "expected": "datetime_calc"},
    {"id": "date_sel_04", "prompt": "How long between March 17, 2024 and October 9, 2024 in days?", "expected": "datetime_calc"},
    {"id": "date_sel_05", "prompt": "What day of the week will September 29, 2025 fall on?", "expected": "datetime_calc"},
]

# Powerlifting Dots cases — new skill routing coverage
_POWERLIFTING_CASES: list[dict[str, Any]] = [
    {"id": "pow_sel_01", "prompt": "What is the Dots coefficient for a 83.2kg male with 620kg total?", "expected": "powerlifting"},
    {"id": "pow_sel_02", "prompt": "Calculate the IPF points for a female lifter at 57kg bodyweight with a 390kg total", "expected": "powerlifting"},
    {"id": "pow_sel_03", "prompt": "Score this powerlifting performance: male, 74kg, 580kg total", "expected": "powerlifting"},
    {"id": "pow_sel_04", "prompt": "How many Dots is 410kg at 63kg bodyweight (F)?", "expected": "powerlifting"},
    {"id": "pow_sel_05", "prompt": "Give me the Dots rating for my last meet: 700kg total at 100kg bodyweight male", "expected": "powerlifting"},
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

        # Build the list of valid answer tokens once so _run_single can use them
        # for think-block recovery without needing the full registry object.
        known_tokens: list[str] = (skills.names if skills else []) + ["none"]

        # Run all test cases concurrently
        tasks = [
            self._run_single(tc, model, system_prompt, known_tokens, **kwargs)
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
        if skills and "powerlifting" in skills:
            cases.extend(_POWERLIFTING_CASES)

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
        known_tokens: list[str],
        **kwargs: Any,
    ) -> TestResult:
        from sLLM_eval_framework.adapters.base import AdapterError

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

        from .utils import strip_think_tags, recover_answer_from_think_block
        import re
        raw_output = response.content.strip()

        # Strip thinking blocks emitted by reasoning models (e.g. Qwen, DeepSeek).
        # Handles both closed (<think>...</think>) and unclosed (<think>...) tags.
        cleaned = strip_think_tags(raw_output)

        if not cleaned:
            # The entire response was inside think tags (possibly truncated at the
            # token limit before the model emitted a final answer).  Recover by
            # scanning raw content for the last known valid answer token.
            recovered = recover_answer_from_think_block(raw_output, known_tokens)
            cleaned = recovered if recovered is not None else raw_output

        cleaned_lower = cleaned.lower()

        # Take the last non-empty line — models often put the answer at the end
        lines = [l.strip() for l in cleaned_lower.splitlines() if l.strip()]
        candidate = lines[-1] if lines else ""

        # Extract the first word and strip punctuation
        first_word = candidate.split()[0] if candidate.split() else ""
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
