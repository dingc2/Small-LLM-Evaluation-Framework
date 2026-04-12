"""
Unit tests for benchmark scoring logic and result aggregation.

All tests use mock adapters — no real model API calls are made.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval_framework.adapters.base import ModelAdapter, ModelResponse, ToolCall, ToolDefinition
from eval_framework.benchmarks.base import Benchmark, BenchmarkResult, TestCase, TestResult
from eval_framework.benchmarks.end_to_end import EndToEndBenchmark, _default_scorer
from eval_framework.benchmarks.skill_selection import SkillSelectionBenchmark
from eval_framework.skills.registry import SkillRegistry

SKILLS_DIR = Path(__file__).parent.parent / "skills"


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------


class ConstantAdapter(ModelAdapter):
    """Always returns the same text response."""

    def __init__(self, response: str, model: str = "constant-mock") -> None:
        self._response = response
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, prompt, tools=None, system_prompt=None, **kw) -> ModelResponse:
        return ModelResponse(
            content=self._response,
            model_name=self._model,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=8.0,
        )


class ToolCallAdapter(ModelAdapter):
    """Emits a tool call with the given name and query arg, then on second call returns result."""

    def __init__(self, tool_name: str, expression: str, final_answer: str) -> None:
        self._tool_name = tool_name
        self._expression = expression
        self._final_answer = final_answer
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return "tool-call-mock"

    async def generate(self, prompt, tools=None, system_prompt=None, **kw) -> ModelResponse:
        self._call_count += 1
        if self._call_count == 1 and tools:
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(
                    id="tc_1",
                    name=self._tool_name,
                    arguments={"query": self._expression, "expression": self._expression},
                )],
                model_name=self.model_name,
                latency_ms=5.0,
                prompt_tokens=8,
                completion_tokens=3,
            )
        # Second call: return the final answer
        return ModelResponse(
            content=self._final_answer,
            model_name=self.model_name,
            latency_ms=5.0,
            prompt_tokens=15,
            completion_tokens=3,
        )


class RoutingAdapter(ModelAdapter):
    """
    Returns the correct skill name for queries containing keywords;
    otherwise returns 'none'.
    """

    ROUTING_MAP = {
        # Calculator keywords (natural phrasing in new harder cases)
        "times": "calculator",
        "power": "calculator",
        "sine": "calculator",
        "square root": "calculator",
        "divided by": "calculator",
        # Unit converter keywords
        "miles": "unit_converter",
        "ounces": "unit_converter",
        "fahrenheit": "unit_converter",
        "gallons": "unit_converter",
        "inches": "unit_converter",
        # Dictionary keywords
        "perplexity": "dictionary",
        "tokenization": "dictionary",
        "entropy": "dictionary",
        "gradient": "dictionary",
        "hallucination": "dictionary",
        # Datetime keywords
        "lease": "datetime_calc",
        "born on": "datetime_calc",
        "challenge": "datetime_calc",
        "how long between": "datetime_calc",
        "day of the week": "datetime_calc",
        "fall on": "datetime_calc",
    }

    @property
    def model_name(self) -> str:
        return "routing-mock"

    async def generate(self, prompt, tools=None, system_prompt=None, **kw) -> ModelResponse:
        prompt_lower = prompt.lower()
        for keyword, skill in self.ROUTING_MAP.items():
            if keyword in prompt_lower:
                return ModelResponse(content=skill, model_name=self.model_name, latency_ms=3.0)
        return ModelResponse(content="none", model_name=self.model_name, latency_ms=3.0)


# ---------------------------------------------------------------------------
# Tests: BenchmarkResult aggregation
# ---------------------------------------------------------------------------


def test_benchmark_result_finalise_computes_score():
    br = BenchmarkResult(
        benchmark_name="test",
        model_name="m",
        test_results=[
            TestResult(test_id="t1", passed=True, score=1.0, latency_ms=10),
            TestResult(test_id="t2", passed=False, score=0.0, latency_ms=20),
            TestResult(test_id="t3", passed=True, score=1.0, latency_ms=30),
        ],
    )
    br.finalise()
    assert br.total_tests == 3
    assert br.passed_tests == 2
    assert math.isclose(br.score, 2 / 3, rel_tol=1e-6)
    assert math.isclose(br.avg_latency_ms, 20.0)


def test_benchmark_result_pass_rate():
    br = BenchmarkResult(
        benchmark_name="b", model_name="m",
        test_results=[
            TestResult(test_id="a", passed=True, score=1.0),
            TestResult(test_id="b", passed=True, score=1.0),
        ],
    )
    br.finalise()
    assert br.pass_rate == 1.0


def test_benchmark_result_empty():
    br = BenchmarkResult(benchmark_name="b", model_name="m")
    br.finalise()
    assert br.score == 0.0
    assert br.pass_rate == 0.0
    assert br.total_tests == 0


def test_benchmark_result_total_tokens():
    br = BenchmarkResult(
        benchmark_name="b", model_name="m",
        test_results=[
            TestResult(test_id="a", passed=True, score=1.0,
                       prompt_tokens=10, completion_tokens=5),
        ],
    )
    br.finalise()
    assert br.total_prompt_tokens == 10
    assert br.total_completion_tokens == 5
    assert br.total_tokens == 15


def test_benchmark_result_errors_count():
    br = BenchmarkResult(
        benchmark_name="b", model_name="m",
        test_results=[
            TestResult(test_id="a", passed=False, score=0.0, error="timeout"),
            TestResult(test_id="b", passed=True, score=1.0),
        ],
    )
    br.finalise()
    assert br.errors == 1


# ---------------------------------------------------------------------------
# Tests: Benchmark ABC
# ---------------------------------------------------------------------------


def test_cannot_instantiate_benchmark_abstract():
    with pytest.raises(TypeError):
        Benchmark()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Tests: _default_scorer
# ---------------------------------------------------------------------------


def test_default_scorer_exact_int():
    assert _default_scorer(42, 42) == 1.0


def test_default_scorer_float_within_tolerance():
    assert _default_scorer(3.141592, 3.141593, tolerance=1e-4) == 1.0


def test_default_scorer_float_outside_tolerance():
    assert _default_scorer(1.0, 2.0) == 0.0


def test_default_scorer_string_match_case_insensitive():
    assert _default_scorer("Hello", "hello") == 1.0


def test_default_scorer_string_mismatch():
    assert _default_scorer("foo", "bar") == 0.0


def test_default_scorer_none_actual():
    assert _default_scorer(None, 42) == 0.0


def test_default_scorer_int_string_coercion():
    # "42" should match numeric 42
    assert _default_scorer("42", 42) == 1.0


# ---------------------------------------------------------------------------
# Tests: SkillSelectionBenchmark
# ---------------------------------------------------------------------------


def _load_registry() -> SkillRegistry:
    reg = SkillRegistry(SKILLS_DIR)
    reg.load()
    return reg


def test_skill_selection_perfect_routing():
    adapter = RoutingAdapter()
    registry = _load_registry()
    bench = SkillSelectionBenchmark(include_no_skill_cases=True)
    result = asyncio.run(bench.run(adapter, registry))
    assert result.total_tests > 0
    # Routing adapter is hand-crafted to be correct, expect high score
    assert result.score >= 0.7


def test_skill_selection_always_wrong():
    adapter = ConstantAdapter("wrong_skill_name_xyz")
    registry = _load_registry()
    bench = SkillSelectionBenchmark(include_no_skill_cases=False)
    result = asyncio.run(bench.run(adapter, registry))
    assert result.score == 0.0


def test_skill_selection_always_none():
    """Adapter that always says 'none' should ace the no-skill cases."""
    adapter = ConstantAdapter("none")
    registry = _load_registry()
    bench = SkillSelectionBenchmark(include_no_skill_cases=True)
    result = asyncio.run(bench.run(adapter, registry))
    # It passes "no skill" cases but fails calculator cases
    assert result.total_tests > 0
    assert 0.0 <= result.score <= 1.0


def test_skill_selection_result_shape():
    adapter = RoutingAdapter()
    registry = _load_registry()
    bench = SkillSelectionBenchmark()
    result = asyncio.run(bench.run(adapter, registry))
    assert result.benchmark_name == "skill_selection_accuracy"
    assert result.model_name == "routing-mock"
    assert isinstance(result.skill_config, list)
    assert all(isinstance(tr, TestResult) for tr in result.test_results)


def test_skill_selection_without_registry():
    """With no registry, no test cases should be generated (or all should be no-skill)."""
    adapter = ConstantAdapter("none")
    bench = SkillSelectionBenchmark(include_no_skill_cases=True)
    result = asyncio.run(bench.run(adapter, skills=None))
    # All no-skill cases should pass
    assert result.score == 1.0 or result.total_tests == 0


def test_skill_selection_extra_cases():
    adapter = ConstantAdapter("calculator")
    registry = _load_registry()
    extra = [{"id": "custom_01", "prompt": "custom test", "expected": "calculator"}]
    bench = SkillSelectionBenchmark(
        extra_cases=extra, include_no_skill_cases=False
    )
    result = asyncio.run(bench.run(adapter, registry))
    # All test cases should pass (adapter always returns "calculator")
    assert result.total_tests >= 1
    custom_results = [r for r in result.test_results if r.test_id == "custom_01"]
    assert len(custom_results) == 1
    assert custom_results[0].passed


# ---------------------------------------------------------------------------
# Tests: EndToEndBenchmark
# ---------------------------------------------------------------------------


def test_end_to_end_no_tools_correct_answer():
    adapter = ConstantAdapter("hello")
    bench = EndToEndBenchmark(extra_cases=[
        {"id": "custom_hello", "prompt": "Say exactly: hello", "expected": "hello"}
    ])
    result = asyncio.run(bench.run(adapter, skills=None))
    # Built-in cases are also included; find our specific test by id
    assert result.total_tests >= 1
    custom = next(r for r in result.test_results if r.test_id == "custom_hello")
    assert custom.passed


def test_end_to_end_no_tools_wrong_answer():
    adapter = ConstantAdapter("goodbye")
    bench = EndToEndBenchmark(extra_cases=[
        {"id": "t1", "prompt": "Say exactly: hello", "expected": "hello"}
    ])
    result = asyncio.run(bench.run(adapter, skills=None))
    assert result.test_results[0].passed is False


def test_end_to_end_tool_call_correct():
    """Model calls calculator tool and gets the right answer."""
    adapter = ToolCallAdapter("calculator", "17 * 23", "391")
    registry = _load_registry()
    bench = EndToEndBenchmark(extra_cases=[
        {"id": "t_tool", "prompt": "calculate 17 * 23", "expected": 391}
    ])
    result = asyncio.run(bench.run(adapter, registry))
    assert result.total_tests >= 1
    tool_result = next(r for r in result.test_results if r.test_id == "t_tool")
    assert tool_result.passed


def test_end_to_end_result_shape():
    adapter = ConstantAdapter("42")
    bench = EndToEndBenchmark()
    result = asyncio.run(bench.run(adapter, skills=None))
    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_name == "end_to_end_task_completion"
    assert result.score >= 0.0
    assert result.avg_latency_ms >= 0.0


def test_end_to_end_numeric_extraction():
    """Model returns 'The answer is 391.' — should still pass numeric extraction."""
    adapter = ConstantAdapter("The answer is 391.")
    bench = EndToEndBenchmark(extra_cases=[
        {"id": "t_parse", "prompt": "What is 17*23?", "expected": 391}
    ])
    result = asyncio.run(bench.run(adapter, skills=None))
    r = next(r for r in result.test_results if r.test_id == "t_parse")
    assert r.passed


def test_end_to_end_float_tolerance():
    adapter = ConstantAdapter("3.14159")
    bench = EndToEndBenchmark(
        float_tolerance=1e-4,
        extra_cases=[{"id": "t_pi", "prompt": "What is pi?", "expected": math.pi}],
    )
    result = asyncio.run(bench.run(adapter, skills=None))
    # 3.14159 vs 3.14159265... — within 1e-4 tolerance; find by id
    pi_result = next(r for r in result.test_results if r.test_id == "t_pi")
    assert pi_result.passed


def test_end_to_end_records_turn_count():
    adapter = ToolCallAdapter("calculator", "2+2", "4")
    registry = _load_registry()
    bench = EndToEndBenchmark(extra_cases=[
        {"id": "tc", "prompt": "calculate 2+2", "expected": 4}
    ])
    result = asyncio.run(bench.run(adapter, registry))
    r = next(r for r in result.test_results if r.test_id == "tc")
    assert r.metadata.get("turns", 0) >= 1


# ---------------------------------------------------------------------------
# Regression tests: bugs found in run 20260412T050357Z
# ---------------------------------------------------------------------------


# --- _default_scorer: last-number extraction from prose ---

def test_default_scorer_last_number_in_prose():
    """
    "5 km is approximately 3.10686 miles." — answer is the LAST number (3.10686),
    not the first (5).  Old _extract_number grabbed 5 and the test failed.
    """
    assert _default_scorer("5 km is approximately 3.10686 miles.", 3.10686, tolerance=0.001) == 1.0


def test_default_scorer_first_number_not_grabbed():
    """Ensure the first number in prose doesn't shadow the actual answer."""
    assert _default_scorer("100 F is about 37.78 C", 37.78, tolerance=0.01) == 1.0


def test_default_scorer_last_number_unit_conversion():
    """1 kg to lb — model says '1 kg is 2.20462 lb'."""
    assert _default_scorer("1 kg is 2.20462 lb", 2.20462, tolerance=0.001) == 1.0


# --- _default_scorer: date-string exact match (not numeric extraction) ---

def test_default_scorer_date_string_exact_match():
    """
    Model correctly outputs "2024-02-14" but old code extracted 2024.0
    and compared that to the string "2024-02-14", which always failed.
    """
    assert _default_scorer("2024-02-14", "2024-02-14") == 1.0


def test_default_scorer_date_string_mismatch():
    """Two different dates should NOT match."""
    assert _default_scorer("2024-07-04", "2024-02-14") == 0.0


def test_default_scorer_date_output_wrong_format():
    """
    Model returns the date "2024-07-04" but expected is the *day name* "Thursday".
    Should fail — extracting 2024 and comparing to "Thursday" must also fail.
    """
    assert _default_scorer("2024-07-04", "Thursday") == 0.0


# --- EndToEndBenchmark: prose output with answer at end ---

def test_end_to_end_prose_answer_last_number():
    """
    Adapter returns explanatory prose; scoring should use the last number.
    Simulates gemma4 saying "5 km is approximately 3.10686 miles." for a
    unit-conversion case.
    """
    adapter = ConstantAdapter("5 km is approximately 3.10686 miles.")
    bench = EndToEndBenchmark(
        float_tolerance=0.001,
        extra_cases=[{"id": "conv_prose", "prompt": "Convert 5 km to miles",
                      "expected": 3.10686}],
    )
    result = asyncio.run(bench.run(adapter, skills=None))
    r = next(r for r in result.test_results if r.test_id == "conv_prose")
    assert r.passed, f"actual={r.actual!r}, expected={r.expected!r}"


def test_end_to_end_date_string_correct():
    """
    Model returns the correct date string "2024-02-14".
    Old code extracted 2024.0; new code preserves the string for exact match.
    """
    adapter = ConstantAdapter("2024-02-14")
    bench = EndToEndBenchmark(extra_cases=[
        {"id": "date_str", "prompt": "Add 30 days to 2024-01-15", "expected": "2024-02-14"}
    ])
    result = asyncio.run(bench.run(adapter, skills=None))
    r = next(r for r in result.test_results if r.test_id == "date_str")
    assert r.passed, f"actual={r.actual!r}"


# --- SkillSelectionBenchmark: empty-output recovery (think-tag truncation) ---

class TruncatedThinkAdapter(ModelAdapter):
    """
    Simulates a reasoning model that hits the token limit inside a <think> block
    without emitting any final answer.  The model has decided on "none" inside
    the think block, but the output is truncated before the closing tag.
    """

    @property
    def model_name(self) -> str:
        return "truncated-think-mock"

    async def generate(self, prompt, tools=None, system_prompt=None, **kw) -> ModelResponse:
        # Unclosed think block — model decided "none" but never closed the tag
        content = "<think>The user just wants a joke, so the answer is none and I should"
        return ModelResponse(
            content=content, model_name=self.model_name, latency_ms=5.0,
        )


def test_skill_selection_recovers_from_truncated_think_block():
    """
    When the entire response is inside an unclosed <think> block (token-limit
    truncation), the benchmark should recover the decision from within the
    think content rather than returning an empty string that always fails.
    """
    adapter = TruncatedThinkAdapter()
    registry = _load_registry()
    bench = SkillSelectionBenchmark(include_no_skill_cases=True)
    result = asyncio.run(bench.run(adapter, registry))
    # All no-skill cases expect the string "none" exactly
    no_skill_results = [
        r for r in result.test_results
        if r.expected == "none"
    ]
    assert len(no_skill_results) == 5
    passed = sum(1 for r in no_skill_results if r.passed)
    assert passed == 5, (
        f"Expected all 5 no-skill cases to pass via think-block recovery, "
        f"got {passed}/5 — normalised outputs: "
        f"{[r.metadata.get('normalised_output') for r in no_skill_results]}"
    )


# ---------------------------------------------------------------------------
# Case count parity: all_skills vs no_skills must run the same set
# ---------------------------------------------------------------------------


def test_e2e_all_skills_and_no_skills_same_case_count():
    """
    Both skill conditions must run all 20 built-in cases.

    no_skills (skills=None) passes the filter because
    ``skills is not None`` is False, so no case is ever skipped.
    all_skills (full registry) passes because every skill is available.
    If this test fails it means the no_skills regression bug was reintroduced:
    the old code used ``skills is None`` as the skip condition, which caused
    no_skills to run only the 2 no-tool baseline cases.
    """
    bench = EndToEndBenchmark()
    registry = _load_registry()

    cases_with_skills = bench._build_test_cases(registry)
    cases_without_skills = bench._build_test_cases(None)

    assert len(cases_with_skills) == 20, (
        f"all_skills should run 20 cases, got {len(cases_with_skills)}"
    )
    assert len(cases_without_skills) == 20, (
        f"no_skills should also run 20 cases, got {len(cases_without_skills)} "
        "(regression: check the filter in _build_test_cases)"
    )
    # The case IDs must match exactly — same population, same order
    with_ids = [tc.id for tc in cases_with_skills]
    without_ids = [tc.id for tc in cases_without_skills]
    assert with_ids == without_ids, (
        f"Case ID mismatch between conditions:\n"
        f"  all_skills only: {set(with_ids) - set(without_ids)}\n"
        f"  no_skills only:  {set(without_ids) - set(with_ids)}"
    )


# ---------------------------------------------------------------------------
# Dictionary scoring unit tests (keyword-overlap heuristic)
# ---------------------------------------------------------------------------


def test_default_scorer_dictionary_paraphrase_passes():
    """A genuine paraphrase with >60% keyword overlap should score 1.0."""
    expected = "a measurement of how well a probability model predicts a sample; lower is better"
    # Overlap words (3+ chars): measurement, how, well, probability, model, predicts, lower, better
    # = 8 of the 9 expected keywords → ratio 8/9 ≈ 0.89 > 0.60
    actual = "perplexity measures how well a probability model predicts the next token; lower values are better"
    assert _default_scorer(actual, expected) == 1.0


def test_default_scorer_dictionary_borderline_fails():
    """A partial paraphrase with <60% keyword overlap should score 0.0."""
    expected = "a measurement of how well a probability model predicts a sample; lower is better"
    # expected_words (9): measurement, how, well, probability, model, predicts, sample, lower, better
    # actual hits: how, well, model, predicts, better  → 5/9 ≈ 0.556 < 0.60
    actual = "perplexity measures model quality and how well it predicts from training data better than a baseline"
    assert _default_scorer(actual, expected) == 0.0


def test_default_scorer_dictionary_incorrect_fails():
    """A completely wrong definition should score 0.0."""
    expected = "a measurement of how well a probability model predicts a sample; lower is better"
    actual = "a type of cryptocurrency used in digital transactions for secure payments"
    assert _default_scorer(actual, expected) == 0.0


# ---------------------------------------------------------------------------
# Think-tag stripping utility tests (centralized logic in benchmarks/utils.py)
# ---------------------------------------------------------------------------

from eval_framework.benchmarks.utils import strip_think_tags, recover_answer_from_think_block


def test_strip_think_tags_closed_block():
    """Closed <think>...</think> block is removed; answer is preserved."""
    text = "<think>Let me reason about this carefully.</think>calculator"
    assert strip_think_tags(text) == "calculator"


def test_strip_think_tags_closed_thinking_alias():
    """<thinking>...</thinking> variant is also stripped."""
    text = "<thinking>step-by-step reasoning</thinking>\ndictionary"
    assert strip_think_tags(text).strip() == "dictionary"


def test_strip_think_tags_unclosed_block():
    """An unclosed <think> tag (token-limit truncation) is stripped to empty."""
    text = "<think>The user wants a calculation so the answer is"
    result = strip_think_tags(text)
    assert result == ""


def test_strip_think_tags_answer_after_closed_block():
    """Content after a closed think block is returned intact."""
    text = "<think>reasoning here\nmultiline</think>\nunit_converter"
    assert strip_think_tags(text).strip() == "unit_converter"


def test_strip_think_tags_no_tags_unchanged():
    """Text with no think tags is returned unchanged."""
    text = "calculator"
    assert strip_think_tags(text) == "calculator"


def test_recover_answer_from_think_block_finds_last_token():
    """Recovery returns the last-occurring known token from raw output."""
    raw = "<think>maybe calculator? no wait, the user wants datetime_calc and I should"
    known = ["calculator", "datetime_calc", "dictionary", "none"]
    result = recover_answer_from_think_block(raw, known)
    assert result == "datetime_calc"


def test_recover_answer_from_think_block_no_match_returns_none():
    """Returns None when no known token appears anywhere in the output."""
    raw = "<think>the user said something completely unrecognisable"
    known = ["calculator", "dictionary", "none"]
    result = recover_answer_from_think_block(raw, known)
    assert result is None
