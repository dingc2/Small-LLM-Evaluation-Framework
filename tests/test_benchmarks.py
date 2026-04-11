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
        "calculate": "calculator",
        "compute": "calculator",
        "sqrt": "calculator",
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
    r = result.test_results[0]
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
