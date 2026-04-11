"""
Abstract benchmark base class and result types.

Every benchmark must:
  1. Subclass ``Benchmark``
  2. Define ``name`` and ``description`` class attributes
  3. Implement ``run(model, skills, **kwargs) -> BenchmarkResult``

Results use pydantic models so they serialise cleanly to JSON.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------


class ScoringStrategy(str, Enum):
    """How individual test-case scores are aggregated."""
    EXACT_MATCH = "exact_match"
    PARTIAL_CREDIT = "partial_credit"
    BINARY = "binary"
    CUSTOM = "custom"


class TestCase(BaseModel):
    """A single test case inside a benchmark."""
    id: str
    prompt: str
    expected: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0        # relative importance for weighted average


class TestResult(BaseModel):
    """Outcome of running one TestCase against one model."""
    test_id: str
    passed: bool
    score: float                    # 0.0 – 1.0
    model_output: str = ""
    expected: Any = None
    actual: Any = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Aggregated result of running a full benchmark."""
    benchmark_name: str
    model_name: str
    skill_config: list[str] = Field(default_factory=list)   # enabled skill names
    test_results: list[TestResult] = Field(default_factory=list)

    # Aggregate stats (populated by ``finalise()``)
    total_tests: int = 0
    passed_tests: int = 0
    score: float = 0.0              # weighted mean of per-test scores
    avg_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    duration_s: float = 0.0
    errors: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def finalise(self) -> "BenchmarkResult":
        """Compute aggregate statistics from ``test_results``. Call after all tests."""
        self.total_tests = len(self.test_results)
        self.passed_tests = sum(1 for t in self.test_results if t.passed)
        self.errors = sum(1 for t in self.test_results if t.error is not None)

        if self.test_results:
            total_weight = sum(1.0 for _ in self.test_results)
            self.score = sum(t.score for t in self.test_results) / total_weight
            self.avg_latency_ms = (
                sum(t.latency_ms for t in self.test_results) / self.total_tests
            )
        else:
            self.score = 0.0
            self.avg_latency_ms = 0.0

        self.total_prompt_tokens = sum(t.prompt_tokens for t in self.test_results)
        self.total_completion_tokens = sum(t.completion_tokens for t in self.test_results)
        return self

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Benchmark(ABC):
    """
    Abstract base class for all benchmarks.

    Subclasses must set ``name`` / ``description`` and implement ``run()``.

    The runner instantiates each benchmark once and calls ``run()`` for
    every (model, skill_config) combination — so benchmarks must be
    stateless across runs (or reset themselves inside ``run()``).
    """

    #: Unique benchmark identifier (used in result JSON and summary tables)
    name: str = "unnamed_benchmark"
    #: Human-readable description
    description: str = ""

    @abstractmethod
    async def run(
        self,
        model: Any,         # ModelAdapter
        skills: Optional[Any] = None,   # SkillRegistry | None
        **kwargs: Any,
    ) -> BenchmarkResult:
        """
        Execute the full benchmark against *model*.

        Parameters
        ----------
        model:
            A ``ModelAdapter`` instance.
        skills:
            A ``SkillRegistry`` (may be ``None`` for baseline no-skill runs).
        **kwargs:
            Benchmark-specific overrides (e.g. ``n_samples``, ``temperature``).

        Returns
        -------
        BenchmarkResult
            Fully populated and finalised result object.
        """

    # ------------------------------------------------------------------
    # Helpers shared by concrete subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _make_result(
        model_name: str,
        benchmark_name: str,
        skill_config: list[str],
    ) -> tuple[BenchmarkResult, float]:
        """Create a blank BenchmarkResult and record the wall-clock start time."""
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            model_name=model_name,
            skill_config=skill_config,
        )
        return result, time.perf_counter()

    @staticmethod
    def _close_result(result: BenchmarkResult, start_time: float) -> BenchmarkResult:
        """Finalise stats and record total wall-clock duration."""
        result.duration_s = time.perf_counter() - start_time
        return result.finalise()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
