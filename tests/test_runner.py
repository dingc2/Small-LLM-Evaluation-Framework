"""Tests for runner utilities (no live model calls)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sLLM_eval_framework.benchmarks.base import BenchmarkResult, TestResult
from sLLM_eval_framework.runner import RunSummary, _write_failures_jsonl


def test_write_failures_jsonl_only_all_skills_failures() -> None:
    summary = RunSummary(
        run_id="TEST_RUN_ID",
        started_at="x",
        results=[
            BenchmarkResult(
                benchmark_name="skill_selection_accuracy",
                model_name="fake-model",
                skill_config=["calculator"],
                test_results=[
                    TestResult(
                        test_id="t1",
                        passed=False,
                        score=0.0,
                        model_output="wrong",
                        prompt="hello prompt",
                        expected="calculator",
                        actual="none",
                        metadata={"k": 1},
                    ),
                    TestResult(
                        test_id="t_ok",
                        passed=True,
                        score=1.0,
                        prompt="p",
                    ),
                ],
                metadata={"skill_config_name": "all_skills", "run_index": 2},
            ),
            BenchmarkResult(
                benchmark_name="skill_selection_accuracy",
                model_name="fake-model",
                skill_config=[],
                test_results=[
                    TestResult(
                        test_id="t2",
                        passed=False,
                        score=0.0,
                        model_output="x",
                        prompt="should not appear",
                    ),
                ],
                metadata={"skill_config_name": "no_skills", "run_index": 0},
            ),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write_failures_jsonl(summary, out)
        path = out / "TEST_RUN_ID_failures.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["test_id"] == "t1"
        assert row["prompt"] == "hello prompt"
        assert row["skill_config"] == "all_skills"
        assert row["run_index"] == 2
        assert row["model_output"] == "wrong"


def test_write_failures_jsonl_empty_file_when_no_failures() -> None:
    summary = RunSummary(
        run_id="EMPTY_RUN",
        started_at="x",
        results=[
            BenchmarkResult(
                benchmark_name="b",
                model_name="m",
                test_results=[
                    TestResult(test_id="ok", passed=True, score=1.0, prompt="p"),
                ],
                metadata={"skill_config_name": "all_skills", "run_index": 0},
            ),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write_failures_jsonl(summary, out)
        path = out / "EMPTY_RUN_failures.jsonl"
        assert path.read_text(encoding="utf-8") == ""
