"""
Evaluation runner — orchestrates the full (model × skill_config × benchmark) sweep.

Usage
-----
Run from inside the sLLM_eval_framework/ directory:

    # Quick smoke test
    python runner.py --config config_quick.yaml --verbose

    # Full evaluation sweep
    python runner.py --config config_ollama.yaml --verbose

    # Override output directory
    python runner.py --config config_ollama.yaml --output results/my_run --verbose

    # Programmatic
    runner = EvaluationRunner.from_yaml("config_ollama.yaml")
    summary = await runner.run_all()
    summary.print_table()
    summary.save_json("results/run.json")

Config shape (YAML / dict)
--------------------------
    models:
      - type: ollama              # recommended for local inference
        model: qwen3.5:4b
        kwargs:
          temperature: 0.0
          num_predict: 512

      - type: openai              # OpenAI or any compatible endpoint
        model: gpt-4o-mini
        api_key: sk-...           # optional; falls back to OPENAI_API_KEY

    skills_dir: ./skills          # relative to sLLM_eval_framework/
    skill_configs:
      - name: all_skills          # label used in results
        enabled: []               # empty = all skills; list names to restrict
      - name: no_skills
        enabled: null             # null = skip registry (baseline)

    benchmarks:
      - type: skill_selection_accuracy
      - type: end_to_end_task_completion
        kwargs:
          max_turns: 3

    runs: 3                       # repeat each combination N times (for variance)
    output_dir: ./results         # relative to sLLM_eval_framework/
    concurrency: 1                # 1 = sequential (best for Ollama)
"""

from __future__ import annotations

# Path shim — makes `python runner.py` work from inside sLLM_eval_framework/
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import asyncio
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from sLLM_eval_framework.adapters.base import ModelAdapter
from sLLM_eval_framework.benchmarks.base import Benchmark, BenchmarkResult
from sLLM_eval_framework.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config models (pydantic)
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    type: str                           # "openai" | "huggingface" | "llamacpp"
    model: Optional[str] = None
    model_path: Optional[str] = None    # llamacpp
    model_id: Optional[str] = None      # huggingface alias
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class SkillConfig(BaseModel):
    name: str
    enabled: Optional[list[str]] = Field(default=None)
    # None  → no registry at all (baseline)
    # []    → load all skills
    # [...]  → load only named skills


class BenchmarkConfig(BaseModel):
    type: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class RunnerConfig(BaseModel):
    models: list[ModelConfig]
    skills_dir: str = "./skills"
    skill_configs: list[SkillConfig] = Field(
        default_factory=lambda: [SkillConfig(name="all_skills", enabled=[])]
    )
    benchmarks: list[BenchmarkConfig]
    runs: int = 1
    output_dir: str = "./results"
    concurrency: int = 4

    @field_validator("runs")
    @classmethod
    def runs_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("runs must be >= 1")
        return v


# ---------------------------------------------------------------------------
# Summary / comparison result
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Aggregated result of a full runner sweep."""
    run_id: str
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[BenchmarkResult] = Field(default_factory=list)

    # Comparison table rows: {model, benchmark, skill_config, score, delta}
    comparison_table: list[dict[str, Any]] = Field(default_factory=list)

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.model_dump(), fh, indent=2, default=str)
        logger.info("Results saved to %s", path)

    def save_csv(self, path: str | Path) -> None:
        if not self.comparison_table:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.comparison_table[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.comparison_table)
        logger.info("CSV saved to %s", path)

    def print_table(self) -> None:
        """Print an ASCII summary table to stdout."""
        if not self.comparison_table:
            print("No results to display.")
            return

        # Determine column widths
        rows = self.comparison_table
        cols = list(rows[0].keys())
        widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

        sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
        header = "| " + " | ".join(str(c).ljust(widths[c]) for c in cols) + " |"

        print(sep)
        print(header)
        print(sep)
        for row in rows:
            line = "| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols) + " |"
            print(line)
        print(sep)


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------


def _build_adapter(cfg: ModelConfig) -> ModelAdapter:
    t = cfg.type.lower()
    if t == "openai":
        from sLLM_eval_framework.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(
            model=cfg.model or "gpt-4o-mini",
            api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=cfg.base_url,
            default_kwargs=cfg.kwargs,
        )
    elif t in ("huggingface", "hf"):
        from sLLM_eval_framework.adapters.huggingface_adapter import HuggingFaceAdapter
        return HuggingFaceAdapter(
            model_id=cfg.model_id or cfg.model or "",
            **cfg.kwargs,
        )
    elif t in ("llamacpp", "llama_cpp", "llama"):
        from sLLM_eval_framework.adapters.llamacpp_adapter import LlamaCppAdapter
        return LlamaCppAdapter(
            model_path=cfg.model_path or "",
            **cfg.kwargs,
        )
    elif t == "ollama":
        from sLLM_eval_framework.adapters.ollama_adapter import OllamaAdapter
        return OllamaAdapter(
            model=cfg.model or "gemma3:4b",
            host=cfg.base_url or "http://localhost:11434",
            default_kwargs=cfg.kwargs,
        )
    elif t == "anthropic":
        from sLLM_eval_framework.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(
            model=cfg.model or "claude-haiku-4-5-20251001",
            api_key=cfg.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=cfg.base_url,
        )
    else:
        raise ValueError(f"Unknown adapter type: {cfg.type!r}")


# ---------------------------------------------------------------------------
# Benchmark factory
# ---------------------------------------------------------------------------


def _build_benchmark(cfg: BenchmarkConfig) -> Benchmark:
    t = cfg.type.lower()
    if t == "skill_selection_accuracy":
        from sLLM_eval_framework.benchmarks.skill_selection import SkillSelectionBenchmark
        return SkillSelectionBenchmark(**cfg.kwargs)
    elif t == "end_to_end_task_completion":
        from sLLM_eval_framework.benchmarks.end_to_end import EndToEndBenchmark
        return EndToEndBenchmark(**cfg.kwargs)
    else:
        raise ValueError(
            f"Unknown benchmark type: {cfg.type!r}. "
            "Register custom benchmarks by passing them directly to EvaluationRunner."
        )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """
    Orchestrates the (model × skill_config × benchmark × run_index) sweep.

    Parameters
    ----------
    config:
        A ``RunnerConfig`` pydantic model.
    extra_benchmarks:
        Additional ``Benchmark`` instances not described in the config
        (useful for programmatic use).
    """

    def __init__(
        self,
        config: RunnerConfig,
        extra_benchmarks: Optional[list[Benchmark]] = None,
    ) -> None:
        self._config = config
        self._extra_benchmarks = extra_benchmarks or []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "EvaluationRunner":
        """Load config from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        data.update(overrides)
        config = RunnerConfig(**data)
        return cls(config)

    @classmethod
    def from_dict(cls, data: dict[str, Any], **overrides: Any) -> "EvaluationRunner":
        data = {**data, **overrides}
        return cls(RunnerConfig(**data))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_all(self) -> RunSummary:
        """Execute the full sweep. Returns a populated RunSummary."""
        cfg = self._config
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        start = time.perf_counter()

        summary = RunSummary(
            run_id=run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            config=cfg.model_dump(),
        )

        # Build objects
        adapters = [_build_adapter(m) for m in cfg.models]
        benchmarks: list[Benchmark] = [_build_benchmark(b) for b in cfg.benchmarks]
        benchmarks.extend(self._extra_benchmarks)

        # Build skill registries for each skill config
        skill_registries: dict[str, Optional[SkillRegistry]] = {}
        for sc in cfg.skill_configs:
            if sc.enabled is None:
                skill_registries[sc.name] = None           # baseline: no skills
            else:
                reg = SkillRegistry(cfg.skills_dir)
                reg.load()
                if sc.enabled:                              # restrict to subset
                    allowed = set(sc.enabled)
                    for name in list(reg._skills.keys()):
                        if name not in allowed:
                            del reg._skills[name]
                skill_registries[sc.name] = reg

        # Build job list
        jobs: list[tuple[ModelAdapter, str, Optional[SkillRegistry], Benchmark, int]] = []
        for adapter in adapters:
            for sc_name, registry in skill_registries.items():
                for benchmark in benchmarks:
                    if registry is None and benchmark.name == "skill_selection_accuracy":
                        logger.info(
                            "Skipping %s for %s/%s (no skills loaded — nothing to route)",
                            benchmark.name, adapter.model_name, sc_name,
                        )
                        continue
                    for run_idx in range(cfg.runs):
                        jobs.append((adapter, sc_name, registry, benchmark, run_idx))

        logger.info(
            "Starting evaluation: %d models × %d skill_configs × %d benchmarks × %d runs = %d jobs",
            len(adapters), len(skill_registries), len(benchmarks), cfg.runs, len(jobs),
        )

        # Run with bounded concurrency
        semaphore = asyncio.Semaphore(cfg.concurrency)
        tasks = [
            self._run_job(semaphore, adapter, sc_name, registry, benchmark, run_idx)
            for adapter, sc_name, registry, benchmark, run_idx in jobs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for job, result in zip(jobs, results):
            if isinstance(result, Exception):
                adapter, sc_name, _, benchmark, run_idx = job
                logger.error(
                    "Job failed [model=%s, skill=%s, bench=%s, run=%d]: %s",
                    adapter.model_name, sc_name, benchmark.name, run_idx, result,
                )
            else:
                summary.results.append(result)

        summary.finished_at = datetime.now(timezone.utc).isoformat()
        summary.duration_s = time.perf_counter() - start
        summary.comparison_table = _build_comparison_table(summary.results)

        # Auto-save per-run files
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.save_json(output_dir / f"{run_id}_results.json")
        summary.save_csv(output_dir / f"{run_id}_summary.csv")

        # Auto-merge all results in the output directory into aggregated_results.json.
        # Wrapped in try/except so a merge failure never blocks the primary output.
        try:
            from merge_results import auto_merge
            agg_path = auto_merge(output_dir)
            if agg_path:
                logger.info("Aggregated results → %s", agg_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-merge failed (per-run file is still saved): %s", exc)

        return summary

    async def _run_job(
        self,
        sem: asyncio.Semaphore,
        adapter: ModelAdapter,
        sc_name: str,
        registry: Optional[SkillRegistry],
        benchmark: Benchmark,
        run_idx: int,
    ) -> BenchmarkResult:
        async with sem:
            logger.info(
                "Running [model=%s | skills=%s | bench=%s | run=%d]",
                adapter.model_name, sc_name, benchmark.name, run_idx,
            )
            result = await benchmark.run(adapter, registry)
            result.metadata["skill_config_name"] = sc_name
            result.metadata["run_index"] = run_idx
            return result


# ---------------------------------------------------------------------------
# Comparison / delta utilities
# ---------------------------------------------------------------------------


def _build_comparison_table(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    """
    Build a flat list of rows for the model × benchmark matrix.

    Adds a ``skill_delta`` column showing the score improvement when skills
    are enabled vs. the baseline run with the same model + benchmark.
    """
    import statistics
    from collections import defaultdict

    # Group by (model, benchmark, skill_config_name)
    # key: (model_name, benchmark_name) → {skill_config_name: [scores]}
    data: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latency: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    tokens: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    test_counts: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for r in results:
        sc_name = r.metadata.get("skill_config_name", "unknown")
        key = (r.model_name, r.benchmark_name)
        data[key][sc_name].append(r.score)
        latency[(r.model_name, r.benchmark_name, sc_name)].append(r.avg_latency_ms)
        tokens[(r.model_name, r.benchmark_name, sc_name)].append(r.total_tokens)
        test_counts[(r.model_name, r.benchmark_name, sc_name)].append(r.total_tests)

    rows: list[dict[str, Any]] = []
    for (model, bench), sc_scores in sorted(data.items()):
        # Find "no_skills" baseline (exact name or any config with no enabled skills)
        baseline_score: Optional[float] = None
        if "no_skills" in sc_scores:
            baseline_score = sum(sc_scores["no_skills"]) / len(sc_scores["no_skills"])

        for sc_name, scores in sorted(sc_scores.items()):
            avg_score = sum(scores) / len(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

            lat_vals = latency[(model, bench, sc_name)]
            avg_lat = sum(lat_vals) / len(lat_vals) if lat_vals else 0.0
            std_lat = statistics.stdev(lat_vals) if len(lat_vals) > 1 else 0.0

            avg_tok = (
                sum(tokens[(model, bench, sc_name)]) / len(tokens[(model, bench, sc_name)])
                if tokens[(model, bench, sc_name)] else 0
            )

            delta: str
            if sc_name == "no_skills":
                delta = "—"
            elif baseline_score is not None:
                d = avg_score - baseline_score
                delta = f"{d:+.3f}"
            else:
                delta = "n/a"

            # Average number of test cases (important: with_skills and
            # no_skills conditions may run different numbers of cases)
            tc_key = (model, bench, sc_name)
            avg_cases = (
                sum(test_counts[tc_key]) / len(test_counts[tc_key])
                if test_counts[tc_key] else 0
            )

            rows.append({
                "model": model,
                "benchmark": bench,
                "skill_config": sc_name,
                "score": f"{avg_score:.3f}",
                "score_std": f"{std_score:.3f}",
                "pass_rate": f"{avg_score:.1%}",
                "n_cases": int(avg_cases),
                "avg_latency_ms": f"{avg_lat:.1f}",
                "latency_std": f"{std_lat:.1f}",
                "avg_tokens": f"{avg_tok:.0f}",
                "skill_delta": delta,
                "n_runs": len(scores),
            })

    return rows


def compare_results(
    results: list[BenchmarkResult],
    with_skills_config: str = "all_skills",
    without_skills_config: str = "no_skills",
) -> list[dict[str, Any]]:
    """
    Return a comparison table focusing on the with_skills vs without_skills delta.

    Parameters
    ----------
    results:
        List of BenchmarkResult objects (from RunSummary.results).
    with_skills_config:
        The ``skill_config_name`` to treat as the "with skills" condition.
    without_skills_config:
        The ``skill_config_name`` to treat as the "without skills" baseline.

    Returns
    -------
    List of dicts with keys: model, benchmark, with_skills_score,
    without_skills_score, delta.
    """
    from collections import defaultdict

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        sc = r.metadata.get("skill_config_name", "unknown")
        grouped[(r.model_name, r.benchmark_name)][sc].append(r.score)

    comparison = []
    for (model, bench), sc_map in sorted(grouped.items()):
        with_scores = sc_map.get(with_skills_config, [])
        without_scores = sc_map.get(without_skills_config, [])

        w_score = sum(with_scores) / len(with_scores) if with_scores else None
        wo_score = sum(without_scores) / len(without_scores) if without_scores else None
        delta = (w_score - wo_score) if (w_score is not None and wo_score is not None) else None

        comparison.append({
            "model": model,
            "benchmark": bench,
            f"{with_skills_config}_score": f"{w_score:.3f}" if w_score is not None else "—",
            f"{without_skills_config}_score": f"{wo_score:.3f}" if wo_score is not None else "—",
            "delta (with - without)": f"{delta:+.3f}" if delta is not None else "—",
        })
    return comparison


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _main_async(config_path: str, output: Optional[str] = None) -> None:
    runner = EvaluationRunner.from_yaml(
        config_path,
        **({"output_dir": output} if output else {}),
    )
    summary = await runner.run_all()
    summary.print_table()
    print(f"\nRun ID : {summary.run_id}")
    print(f"Duration: {summary.duration_s:.1f}s")
    print(f"Results : {len(summary.results)} benchmark result(s) saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM skill-evaluation framework runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    asyncio.run(_main_async(args.config, args.output))
