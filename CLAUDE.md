# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular, async-first Python framework for benchmarking LLM skill-use accuracy across multiple models, skill configurations, and task types. It evaluates whether models can correctly select and execute tools/skills.

## Common Commands

```bash
# Install core dependencies (Python 3.10+)
pip install -r requirements.txt

# Install optional adapter extras
pip install -r requirements.txt -r requirements-hf.txt        # HuggingFace
pip install -r requirements.txt -r requirements-llamacpp.txt   # llama.cpp (CPU)

# Run tests (no API keys needed — uses mock adapters)
pytest tests/ -v

# Run a single test file
pytest tests/test_adapters.py -v

# Run the evaluation sweep
python -m eval_framework.runner --config config.yaml --verbose
```

## Architecture

The framework runs a cross-product sweep: **model × skill_config × benchmark × n_runs**.

```
EvaluationRunner (runner.py)
  ├── Reads RunnerConfig (pydantic model from config.yaml)
  ├── Builds ModelAdapter instances via _build_adapter() factory
  ├── Builds SkillRegistry instances per SkillConfig
  ├── Builds Benchmark instances via _build_benchmark() factory
  ├── Runs all jobs with bounded asyncio.Semaphore concurrency
  └── Produces RunSummary → JSON + CSV in results/
```

**Three pluggable pillars**, each with an ABC in `base.py`:

1. **adapters/** — `ModelAdapter` ABC. Subclass must implement `model_name` property and `async generate()`. Existing: `OpenAIAdapter`, `HuggingFaceAdapter`, `LlamaCppAdapter`. The `OpenAIAdapter` also works with vLLM/Ollama/LM Studio via `base_url`. All adapters share `ModelResponse`, `ToolCall`, `ToolDefinition` data models.

2. **skills/** — Self-contained folders with `SKILL.md` + `skill.py` (must define `SKILL_META` dict and `execute()` function). `SkillRegistry` auto-discovers them by scanning subdirectories. Each skill has `trigger_patterns` (regex) used for matching and `to_tool_definition()` for injecting into model prompts.

3. **benchmarks/** — `Benchmark` ABC with `name`, `description`, and `async run(model, skills)`. Returns a `BenchmarkResult` containing `TestResult` items. Two built-in:
   - `SkillSelectionBenchmark` — tests whether the model routes queries to the correct skill
   - `EndToEndBenchmark` — tests whether the model produces correct final answers (multi-turn tool-call loop)

**Data flow**: `Benchmark.run()` receives a `ModelAdapter` and optional `SkillRegistry`, calls `model.generate()`, optionally executes skills via `skill.execute()`, and returns a finalised `BenchmarkResult`.

## Key Patterns

- All async — benchmarks and adapters use `async/await`; `asyncio.gather` for parallel test execution
- Pydantic models throughout for config (`RunnerConfig`, `ModelConfig`, etc.) and results (`BenchmarkResult`, `TestResult`)
- Skills are hot-pluggable — drop a folder under `skills/` with `skill.py` + `SKILL.md`, the registry auto-loads it
- `SkillConfig.enabled`: `None` = no skills (baseline), `[]` = all skills, `["calculator"]` = subset only
- `ModelAdapter.generate_with_retry()` provides exponential-backoff retries (default 3 attempts)
- Tests use `MockAdapter` and `ToolCallingMockAdapter` — no real API calls needed

## Adding New Components

- **New model adapter**: Subclass `ModelAdapter` in `adapters/<name>_adapter.py`, add a case in `runner._build_adapter()`
- **New skill**: Create `skills/<name>/skill.py` with `SKILL_META` dict (requires `name`, `description`, `trigger_patterns`) and `execute(input: SkillInput) -> SkillOutput`
- **New benchmark**: Subclass `Benchmark` in `benchmarks/<name>.py`, add a case in `runner._build_benchmark()`, or pass via `extra_benchmarks=` kwarg to `EvaluationRunner`