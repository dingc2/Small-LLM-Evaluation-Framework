# LLM Skill-Evaluation Framework

A modular, async-first Python framework for benchmarking LLM skill-use
accuracy across multiple models, skill configurations, and task types.

---

## Directory Structure

```
eval_framework/
├── adapters/                   # ModelAdapter subclasses
│   ├── base.py                 #   ABC + ToolDefinition / ModelResponse
│   ├── openai_adapter.py       #   OpenAI-compatible (also vLLM, Ollama, …)
│   ├── huggingface_adapter.py  #   Local HuggingFace Transformers
│   └── llamacpp_adapter.py     #   GGUF via llama-cpp-python
├── skills/                     # Self-contained skill modules
│   ├── registry.py             #   Dynamic loader + SkillRegistry
│   └── calculator/             #   Example skill
│       ├── SKILL.md            #     Description & trigger patterns
│       └── skill.py            #     SKILL_META + execute()
├── benchmarks/                 # Benchmark subclasses
│   ├── base.py                 #   ABC + BenchmarkResult / TestCase / TestResult
│   ├── skill_selection.py      #   "Which skill should be called?"
│   └── end_to_end.py           #   "Does the model get the right answer?"
├── runner.py                   # Orchestrator (EvaluationRunner)
├── config.yaml                 # Example configuration
├── results/                    # Auto-created output directory
├── tests/
│   ├── test_adapters.py
│   ├── test_registry.py
│   └── test_benchmarks.py
├── requirements.txt
├── requirements-hf.txt         # Optional: HuggingFace extras
└── requirements-llamacpp.txt   # Optional: llama.cpp extras
```

---

## Installation

```bash
# 1. Clone / copy the eval_framework/ folder into your project root.

# 2. Install core dependencies (Python 3.10+ required):
pip install -r eval_framework/requirements.txt

# 3. Optional: HuggingFace local models
pip install -r eval_framework/requirements-hf.txt

# 4. Optional: llama.cpp GGUF models (CPU)
pip install -r eval_framework/requirements-llamacpp.txt
# GPU (CUDA):
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```

---

## Quickstart

### Run from YAML config

```bash
# Set your OpenAI key:
export OPENAI_API_KEY=sk-...

# Run the full evaluation sweep:
cd eval_framework
python -m eval_framework.runner --config config.yaml --verbose
```

Results are auto-saved to `results/<run_id>_results.json` and
`results/<run_id>_summary.csv`.

---

### Programmatic usage

```python
import asyncio
from eval_framework.runner import EvaluationRunner

async def main():
    runner = EvaluationRunner.from_yaml("eval_framework/config.yaml")
    summary = await runner.run_all()
    summary.print_table()           # ASCII comparison table
    summary.save_json("out.json")   # Full JSON results
    summary.save_csv("out.csv")     # Flat CSV for spreadsheets

asyncio.run(main())
```

### With-skills vs without-skills delta

```python
from eval_framework.runner import compare_results

delta_table = compare_results(
    summary.results,
    with_skills_config="all_skills",
    without_skills_config="no_skills",
)
for row in delta_table:
    print(row)
```

---

## Adding a New Model

Create a new file in `adapters/` that subclasses `ModelAdapter`:

```python
# adapters/my_provider_adapter.py
from eval_framework.adapters.base import ModelAdapter, ModelResponse, ToolDefinition

class MyProviderAdapter(ModelAdapter):

    @property
    def model_name(self) -> str:
        return "my-model-v1"

    async def generate(self, prompt, tools=None, system_prompt=None, **kwargs) -> ModelResponse:
        # ... call your API here ...
        return ModelResponse(content="...", model_name=self.model_name)
```

Then reference it in `config.yaml`:

```yaml
models:
  - type: my_provider       # add a case in runner._build_adapter()
    model: my-model-v1
```

Or pass it directly to the runner:

```python
from eval_framework.runner import EvaluationRunner, RunnerConfig, ModelConfig
from my_adapter import MyProviderAdapter

# Bypass the config factory entirely — inject the adapter at runtime
runner = EvaluationRunner(config)
# Use the programmatic API described below.
```

---

## Adding a New Skill

1. Create a subfolder under `skills/`:

```
skills/
  web_search/
    SKILL.md       ← human-readable description (not parsed by the framework)
    skill.py       ← must define SKILL_META and execute()
```

2. Define `SKILL_META` and `execute()` in `skill.py`:

```python
# skills/web_search/skill.py
SKILL_META = {
    "name": "web_search",
    "description": "Searches the web and returns the top results.",
    "trigger_patterns": [r"\bsearch\b", r"\blook up\b", r"\bfind\b"],
    "version": "1.0.0",
}

def execute(input):          # sync or async — both work
    from eval_framework.skills.registry import SkillOutput
    query = input.query
    # ... call your search API ...
    return SkillOutput(result="Top result: ...", success=True)
```

3. That's it.  The registry picks it up automatically on the next `load()`.

---

## Adding a New Benchmark

Subclass `Benchmark` and implement `run()`:

```python
# benchmarks/my_benchmark.py
from eval_framework.benchmarks.base import Benchmark, BenchmarkResult, TestResult

class MyBenchmark(Benchmark):
    name = "my_benchmark"
    description = "Tests something specific."

    async def run(self, model, skills=None, **kwargs) -> BenchmarkResult:
        result, t0 = self._make_result(model.model_name, self.name, skills.names if skills else [])

        response = await model.generate("Some prompt")
        passed = "expected" in response.content.lower()
        result.test_results.append(TestResult(
            test_id="test_01",
            passed=passed,
            score=1.0 if passed else 0.0,
            model_output=response.content,
            expected="expected",
            latency_ms=response.latency_ms,
        ))
        return self._close_result(result, t0)
```

Register it in `config.yaml` or pass it directly:

```python
runner = EvaluationRunner(config, extra_benchmarks=[MyBenchmark()])
```

---

## Config Reference

| Key | Type | Default | Description |
|---|---|---|---|
| `models` | list | required | One entry per model/adapter |
| `skills_dir` | str | `./skills` | Path to skill folder tree |
| `skill_configs` | list | all skills | Named skill enable/disable sets |
| `benchmarks` | list | required | Benchmark types to run |
| `runs` | int | `1` | Repeat count (for variance estimation) |
| `concurrency` | int | `4` | Max parallel model API calls |
| `output_dir` | str | `./results` | Where to write JSON + CSV |

### `skill_configs` semantics

| `enabled` value | Behaviour |
|---|---|
| `[]` (empty list) | Load **all** skills from `skills_dir` |
| `["calculator"]` | Load **only** the named skills |
| `null` | **No** skill registry — pure model baseline |

---

## Result Output

### JSON (`results/<run_id>_results.json`)

Full structured output including per-test-case scores, latencies, token counts,
and metadata.

```json
{
  "run_id": "20240501T120000Z",
  "results": [
    {
      "benchmark_name": "skill_selection_accuracy",
      "model_name": "gpt-4o-mini",
      "skill_config": ["calculator"],
      "score": 0.923,
      "avg_latency_ms": 312.4,
      "total_tokens": 1840,
      ...
    }
  ],
  "comparison_table": [...]
}
```

### CSV (`results/<run_id>_summary.csv`)

Flat table with one row per `(model, benchmark, skill_config)`:

```
model,benchmark,skill_config,score,pass_rate,avg_latency_ms,avg_tokens,skill_delta
gpt-4o-mini,skill_selection_accuracy,all_skills,0.923,92.3%,312.4,1840,+0.154
gpt-4o-mini,skill_selection_accuracy,no_skills,0.769,76.9%,289.1,1620,—
```

The `skill_delta` column is `all_skills_score − no_skills_score`, showing
the performance lift from enabling skills.

---

## Running Tests

```bash
cd eval_framework
pytest tests/ -v
```

Tests use mock adapters — no real API calls, no GPU required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        EvaluationRunner                          │
│  reads RunnerConfig → builds adapters, registries, benchmarks    │
│  runs cross-product: model × skill_config × benchmark × n_runs  │
│  collects BenchmarkResult list → comparison_table → JSON + CSV  │
└──────────┬──────────────────┬───────────────────────────────────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼──────────┐
    │ ModelAdapter│    │  SkillRegistry   │
    │   (ABC)     │    │  .load()         │
    ├─────────────┤    │  .get(name)      │
    │ OpenAI      │    │  .find_matching()│
    │ HuggingFace │    └──────┬───────────┘
    │ LlamaCpp    │           │ loads
    └──────┬──────┘    ┌──────▼───────────┐
           │           │  Skill           │
    ┌──────▼──────┐    │  .matches(query) │
    │  Benchmark  │    │  .execute(input) │
    │  (ABC)      │    └──────────────────┘
    ├─────────────┤
    │ SkillSelect │  → BenchmarkResult → TestResult[]
    │ EndToEnd    │
    └─────────────┘
```

---

## License

MIT
