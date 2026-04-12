# CLAUDE.md

## Project Context

Cheng is a graduate student in a **Generative AI class at Vanderbilt**. This is his course project. The research question: **"Can small LLMs (<20B) with skill/tool augmentation match or exceed larger LLMs without tools?"** The rubric has categories: Problem Statement (10), Methodology (50), Implementation (20), Assessment (15), Model Cards (5), Critical Analysis (10), Documentation (5), Presentation (10) — max 135 pts. **Timeline is tight (1 week).**

## Common Commands

All commands run from **inside `eval_framework/`** (not the parent directory). A `sys.path` shim in `runner.py` and `analyze.py` makes this work.

```bash
cd eval_framework

# Run tests (no Ollama needed — uses mock adapters)
pytest tests/ -v

# Quick smoke test (3 Ministral models, ~10-20 min, requires Ollama running)
python runner.py --config config_quick.yaml --verbose

# Full 10-model sweep (~150-200 min with runs=3)
python runner.py --config config_ollama.yaml --verbose

# Merge individual model runs into one aggregated file
python merge_results.py results/

# Generate charts from results (single run or aggregated)
python analyze.py results/aggregated_results.json

# Standalone chart generation (uses hardcoded results from a completed sweep)
python charts_gen.py
```

## The 10 Models (all via Ollama on 24GB MacBook Pro)

| Model | Params | Family | Notes |
|---|---|---|---|
| qwen3.5:2b | 2B | Qwen | Thinking model — emits `<think>` blocks |
| qwen3.5:4b | 4B | Qwen | Thinking model |
| qwen3.5:9b | 9B | Qwen | Thinking model |
| gemma4:e2b | 2B | Gemma | Struggles with tool-call JSON formatting |
| gemma4:e4b | 4B | Gemma | |
| nemotron-3-nano:4b | 4B | Nemotron | NVIDIA edge-optimized |
| gpt-oss:20b | 20B | GPT-OSS | Largest model, serves as upper bound |
| ministral-3:3b | 3B | Ministral | |
| ministral-3:8b | 8B | Ministral | |
| ministral-3:14b | 14B | Ministral | |

## Architecture

Cross-product sweep: **model x skill_config x benchmark x n_runs**

```
EvaluationRunner (runner.py)
  ├── Adapters: OllamaAdapter (primary), OpenAI, HuggingFace, LlamaCpp
  ├── Skills: calculator, unit_converter, dictionary, datetime_calc
  ├── Benchmarks: skill_selection (25 cases), end_to_end (20 cases)
  ├── Skill configs: all_skills (4 tools) vs no_skills (baseline)
  └── Output: JSON + CSV in results/, comparison table with n_cases column
```

Three pluggable ABCs in `base.py` files: `ModelAdapter`, `Benchmark`, `SkillRegistry`.

## Critical Bugs That Were Fixed (don't re-introduce these)

1. **no_skills test case filtering** (`end_to_end.py`): The filter must be `if c.get("skill") and skills is not None and c["skill"] not in skills`. Using `skills is None` to skip cases was wrong — it caused no_skills to run only 2 trivial baseline cases instead of the full 20, making skill_delta meaningless. Now no_skills runs all 20 cases (same as all_skills) but without tool definitions injected.

2. **Thinking model parsing** (both benchmarks): Qwen3.5 emits `<think>...</think>` blocks. Two regex passes needed — one for closed tags, one for unclosed `<think>` tags (model sometimes doesn't close them). Without stripping, skill_selection accuracy drops to ~28%.

3. **Dictionary scoring** (`end_to_end.py`): Exact string match is impossible for definitions. Uses keyword-overlap scoring (60% threshold on words >= 3 chars).

4. **Path/import issues**: `runner.py` and `analyze.py` have `sys.path` shims so they work when run from inside `eval_framework/`. Config paths like `./skills` and `./results` are relative to `eval_framework/`.

## Key Design Decisions

- **OllamaAdapter** uses native `/api/chat` endpoint (not OpenAI-compatible wrapper) via httpx, 300s timeout
- **Temperature = 0.0** for reproducibility (but quantized inference still has some non-determinism)
- **All models use Ollama default quantization** (typically Q4_K_M) — this is a known confound
- **Concurrency = 1** for Ollama (sequential model loading/unloading)
- **Keyword-overlap** scorer for dictionary (not exact match, not BERTScore)
- **Skill selection benchmark** uses last-line extraction after stripping think tags

## SkillConfig Semantics

- `enabled: null` → no_skills baseline (no registry loaded at all)
- `enabled: []` → all_skills (load everything)
- `enabled: ["calculator", "dictionary"]` → load only those skills

## Files of Note

- `runner.py` — Main orchestrator, CLI entry point, comparison table builder (includes score_std/latency_std)
- `analyze.py` — Reads result JSON, generates 5 chart types (with error bars when runs>1)
- `merge_results.py` — Merges multiple run JSONs into aggregated_results.json (incremental runs)
- `charts_gen.py` — Standalone chart script with hardcoded results (update after re-runs)
- `config_ollama.yaml` — Full sweep config (10 models, runs=3)
- `config_quick.yaml` — Smoke test config (3 Ministral models, runs=1)
- `model_cards.md` — Model cards, data card, ethical considerations
- `README.md` — Full documentation including critical analysis section
- `benchmarks/end_to_end.py` — Multi-turn tool-call benchmark with thinking-tag stripping
- `benchmarks/skill_selection.py` — Skill routing accuracy benchmark
- `adapters/ollama_adapter.py` — Native Ollama HTTP adapter

## Known Anomalies in Results

- **Gemma4:e2b negative uplift**: Scores lower with tools because it can't format tool-call JSON reliably. This is actually an interesting finding about formatting as a distinct capability.
- **Qwen3.5 diminishing returns**: Skill uplift decreases as model size increases (2B: +55%, 9B: +25%) because larger models already perform well without tools.

## What Still Needs Doing

- **Re-run full sweep** with `config_ollama.yaml` — previous results were from before all bug fixes. Use `merge_results.py` to aggregate runs across sessions.
- After re-run: update hardcoded values in `charts_gen.py` with new results and regenerate charts.
- Consider creating the final **presentation slides** (.pptx) for the class — rubric allocates 10 pts.

## Recent Improvements

- **Aggregated results DB** (`merge_results.py`): merge multiple run JSONs into one file so individual models can be run incrementally. Runner auto-merges after each run.
- **Centralized think-tag stripping** (`benchmarks/utils.py`): `strip_think_tags()` and `recover_answer_from_think_block()` shared across both benchmarks.
- **Failure mode tracking**: end-to-end results now record `skill_selected_correctly`, `valid_tool_call_format`, `tool_executed_successfully`, `final_answer_correct` in metadata.
- **Std dev + error bars**: comparison table has `score_std` and `latency_std`; charts show ±1 SD error bars when runs>1.
