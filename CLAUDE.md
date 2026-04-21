# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Cheng is a graduate student in a **Generative AI class at Vanderbilt**. This is his course project. The research question: **"Can small LLMs (<20B) with skill/tool augmentation match or exceed larger LLMs without tools?"** The rubric has categories: Problem Statement (10), Methodology (50), Implementation (20), Assessment (15), Model Cards (5), Critical Analysis (10), Documentation (5), Presentation (10) — max 135 pts. **Timeline is tight (1 week).**

The comparison has two arms: (a) local 10-model sweep with/without tools (`config_ollama.yaml`) and (b) frontier small models no-tools baseline (`config_frontier.yaml`: gpt-4.1-mini/nano, gpt-5.4-mini) — so "larger LLMs without tools" spans both the 20B local upper bound and paid-API frontier models.

## Common Commands

All commands run from the **`sLLM_eval_framework/` project root** (the package directory itself — there is no longer a nested subfolder). A `sys.path` shim in `runner.py` and `analyze.py` adds the parent directory so `from sLLM_eval_framework...` imports resolve.

```bash
# from the sLLM_eval_framework/ project root

# Run tests (no Ollama needed — uses mock adapters)
pytest tests/ -v

# Run a single test file / single test function
pytest tests/test_benchmarks.py -v
pytest tests/test_benchmarks.py::test_name -v

# Quick smoke test (3 Ministral models, ~10-20 min, requires Ollama running)
python runner.py --config config_quick.yaml --verbose

# Full 10-model sweep (~3 hours with runs=3 on 24GB M-series)
python runner.py --config config_ollama.yaml --verbose

# Frontier no-tools baseline (paid OpenAI APIs; requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python runner.py --config config_frontier.yaml --verbose

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
| gemma4:e2b | 2B | Gemma | Smallest uplift (+0.152) — historical JSON-formatting struggle largely mitigated by the skill-selection prompt |
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
  ├── Adapters: OllamaAdapter (primary), OpenAI, Anthropic, HuggingFace, LlamaCpp
  ├── Skills: calculator, unit_converter (+ clinical lab units), dictionary, datetime_calc, powerlifting (IPF Dots)
  ├── Benchmarks: skill_selection (33 cases), end_to_end (33 cases)
  ├── Skill configs: all_skills (5 tools) vs no_skills (baseline)
  └── Output: per-run `results/<run_id>_{results.json, summary.csv, failures.jsonl}` plus `results/aggregated_results.json` (auto-merged by runner.py after every sweep)
```

Three pluggable ABCs in `base.py` files: `ModelAdapter`, `Benchmark`, `SkillRegistry`.

## Critical Bugs That Were Fixed (don't re-introduce these)

1. **no_skills test case filtering** (`end_to_end.py`): The filter must be `if c.get("skill") and skills is not None and c["skill"] not in skills`. Using `skills is None` to skip cases was wrong — it caused no_skills to run only 2 trivial baseline cases instead of the full 20, making skill_delta meaningless. Now no_skills runs all 20 cases (same as all_skills) but without tool definitions injected.

2. **Thinking model parsing** (both benchmarks): Qwen3.5 emits `<think>...</think>` blocks. Two regex passes needed — one for closed tags, one for unclosed `<think>` tags (model sometimes doesn't close them). Without stripping, skill_selection accuracy drops to ~28%.

3. **Dictionary scoring** (`end_to_end.py`): Exact string match is impossible for definitions. Uses keyword-overlap scoring (60% threshold on words >= 3 chars).

4. **Path/import issues**: `runner.py` and `analyze.py` have `sys.path` shims that insert the parent directory of the project so `from sLLM_eval_framework...` imports resolve when running scripts directly. Config paths like `./skills` and `./results` are relative to the `sLLM_eval_framework/` project root.

5. **Ollama tool-call recovery** (commits `83be862`, `a50ef55`): skill parsers and `end_to_end.py` now tolerate malformed tool-call wrappers that Ollama's smaller models frequently emit (unbalanced braces, escaped quotes inside JSON strings, tool calls buried after prose, etc.). Don't "simplify" the parser by reverting to a strict JSON parse — the uplift numbers in the latest sweep (`20260421T024743Z`) materially depend on this recovery path.

## Key Design Decisions

- **OllamaAdapter** uses native `/api/chat` endpoint (not OpenAI-compatible wrapper) via httpx, 300s timeout
- **Temperature**: frontier arm uses `0.0` for reproducibility; the local sweep (`config_ollama.yaml`) currently runs at `0.5` with `num_predict=5096` to give thinking models enough budget for `<think>` blocks + answer. Quantized inference has residual non-determinism regardless, which is why `runs=3`.
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
- `config_ollama.yaml` — Full sweep config (10 models, runs=3, temperature=0.5, num_predict=5096)
- `config_quick.yaml` — Smoke test config (3 Ministral models, runs=1)
- `config_frontier.yaml` — Frontier no-tools baseline (gpt-4.1-mini/nano + gpt-5.4-mini, runs=3, concurrency=4)
- `config.yaml` — Generic example template (gpt-4o-mini + commented HF/llama.cpp stanzas). Not used by any sweep — don't confuse with the three real configs above.
- `implementation_plan.md` — Historical plan for the SkillsBench port (lab units + powerlifting). Work is shipped; file is retained for context on why those skills exist.
- `model_cards.md` — Model cards, data card, ethical considerations
- `README.md` — Full documentation including critical analysis section and SkillsBench provenance notes
- `benchmarks/end_to_end.py` — Multi-turn tool-call benchmark with thinking-tag stripping
- `benchmarks/skill_selection.py` — Skill routing accuracy benchmark
- `adapters/ollama_adapter.py` — Native Ollama HTTP adapter
- `adapters/openai_adapter.py`, `adapters/anthropic_adapter.py` — Frontier API adapters (env-var API keys)
- `skills/powerlifting/skill.py` — IPF Dots (2019) coefficient (port from benchflow-ai/skillsbench)
- `benchmarks/utils.py` — Centralized think-tag stripping utilities

## Known Anomalies in Results (as of run 20260421T024743Z)

Latest local sweep incorporates the improved tool-call detection/recovery landed in `83be862` + `a50ef55`. Uplifts jumped materially vs. the `20260417T033055Z` snapshot — the Qwen ladder in particular now sits near ceiling with tools. Verify against `results/aggregated_results.json` before citing in writing.

- **All 10 models still show positive uplift, but the magnitudes are larger.** With better tool-call recovery, `all_skills` scores rise across the board and the no_skills baselines become the dominant contributor to the uplift spread.
- **Qwen3.5 peak shifted from 4B to 9B (barely).** Uplift by size: 2B +0.727, 4B +0.919, 9B +0.939. 9B and 4B are both near ceiling once tools are available; the cleanest framing is "tool access transforms the whole Qwen ladder" rather than any specific "peak at N."
- **Smallest uplift is `gpt-oss:20b` at +0.111**, then `nemotron-3-nano:4b` at +0.182. Gemma4:e2b rose to +0.222 — it is no longer in the bottom two. Nemotron's strong `no_skills` baseline (0.616) is what keeps its uplift small.
- **Skill selection accuracy is NOT uniformly near-perfect anymore.** 7 models hit 1.000, but `gemma4:e4b` 0.970, `ministral-3:3b` 0.939, and `qwen3.5:2b` 0.848 are visible laggards. `qwen3.5:2b` in particular is a real routing-bottleneck case worth flagging — previous "routing is solved" framing doesn't hold for the 2B tier.
- **Temperature was 0.5 (not 0.0) for the latest sweep,** with `num_predict=5096`. The "Temperature = 0.0 for reproducibility" design decision describes the default config intent; the 20260421 run relaxed it. Check `config_ollama.yaml` before citing a specific run's settings.

## Recent Improvements

- **Aggregated results DB** (`merge_results.py`): merge multiple run JSONs into one file so individual models can be run incrementally. Runner auto-merges after each run.
- **Centralized think-tag stripping** (`benchmarks/utils.py`): `strip_think_tags()` and `recover_answer_from_think_block()` shared across both benchmarks.
- **Failure mode tracking**: end-to-end results now record `skill_selected_correctly`, `valid_tool_call_format`, `tool_executed_successfully`, `final_answer_correct` in metadata. Full per-case failure traces (prompt, expected, actual, raw model_output, error) are also exported to `results/<run_id>_failures.jsonl` on every run — use these to diagnose tool-call formatting regressions without re-running the sweep.
- **Std dev + error bars**: comparison table has `score_std` and `latency_std`; charts show ±1 SD error bars when runs>1.
