# CLAUDE.md

## Project Context

Cheng is a graduate student in a **Generative AI class at Vanderbilt**. This is his course project. The research question: **"Can small LLMs (<20B) with skill/tool augmentation match or exceed larger LLMs without tools?"** The rubric has categories: Problem Statement (10), Methodology (50), Implementation (20), Assessment (15), Model Cards (5), Critical Analysis (10), Documentation (5), Presentation (10) — max 135 pts. **Timeline is tight (1 week).**

## Common Commands

All commands run from the **`sLLM_eval_framework/` project root** (the package directory itself — there is no longer a nested subfolder). A `sys.path` shim in `runner.py` and `analyze.py` adds the parent directory so `from sLLM_eval_framework...` imports resolve.

```bash
# from the sLLM_eval_framework/ project root

# Run tests (no Ollama needed — uses mock adapters)
pytest tests/ -v

# Quick smoke test (3 Ministral models, ~10-20 min, requires Ollama running)
python runner.py --config config_quick.yaml --verbose

# Full 10-model sweep (~3 hours with runs=3 on 24GB M-series)
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
  ├── Adapters: OllamaAdapter (primary), OpenAI, HuggingFace, LlamaCpp
  ├── Skills: calculator, unit_converter (+ clinical lab units), dictionary, datetime_calc, powerlifting (IPF Dots)
  ├── Benchmarks: skill_selection (33 cases), end_to_end (33 cases)
  ├── Skill configs: all_skills (5 tools) vs no_skills (baseline)
  └── Output: JSON + CSV in results/, comparison table with n_cases column
```

Three pluggable ABCs in `base.py` files: `ModelAdapter`, `Benchmark`, `SkillRegistry`.

## Critical Bugs That Were Fixed (don't re-introduce these)

1. **no_skills test case filtering** (`end_to_end.py`): The filter must be `if c.get("skill") and skills is not None and c["skill"] not in skills`. Using `skills is None` to skip cases was wrong — it caused no_skills to run only 2 trivial baseline cases instead of the full 20, making skill_delta meaningless. Now no_skills runs all 20 cases (same as all_skills) but without tool definitions injected.

2. **Thinking model parsing** (both benchmarks): Qwen3.5 emits `<think>...</think>` blocks. Two regex passes needed — one for closed tags, one for unclosed `<think>` tags (model sometimes doesn't close them). Without stripping, skill_selection accuracy drops to ~28%.

3. **Dictionary scoring** (`end_to_end.py`): Exact string match is impossible for definitions. Uses keyword-overlap scoring (60% threshold on words >= 3 chars).

4. **Path/import issues**: `runner.py` and `analyze.py` have `sys.path` shims that insert the parent directory of the project so `from sLLM_eval_framework...` imports resolve when running scripts directly. Config paths like `./skills` and `./results` are relative to the `sLLM_eval_framework/` project root.

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
- `README.md` — Full documentation including critical analysis section and SkillsBench provenance notes
- `benchmarks/end_to_end.py` — Multi-turn tool-call benchmark with thinking-tag stripping
- `benchmarks/skill_selection.py` — Skill routing accuracy benchmark
- `adapters/ollama_adapter.py` — Native Ollama HTTP adapter
- `skills/powerlifting/skill.py` — IPF Dots (2019) coefficient (port from benchflow-ai/skillsbench)
- `benchmarks/utils.py` — Centralized think-tag stripping utilities

## Known Anomalies in Results (as of run 20260417T033055Z)

- **All 10 models now show positive uplift.** The earlier Gemma4:e2b *negative* uplift (seen in the `20260412T215302Z` run) is gone. With the 33-case benchmark, gemma4:e2b is +0.152 — still the smallest uplift in the sweep, but positive. The "formatting as a distinct capability" story survives as "smallest uplift" rather than "negative uplift."
- **Qwen3.5 does NOT show clean diminishing returns anymore.** Uplift by size: 2B +0.576, 4B +0.808 (largest), 9B +0.758. The 4B is the peak, not the 2B. The more honest framing is "small-to-mid Qwen models gain the most from tools; the 20B upper bound leaves less headroom (gpt-oss:20b: +0.121)."
- **Smallest uplift is actually nemotron-3-nano:4b (+0.141)** and gpt-oss:20b (+0.121) — not gemma4:e2b. Nemotron's baseline is surprisingly strong for a 4B model.
- **Skill selection accuracy is near-perfect** across the board: 9/10 models at ≥0.94, most at 1.00. Routing is not the bottleneck; tool-call formatting is.

## What Still Needs Doing

- **Full sweep complete** (2026-04-17, run `20260417T033055Z`, ~3 hours wall-clock) — charts regenerated via `analyze.py`.
- Consider creating the final **presentation slides** (.pptx) for the class — rubric allocates 10 pts.

## Recent Improvements

- **Aggregated results DB** (`merge_results.py`): merge multiple run JSONs into one file so individual models can be run incrementally. Runner auto-merges after each run.
- **Centralized think-tag stripping** (`benchmarks/utils.py`): `strip_think_tags()` and `recover_answer_from_think_block()` shared across both benchmarks.
- **Failure mode tracking**: end-to-end results now record `skill_selected_correctly`, `valid_tool_call_format`, `tool_executed_successfully`, `final_answer_correct` in metadata.
- **Std dev + error bars**: comparison table has `score_std` and `latency_std`; charts show ±1 SD error bars when runs>1.
