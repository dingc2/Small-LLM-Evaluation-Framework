# Small-LLM Skill-Uplift Evaluation Framework

**Research Question:** Can small open-source LLMs (<20B parameters) with access to
skill/tool augmentation match or exceed larger models operating without tools?

This framework evaluates the **skill uplift** — the performance improvement gained
when small language models are given access to external tools (calculator, unit
converter, dictionary, date/time calculator) — and compares models across the Gemma,
Llama, and Qwen families.

---

## Problem Statement

Large language models (LLMs) demonstrate impressive reasoning capabilities, but
deploying them requires significant computational resources. Small LLMs (1–20B
parameters) can run locally on consumer hardware but often lack the raw accuracy of
their larger counterparts.

**Hypothesis:** Augmenting small LLMs with structured tool access (skills) can
significantly close the performance gap with larger models, making them viable for
many practical applications on resource-constrained hardware.

This project evaluates this hypothesis by:

1. Testing 7 models across 4 families (Qwen, Gemma, Nemotron, GPT-OSS) at sizes from 2B to 20B
2. Measuring performance with and without skill augmentation
3. Quantifying the "skill uplift" — the score delta between tool-augmented and
   baseline conditions
4. Analysing the tradeoff between model size, latency, and accuracy

---

## Directory Structure

```
eval_framework/
├── adapters/                    # Model adapter layer
│   ├── base.py                  #   ABC + ToolDefinition / ModelResponse
│   ├── openai_adapter.py        #   OpenAI-compatible API
│   ├── ollama_adapter.py        #   Ollama native API (recommended for local)
│   ├── huggingface_adapter.py   #   Local HuggingFace Transformers
│   └── llamacpp_adapter.py      #   GGUF via llama-cpp-python
├── skills/                      # Self-contained skill modules
│   ├── registry.py              #   Dynamic loader + SkillRegistry
│   ├── calculator/              #   Arithmetic expression evaluator
│   ├── unit_converter/          #   Measurement unit conversion
│   ├── dictionary/              #   Word definition lookup
│   └── datetime_calc/           #   Date arithmetic & day-of-week
├── benchmarks/                  # Evaluation benchmarks
│   ├── base.py                  #   ABC + BenchmarkResult / TestCase / TestResult
│   ├── skill_selection.py       #   "Which skill should be called?"
│   └── end_to_end.py            #   "Does the model get the right answer?"
├── runner.py                    # Orchestrator (EvaluationRunner)
├── analyze.py                   # Result analysis & chart generation
├── config_ollama.yaml           # Full evaluation config (6 models × 2 conditions)
├── config_quick.yaml            # Quick smoke-test config (1 model)
├── config.yaml                  # Example OpenAI API config
├── model_cards.md               # Model cards & ethical considerations
├── results/                     # Auto-created output directory
├── tests/
│   ├── test_adapters.py
│   ├── test_registry.py
│   └── test_benchmarks.py
├── requirements.txt
├── requirements-hf.txt
└── requirements-llamacpp.txt
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local model inference): https://ollama.com
- **24 GB MacBook Pro** (or equivalent — all tested models fit in 24 GB)

### Step 1: Install Dependencies

```bash
cd eval_framework
pip install -r requirements.txt
```

### Step 2: Install Ollama & Pull Models

```bash
# Install Ollama (macOS)
brew install ollama
# Or download from https://ollama.com

# Start the Ollama server
ollama serve

# Pull models (in a separate terminal)
ollama pull qwen3.5:2b          # ~2.7 GB
ollama pull qwen3.5:4b          # ~3.4 GB
ollama pull qwen3.5:9b          # ~6.6 GB
ollama pull gemma4:e2b          # ~7.2 GB
ollama pull gemma4:e4b          # ~9.6 GB
ollama pull nemotron-3-nano:4b  # ~2.8 GB
ollama pull gpt-oss:20b         # ~13 GB
```

### Step 3: Smoke Test

```bash
# All commands run from inside eval_framework/
cd eval_framework

# Quick test with a single model (~2–3 min)
python runner.py --config config_quick.yaml --verbose
```

### Step 4: Full Evaluation

```bash
# Full 7-model × 2-condition sweep (~45–90 min)
python runner.py --config config_ollama.yaml --verbose
```

### Step 5: Analyse Results

```bash
# Generate summary table + 5 charts
python analyze.py results/<run_id>_results.json
```

The `<run_id>` is printed when the run finishes (e.g. `20260411T120000Z`). Results and charts land in `results/`.

---

## Methodology

### Experimental Design

The evaluation uses a **2 × 7 × 2** factorial design:

| Factor | Levels |
|---|---|
| **Skill Condition** | `all_skills` (4 tools available) vs `no_skills` (baseline) |
| **Model** | qwen3.5:2b, qwen3.5:4b, qwen3.5:9b, gemma4:e2b, gemma4:e4b, nemotron-3-nano:4b, gpt-oss:20b |
| **Benchmark** | Skill Selection Accuracy, End-to-End Task Completion |

Each combination is repeated 3 times for variance estimation.

### Skills (Tools)

| Skill | Purpose | Example |
|---|---|---|
| **Calculator** | Arithmetic & math functions | `sqrt(625)` → `25.0` |
| **Unit Converter** | Measurement conversion | `5 km to miles` → `3.107` |
| **Dictionary** | Word definitions | `define ephemeral` → `lasting for a very short time` |
| **Date/Time Calc** | Date arithmetic | `days between 2024-01-01 and 2024-12-31` → `365` |

### Benchmarks

**1. Skill Selection Accuracy** — Can the model route a query to the correct tool?
- 25 test cases: 5 per skill + 5 negative ("none") cases
- Binary scoring: exact match = 1.0, otherwise 0.0
- Protocol: system prompt lists available skills; model must output ONLY the skill name

**2. End-to-End Task Completion** — Does the model get the right final answer?
- 20 test cases spanning all 4 skills + 2 no-tool baselines
- Multi-turn: model can call a tool, receive the result, and produce a final answer
- Scoring: numeric tolerance (±0.01) for math, exact match for strings

### Key Metric: Skill Uplift

```
skill_uplift = score(model, all_skills) − score(model, no_skills)
```

A positive uplift indicates the model successfully leverages tools to improve
performance beyond its raw reasoning capability.

---

## Results & Analysis

After running the evaluation, use the analysis module:

```bash
python analyze.py results/<run_id>_results.json
```

This generates:

| Output | Description |
|---|---|
| Terminal summary table | Score, pass rate, latency, skill delta per model |
| `skill_uplift.png` | Bar chart of skill uplift per model |
| `score_heatmap.png` | Heatmap of scores across all conditions |
| `latency_comparison.png` | Inference latency by model |
| `size_vs_score.png` | Model parameters vs. performance scatter |
| `per_skill_breakdown.png` | Per-skill pass rates |

---

## Extending the Framework

### Adding a New Model

The Ollama adapter supports any model available in the Ollama library.
Just pull it and add to the config:

```bash
ollama pull mistral:7b
```

```yaml
models:
  - type: ollama
    model: mistral:7b
    kwargs:
      temperature: 0.0
```

### Adding a New Skill

Create a folder under `skills/` with `skill.py` defining `SKILL_META` and `execute()`:

```python
# skills/my_skill/skill.py
SKILL_META = {
    "name": "my_skill",
    "description": "Does something useful.",
    "trigger_patterns": [r"\bmy_keyword\b"],
}

def execute(input):
    from eval_framework.skills.registry import SkillOutput
    return SkillOutput(result="...", success=True)
```

The registry auto-discovers it on the next run.

### Adding a New Benchmark

Subclass `Benchmark` and implement `run()`. See `benchmarks/skill_selection.py`
for a complete example.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EvaluationRunner                             │
│  reads config → builds adapters, registries, benchmarks              │
│  runs cross-product: model × skill_config × benchmark × n_runs      │
│  collects BenchmarkResult list → comparison table → JSON + CSV       │
└─────────┬───────────────────┬────────────────────────────────────────┘
          │                   │
   ┌──────▼──────┐     ┌──────▼──────────┐
   │ ModelAdapter │     │  SkillRegistry   │
   │   (ABC)      │     │  auto-discovers  │
   ├──────────────┤     │  skill folders   │
   │ Ollama  ✦    │     └──────┬───────────┘
   │ OpenAI       │            │
   │ HuggingFace  │     ┌──────▼───────────────────────────┐
   │ LlamaCpp     │     │  Skills: calculator, converter,   │
   └──────┬───────┘     │  dictionary, datetime_calc        │
          │             └──────────────────────────────────┘
   ┌──────▼──────┐
   │  Benchmark  │
   ├─────────────┤
   │ SkillSelect │ → "Which tool should I use?"
   │ EndToEnd    │ → "Did I get the right answer?"
   └─────────────┘
                    ↓
   ┌──────────────────────────────┐
   │  analyze.py                   │
   │  Charts, tables, insights     │
   └──────────────────────────────┘
```

---

## Resource Links

- **Ollama**: https://ollama.com — Local model runner
- **Qwen**: https://qwenlm.github.io — Alibaba's open model family
- **Gemma**: https://ai.google.dev/gemma — Google's open model family
- **Nemotron**: https://developer.nvidia.com/nemotron — NVIDIA's efficient model family
- **Tool-use in LLMs**: Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
- **Small LLM Survey**: Zhu et al., "A Survey on Small Language Models" (2024)
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness

---

## License

MIT
