# Model Cards & Ethical Considerations

## Models Under Evaluation

### Qwen 3.5 2B

| Field | Detail |
|---|---|
| **Developer** | Alibaba Cloud (Qwen Team) |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~2 billion |
| **License** | Apache 2.0 |
| **Training Data** | Multilingual web data, code, academic papers, mathematics |
| **Quantisation** | Default Ollama quant — ~2.7 GB on disk |
| **Intended Use** | Lightweight text generation, instruction following, multilingual tasks |
| **Limitations** | Limited reasoning depth at this scale; may struggle with complex multi-step tasks |

### Qwen 3.5 4B

| Field | Detail |
|---|---|
| **Developer** | Alibaba Cloud (Qwen Team) |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~4 billion |
| **License** | Apache 2.0 |
| **Training Data** | Multilingual web data, code, academic papers, mathematics |
| **Quantisation** | Default Ollama quant — ~3.4 GB on disk |
| **Intended Use** | Text generation, code, tool use, reasoning |
| **Limitations** | May produce culturally-biased outputs; primary training focus on Chinese & English |

### Qwen 3.5 9B

| Field | Detail |
|---|---|
| **Developer** | Alibaba Cloud (Qwen Team) |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~9 billion |
| **License** | Apache 2.0 |
| **Training Data** | Multilingual web data, code, academic papers, mathematics |
| **Quantisation** | Default Ollama quant — ~6.6 GB on disk |
| **Intended Use** | Text generation, code, tool use, complex reasoning, multilingual tasks |
| **Limitations** | May produce culturally-biased outputs |

### Gemma 4 E2B

| Field | Detail |
|---|---|
| **Developer** | Google DeepMind |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~2 billion (efficient variant) |
| **License** | Gemma Terms of Use (permissive, research & commercial use) |
| **Training Data** | Web documents, code, mathematics (filtered & curated by Google) |
| **Quantisation** | Default Ollama quant — ~7.2 GB on disk |
| **Intended Use** | Text generation, instruction following, reasoning |
| **Limitations** | May produce inaccurate or biased outputs; not for safety-critical applications |

### Gemma 4 E4B

| Field | Detail |
|---|---|
| **Developer** | Google DeepMind |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~4 billion (efficient variant) |
| **License** | Gemma Terms of Use |
| **Training Data** | Web documents, code, mathematics (filtered & curated by Google) |
| **Quantisation** | Default Ollama quant — ~9.6 GB on disk |
| **Intended Use** | Text generation, instruction following, reasoning, tool use |
| **Limitations** | May produce inaccurate or biased outputs |

### Nemotron-3-Nano 4B

| Field | Detail |
|---|---|
| **Developer** | NVIDIA |
| **Architecture** | Transformer (decoder-only), optimised for edge/embedded deployment |
| **Parameters** | ~4 billion |
| **License** | NVIDIA Open Model License |
| **Training Data** | Curated multilingual corpus with emphasis on reasoning and instruction following |
| **Quantisation** | Default Ollama quant — ~2.8 GB on disk |
| **Intended Use** | Efficient on-device inference, tool use, code generation |
| **Limitations** | Optimised for efficiency over raw capability; may lag behind same-size general models on some tasks |

### GPT-OSS 20B

| Field | Detail |
|---|---|
| **Developer** | Open-source community |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~20 billion |
| **License** | Open-source (varies by specific release) |
| **Training Data** | Large-scale web corpus |
| **Quantisation** | Default Ollama quant — ~13 GB on disk |
| **Intended Use** | General text generation, reasoning, tool use; largest model in our evaluation |
| **Limitations** | Requires significant memory; quantisation may affect quality vs full precision |

### Ministral-3 3B

| Field | Detail |
|---|---|
| **Developer** | Mistral AI |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~3 billion |
| **License** | MRL (Mistral Research License) |
| **Training Data** | Multilingual web data, code, mathematics |
| **Quantisation** | Default Ollama quant — ~2.0 GB on disk |
| **Intended Use** | Lightweight instruction following, tool use, code generation |
| **Limitations** | Smaller context window than larger Ministral variants; may struggle with complex multi-step reasoning |

### Ministral-3 8B

| Field | Detail |
|---|---|
| **Developer** | Mistral AI |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~8 billion |
| **License** | MRL (Mistral Research License) |
| **Training Data** | Multilingual web data, code, mathematics |
| **Quantisation** | Default Ollama quant — ~4.9 GB on disk |
| **Intended Use** | Instruction following, tool use, code generation, reasoning |
| **Limitations** | May produce inaccurate outputs on highly domain-specific tasks |

### Ministral-3 14B

| Field | Detail |
|---|---|
| **Developer** | Mistral AI |
| **Architecture** | Transformer (decoder-only) |
| **Parameters** | ~14 billion |
| **License** | MRL (Mistral Research License) |
| **Training Data** | Multilingual web data, code, mathematics |
| **Quantisation** | Default Ollama quant — ~8.9 GB on disk |
| **Intended Use** | Complex instruction following, tool use, reasoning, code generation |
| **Limitations** | Requires substantial memory; quantisation may affect quality vs full precision |

---

## Frontier Baseline Models (paid API)

These three models were evaluated as **no-tools baselines** in the intersection experiment (initial run `20260418T030110Z` with mini+nano; re-run `20260418T031413Z` added `gpt-5.4-mini-2026-03-17`). They run on OpenAI's cloud infrastructure; prompts are sent off-device to a paid API endpoint.

### GPT-4.1 Mini

| Field | Detail |
|---|---|
| **Developer** | OpenAI |
| **Architecture** | Transformer (decoder-only); proprietary |
| **Parameters** | Proprietary / undisclosed |
| **License** | OpenAI API Terms of Service (commercial, not open-source) |
| **Training Data** | Proprietary; large-scale multilingual web corpus, code, RLHF fine-tuning |
| **Context Window** | Long context (≥128k tokens typical; verify current limits at platform.openai.com) |
| **Quantisation** | N/A — cloud-hosted; quantisation scheme not disclosed by OpenAI |
| **Intended Use in this framework** | No-tools frontier baseline for the intersection experiment; tests how well a paid frontier model scores on the 24-case intersection subset *without* any tool access |
| **Intersection-24 accuracy** | 0.764 overall; Strong-17: 0.725; Moderate-7: 0.857 |
| **Pricing (public estimate — verify before use)** | ~$0.40 / $1.60 per million input/output tokens |
| **Limitations** | Cloud inference sends prompts off-device to OpenAI servers — not suitable for privacy-sensitive data; paid tier means results are not freely reproducible for readers without API credit; evaluated without tools, so scores are a lower bound — with tool access this model would likely score higher |

### GPT-4.1 Nano

| Field | Detail |
|---|---|
| **Developer** | OpenAI |
| **Architecture** | Transformer (decoder-only); proprietary |
| **Parameters** | Proprietary / undisclosed |
| **License** | OpenAI API Terms of Service (commercial, not open-source) |
| **Training Data** | Proprietary; large-scale multilingual web corpus, code, RLHF fine-tuning |
| **Context Window** | Long context (≥128k tokens typical; verify current limits at platform.openai.com) |
| **Quantisation** | N/A — cloud-hosted; quantisation scheme not disclosed by OpenAI |
| **Intended Use in this framework** | No-tools frontier baseline for the intersection experiment; smallest/cheapest OpenAI API tier tested |
| **Intersection-24 accuracy** | 0.667 overall; Strong-17: 0.588; Moderate-7: 0.857 |
| **Pricing (public estimate — verify before use)** | ~$0.10 / $0.40 per million input/output tokens |
| **Limitations** | Cloud inference sends prompts off-device to OpenAI servers — not suitable for privacy-sensitive data; paid tier means results are not freely reproducible for readers without API credit; evaluated without tools, so scores are a lower bound — with tool access this model would likely score higher; nano tier is optimised for cost/speed and may lag behind larger frontier models on precision arithmetic tasks |

### GPT-5.4 Mini (snapshot `2026-03-17`)

| Field | Detail |
|---|---|
| **Developer** | OpenAI |
| **Architecture** | Transformer (decoder-only); proprietary GPT-5 family |
| **Parameters** | Proprietary / undisclosed |
| **License** | OpenAI API Terms of Service (commercial, not open-source) |
| **Training Data** | Proprietary; post–GPT-4.1 generation, knowledge cutoff per OpenAI disclosure |
| **Context Window** | Long context (verify current limits at platform.openai.com) |
| **Quantisation** | N/A — cloud-hosted; quantisation scheme not disclosed by OpenAI |
| **Intended Use in this framework** | Newer-generation frontier no-tools baseline — added after the initial mini/nano run to check whether a later-generation small-tier frontier model closes the gap with small-local-with-tools. It does not. |
| **Intersection-24 accuracy** | 0.778 overall (highest of the 3 frontier tiers tested); Strong-17: 0.745; Moderate-7: 0.857 |
| **API quirk** | Rejects the legacy `max_tokens` parameter; requires `max_completion_tokens` instead. The `OpenAIAdapter` auto-translates for model IDs starting with `gpt-5`, `o1`, `o3`, or `o4`. |
| **Pricing** | Verify at platform.openai.com; ~$0.30 OpenAI spend for the full 3-run × 33-case sweep of this model alone |
| **Limitations** | Same cloud-inference + reproducibility caveats as gpt-4.1-mini/nano; the +1.4 pp lift over gpt-4.1-mini on Intersection-24 is small relative to the run-to-run noise (≈3 pp SD), so "GPT-5.4 mini > GPT-4.1 mini" should be read as "comparable," not "decisively better." Still well behind small+tool setups. |

---

## Evaluation Data Card

### Dataset Description

The evaluation dataset consists of **66 hand-crafted test cases** across two benchmarks (with 8 clinical lab and 5 powerlifting cases ported from [SkillsBench](https://github.com/benchflow-ai/skillsbench)):

| Benchmark | Test Cases | Skill Categories |
|---|---|---|
| Skill Selection | 33 | Calculator (5), Unit Converter (5+3 clinical), Dictionary (5), Date/Time (5), Powerlifting (5), None (5) |
| End-to-End | 33 | Calculator (6), Unit Converter (5+8 clinical), Dictionary (3), Date/Time (4), Powerlifting (5), No Tool (2) |

### Data Characteristics

- **Language**: English only
- **Difficulty**: Elementary to intermediate (clinical lab values require basic unit conversion knowledge; powerlifting requires knowing bodyweight, sex, and total)
- **Bias Considerations**: Test cases use culturally neutral queries; no personally identifiable information
- **Ground Truth**: All expected answers are deterministic and verifiable (computed values, dictionary entries, date calculations, IPF Dots formula)
- **SkillsBench provenance**: 8 clinical lab cases (`lab-unit-harmonization`) and 5 powerlifting cases (`powerlifting-coef-calc`) are ported from benchflow-ai/skillsbench (MIT). Only the formula/calculation subset was ported; Excel/file I/O and OCR portions are out of scope.

### Limitations

- Small dataset size (66 cases) — results should be interpreted as indicative, not definitive
- English-only queries do not test multilingual capabilities
- Skills are simple (no multi-hop reasoning or complex API chains)
- Quantised models may perform differently from full-precision versions
- Both `all_skills` and `no_skills` conditions run the same 33 end-to-end cases; in the `no_skills` condition, tool definitions are not injected so the model answers from raw reasoning — the comparison table includes `n_cases` to confirm parity
- Keyword-overlap scoring for dictionary definitions (≥60% overlap threshold) is a rough heuristic; more sophisticated semantic similarity scoring (e.g. BERTScore) would be more rigorous
- Temperature=0.0 does not guarantee fully deterministic outputs under quantised inference; 3 runs provide some variance estimate but more would be ideal

---

## Ethical Considerations

### Intended Use

This evaluation framework is designed for **academic research** comparing the tool-use
capabilities of small open-source language models. It is intended to:

1. Inform researchers about the tradeoffs between model size and tool-augmented performance
2. Help practitioners select appropriate models for resource-constrained deployments
3. Contribute to understanding how tool access affects LLM capability

### Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Overgeneralisation** | We test 5 skill types; results may not generalise to all tool categories. We clearly state this limitation. |
| **Quantisation artefacts** | All models use default Ollama quantisation for fair comparison, but full-precision results may differ. |
| **Benchmark gaming** | Test cases are simple and deterministic; we do not claim these results predict real-world deployment performance. |
| **Environmental impact** | Running all models locally avoids API costs but consumes electricity. The full sweep (~45–90 min on M-series Mac) has modest energy use. |
| **Bias propagation** | Models may encode biases from training data. Our benchmark focuses on factual/computational tasks that minimise subjective bias exposure. |

### Responsible AI Considerations

- **Transparency**: All code, configs, and test cases are open source
- **Reproducibility**: Fixed seeds (temperature=0.0), deterministic skills, version-pinned models via Ollama tags
- **Fairness**: Same evaluation conditions for all models; no model-specific prompt engineering
- **Privacy**: No user data is collected or processed; all test inputs are synthetic
- **Sustainability**: Local inference avoids carbon costs of cloud API roundtrips

### Broader Impact

This work supports the goal of making capable AI more accessible by demonstrating
that small models, when properly augmented with tools, can achieve practical performance
levels. This has positive implications for:

- **Accessibility**: Reducing hardware requirements for useful AI applications
- **Privacy**: Enabling fully local inference without sending data to cloud APIs
- **Cost**: Eliminating per-token API costs for tool-augmented applications
- **Autonomy**: Empowering users to run and customise their own AI systems
