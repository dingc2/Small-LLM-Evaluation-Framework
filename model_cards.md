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

---

## Evaluation Data Card

### Dataset Description

The evaluation dataset consists of **45 hand-crafted test cases** across two benchmarks:

| Benchmark | Test Cases | Skill Categories |
|---|---|---|
| Skill Selection | 25 | Calculator (5), Unit Converter (5), Dictionary (5), Date/Time (5), None (5) |
| End-to-End | 20 | Calculator (6), Unit Converter (5), Dictionary (3), Date/Time (4), No Tool (2) |

### Data Characteristics

- **Language**: English only
- **Difficulty**: Elementary to intermediate (no domain-specific expertise required)
- **Bias Considerations**: Test cases use culturally neutral queries; no personally identifiable information
- **Ground Truth**: All expected answers are deterministic and verifiable (computed values, dictionary entries, date calculations)

### Limitations

- Small dataset size (45 cases) — results should be interpreted as indicative, not definitive
- English-only queries do not test multilingual capabilities
- Skills are simple (no multi-hop reasoning or complex API chains)
- Quantised models may perform differently from full-precision versions
- Both `all_skills` and `no_skills` conditions run the same 20 end-to-end cases; in the `no_skills` condition, tool definitions are not injected so the model answers from raw reasoning — the comparison table includes `n_cases` to confirm parity
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
| **Overgeneralisation** | We test only 4 skill types; results may not generalise to all tool categories. We clearly state this limitation. |
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
