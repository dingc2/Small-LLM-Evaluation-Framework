# Plan: Port SkillsBench Tasks — Remaining Work

## Status

Implementation of the v2 plan is **mostly complete**. The code, test cases, and unit tests are in place. This file tracks only what still needs to happen before the PR ships.

### Done

- [x] Verified Dots expected values with the real IPF polynomial (417.99 / 445.29 / 419.72 / 440.96 / 430.86)
- [x] Extended `skills/unit_converter/skill.py` with clinical lab units: 8 analytes (creatinine, hemoglobin, glucose, cholesterol, calcium, BUN, urea, vitamin D, triglycerides) in both directions, via a new `_CLINICAL` table + `_parse_clinical_query` + `_identify_analyte`. Updated `SKILL_META` description and trigger patterns.
- [x] Updated `skills/unit_converter/SKILL.md` with the new clinical category and provenance note.
- [x] Created `skills/powerlifting/` (`__init__.py`, `SKILL.md`, `skill.py`) implementing IPF Dots (2019) with NL parsing + structured-params path.
- [x] Appended 13 cases to `benchmarks/end_to_end.py` `_BUILTIN_CASES` (8 lab + 5 dots). Total e2e cases: 20 → 33.
- [x] Added skill-selection cases to `benchmarks/skill_selection.py`: 3 new clinical cases appended to `_UNIT_CONVERTER_CASES`, new `_POWERLIFTING_CASES` (5), registered in `_build_test_cases`, added a powerlifting example to the few-shot system prompt. Total routing cases: 25 → 33.
- [x] Added ~20 unit tests to `tests/test_registry.py` (10 clinical + 10 powerlifting).
- [x] Created symlink `.claude/worktrees/eval_framework -> hopeful-hawking-3ba36f` so `from eval_framework.skills.registry import ...` resolves for pytest inside the worktree. Existing 43 registry tests pass.

---

## Remaining Work

### 1. Run the full pytest suite

```bash
cd eval_framework   # or the worktree root
pytest tests/ -v
```

Expect all tests to pass — the original 43 registry tests + the new ~20 clinical/powerlifting tests + the existing benchmark and adapter tests.

If any test fails, the most likely culprits are:
- `test_dots_negative_bodyweight_errors` — depends on the error-message string "bodyweight" appearing in the raised message.
- `test_dots_missing_sex_errors` — depends on the "sex" token in the error.
- Clinical parser edge cases if the NL query form changes.

### 2. Update `README.md`

- Skills table: 4 skills → **5 skills** (add `powerlifting`).
- Note that `unit_converter` now covers clinical lab values (mg/dL ↔ mmol/L, etc.).
- Benchmark counts: skill_selection 25 → **33**, end_to_end 20 → **33**.
- Under "Tasks ported from SkillsBench" (new subsection): document the two ports, the reference-only relationship, and the out-of-scope items (Docker, file I/O, OCR).

### 3. Update `model_cards.md`

- Data card: note the new clinical unit and powerlifting tasks, their SkillsBench provenance (`benchflow-ai/skillsbench`, tasks `lab-unit-harmonization` and `powerlifting-coef-calc`), and that only the formula/calculation subset was ported (not the Excel / file-based portions).

### 4. Update `CLAUDE.md`

- "4 tools" → "5 tools".
- Under **Architecture**: mention `powerlifting` in the skill list and note the clinical extension of `unit_converter`.
- **Do not** re-describe things already in the code — keep this as a reference doc, not a duplicate of the skill files.

### 5. Verification (do this LAST, before opening the PR)

1. `pytest tests/ -v` → all green.
2. Smoke test via the runner (requires Ollama running):
   ```bash
   python runner.py --config config_quick.yaml --verbose
   ```
   Expect: 3 Ministral models × 2 skill configs × 2 benchmarks complete without errors. Check the comparison table for nonzero scores on the new `e2e_lab_*` and `e2e_dots_*` cases.
3. (Optional, time-permitting) Full 10-model sweep with `config_ollama.yaml`. Aggregate via `merge_results.py`. Regenerate charts with `analyze.py`. Update hardcoded values in `charts_gen.py` if keeping it.

---

## Known Risks / Notes

- **Small-model JSON compliance**: Gemma4:e2b has negative skill uplift because it can't format tool-call JSON. Adding `powerlifting` (multi-field schema) may worsen this for sub-4B models. This is a finding, not a bug — record it in `README.md`'s critical-analysis section after the re-run.
- **Quantization non-determinism**: `runs=3` + aggregated DB already mitigates. Keep the score ± σ reporting.
- **Worktree symlink**: `.claude/worktrees/eval_framework` is a workspace convenience symlink pointing at `hopeful-hawking-3ba36f/`. It is **not** committed — it only exists to make test imports resolve inside the worktree. Remove before merge.

---

## Reference: Files Touched So Far

| File | Status | Notes |
|---|---|---|
| `skills/unit_converter/skill.py` | Modified | `_CLINICAL`, `_ANALYTE_KEYWORDS`, `_parse_clinical_query`, `_convert_clinical`; routed before the existing path in `execute()`. Version bumped to 1.1.0. |
| `skills/unit_converter/SKILL.md` | Modified | Added clinical category + provenance. |
| `skills/powerlifting/skill.py` | Created | IPF Dots (2019), NL + structured-param paths. |
| `skills/powerlifting/SKILL.md` | Created | Description, examples, provenance. |
| `skills/powerlifting/__init__.py` | Created | Empty. |
| `benchmarks/end_to_end.py` | Modified | +13 cases in `_BUILTIN_CASES`. |
| `benchmarks/skill_selection.py` | Modified | `_POWERLIFTING_CASES` (5), 3 new clinical in `_UNIT_CONVERTER_CASES`, registered in `_build_test_cases`, powerlifting example in system prompt. |
| `tests/test_registry.py` | Modified | +10 clinical + +10 powerlifting tests. |
| `README.md` | **TODO** | — |
| `model_cards.md` | **TODO** | — |
| `CLAUDE.md` | **TODO** | — |
