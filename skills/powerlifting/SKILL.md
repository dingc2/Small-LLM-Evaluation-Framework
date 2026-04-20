# Powerlifting Skill

## Name
`powerlifting`

## Description
Computes the IPF Dots coefficient for a powerlifting total.  Given the
lifter's sex (male/female), bodyweight in kilograms, and total in
kilograms, returns the Dots score rounded to 2 decimals.

Formula (IPF Dots, 2019 specification):

```
Dots = 500 * total / (A*bw^4 + B*bw^3 + C*bw^2 + D*bw + E)
```

with sex-specific coefficients A–E.

## Trigger Patterns
- Keywords: `dots`, `IPF`, `powerlifting`, `wilks`
- Phrases: `bodyweight`, `total NNN kg`

## Input
Natural language:
```json
{"query": "Calculate Dots for male, bodyweight 83.2kg, total 620kg"}
```

Structured parameters:
```json
{
  "query": "",
  "parameters": {"sex": "M", "bodyweight_kg": 83.2, "total_kg": 620}
}
```

Sex can be "M"/"F" or "male"/"female" (case-insensitive).

## Output
```json
{
  "result": 417.99,
  "success": true,
  "metadata": {
    "sex": "M",
    "bodyweight_kg": 83.2,
    "total_kg": 620.0,
    "formula": "IPF Dots (2019)",
    "formatted": "M 83.2kg 620kg -> Dots 417.99"
  }
}
```

## Error handling
- Negative or zero bodyweight → error
- Negative total → error
- Unrecognised sex → error
- Parser can't extract sex or numbers → error

## Provenance
IPF Dots coefficients sourced from the SkillsBench
`powerlifting-coef-calc` task (benchflow-ai/skillsbench) for cross-check.
This skill re-implements the formula in the sLLM_eval_framework contract; the
upstream task also tests spreadsheet-style file writing, which is out of
scope here (prompt-based only, no file I/O).
