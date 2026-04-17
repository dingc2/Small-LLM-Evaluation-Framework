# Unit Converter

Converts values between common measurement units.

## Supported Categories

- **Length**: m, km, mi, ft, in, cm, mm, yd
- **Weight/Mass**: kg, g, lb, oz, mg, ton, tonne
- **Temperature**: C, F, K
- **Volume**: L, mL, gal, cup, fl_oz, tbsp, tsp
- **Area**: m2, km2, ft2, acre, hectare
- **Speed**: m/s, km/h, mph, knot
- **Time**: s, min, h, day, week, year
- **Data**: B, KB, MB, GB, TB
- **Clinical lab values** (analyte-dependent, molecular-weight–based factors):
  - Creatinine: mg/dL ↔ µmol/L
  - Hemoglobin: g/dL ↔ g/L
  - Glucose: mg/dL ↔ mmol/L
  - Cholesterol (total): mg/dL ↔ mmol/L
  - Calcium: mg/dL ↔ mmol/L
  - BUN / Urea: mg/dL ↔ mmol/L
  - Vitamin D (25-OH): ng/mL ↔ nmol/L
  - Triglycerides: mg/dL ↔ mmol/L

## Usage

Query: "convert 5 km to miles"
Result: 3.10686

Query: "Convert Serum Creatinine 1.5 mg/dL to µmol/L"
Result: 132.6

## Clinical provenance

Clinical lab conversions are ported from the SkillsBench
`lab-unit-harmonization` task (benchflow-ai/skillsbench). Factors are derived
from each analyte's molecular weight; see `skill.py` `_CLINICAL` for
reference values.
