"""
Powerlifting skill — IPF Dots coefficient calculator.

Given a lifter's sex, bodyweight (kg), and total (kg), returns the IPF Dots
score.  Formula and coefficients are the current IPF Dots specification
(adopted 2019, published on openpowerlifting.org and in the IPF Technical
Rules).

Ported from the SkillsBench ``powerlifting-coef-calc`` task as reference
only — re-implemented to conform to the sLLM_eval_framework skill contract
(``SKILL_META`` + ``execute(SkillInput) -> SkillOutput``).
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Skill metadata
# ---------------------------------------------------------------------------

SKILL_META = {
    "name": "powerlifting",
    "description": (
        "Compute the IPF Dots coefficient for a powerlifting total. "
        "Given sex (male/female), bodyweight in kg, and total in kg, returns "
        "the Dots score. Use whenever the prompt mentions Dots, IPF points, "
        "Wilks, or powerlifting scoring/ranking."
    ),
    "trigger_patterns": [
        r"\bdots\b",
        r"\bIPF\b",
        r"\bpowerlifting\b",
        r"\bwilks\b",
        r"\btotal\s+\d+\s*kg\b",
        r"\bbodyweight\b",
    ],
    "version": "1.0.0",
    "author": "sLLM_eval_framework",
    "examples": [
        {"input": "Dots for male 83.2kg, 620kg total", "expected": 417.99},
        {"input": "Dots for female 57.3kg, 390kg total", "expected": 445.29},
    ],
}

# ---------------------------------------------------------------------------
# IPF Dots coefficients (2019 specification)
# ---------------------------------------------------------------------------
# Denominator polynomial:  A*bw^4 + B*bw^3 + C*bw^2 + D*bw + E
# Dots score:              500 * total / denom

_DOTS_COEFFS: dict[str, tuple[float, float, float, float, float]] = {
    "M": (-1.093e-6, 7.391293e-4, -0.1918759221, 24.0900756, -307.75076),
    "F": (-1.0706e-6, 5.158568e-4, -0.1126655495, 13.6175032, -57.96288),
}


def _dots(sex: str, bw: float, total: float) -> float:
    """Compute the IPF Dots score. Raises ``ValueError`` on invalid input."""
    if bw <= 0:
        raise ValueError(f"bodyweight must be positive (got {bw})")
    if total < 0:
        raise ValueError(f"total must be non-negative (got {total})")

    s = sex.strip().upper()[:1]
    coeffs = _DOTS_COEFFS.get(s)
    if coeffs is None:
        raise ValueError(f"sex must be 'M' or 'F' (got {sex!r})")

    a, b, c, d, e = coeffs
    denom = a * bw**4 + b * bw**3 + c * bw**2 + d * bw + e
    if denom == 0:
        raise ValueError(f"Dots denominator is zero at bodyweight={bw}")
    return 500.0 * total / denom


# ---------------------------------------------------------------------------
# Natural-language query parser
# ---------------------------------------------------------------------------


def _parse_query(query: str) -> tuple[str, float, float]:
    """Extract (sex, bodyweight_kg, total_kg) from a natural-language query."""
    q = query.lower()

    # Sex detection — check "female" before "male" (substring).
    if re.search(r"\bfemale\b|\bwomen\b|\bwoman\b|\(f\)|,\s*f\b", q):
        sex = "F"
    elif re.search(r"\bmale\b|\bmen\b|\bman\b|\(m\)|,\s*m\b", q):
        sex = "M"
    else:
        raise ValueError(
            "Could not determine sex (expected male/female) from query"
        )

    # Look for explicit bodyweight / total keywords in either order.
    m_bw = re.search(
        r"bodyweight\s*(?:is\s*|of\s*|:)?\s*(-?\d+(?:\.\d+)?)", q
    ) or re.search(r"(-?\d+(?:\.\d+)?)\s*kg\s*bodyweight", q)

    m_tot = re.search(
        r"total\s*(?:is\s*|of\s*|:)?\s*(-?\d+(?:\.\d+)?)", q
    ) or re.search(r"(-?\d+(?:\.\d+)?)\s*kg\s*total", q)

    bw = float(m_bw.group(1)) if m_bw else None
    total = float(m_tot.group(1)) if m_tot else None

    # Fallback: when at least one value wasn't keyword-labelled, fall back to
    # the kg-annotated numbers in the query.  For realistic powerlifting data
    # the smaller is always bodyweight (50–200 kg) and the larger is always
    # the total (200–1200 kg).
    if bw is None or total is None:
        nums = [float(n) for n in re.findall(r"(-?\d+(?:\.\d+)?)\s*kg", q)]
        if not nums:
            nums = [float(n) for n in re.findall(r"(-?\d+(?:\.\d+)?)", q)]
        if len(nums) >= 2:
            nums_sorted = sorted(nums)
            if bw is None:
                bw = nums_sorted[0]
            if total is None:
                total = nums_sorted[-1]

    if bw is None or total is None:
        raise ValueError(
            f"Could not extract bodyweight and total from query: {query!r}"
        )

    return sex, bw, total


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(input: Any) -> Any:  # noqa: A002
    """Compute the IPF Dots coefficient."""
    if hasattr(input, "query"):
        query: str = input.query
        params: dict = getattr(input, "parameters", {})
    else:
        query = input.get("query", "")
        params = input.get("parameters", {})

    try:
        from sLLM_eval_framework.skills.registry import SkillOutput  # noqa: PLC0415
    except ImportError:
        SkillOutput = None

    try:
        # Structured params take precedence
        if (
            "sex" in params
            and ("bodyweight_kg" in params or "bodyweight" in params)
            and ("total_kg" in params or "total" in params)
        ):
            sex = str(params["sex"])
            bw = float(params.get("bodyweight_kg", params.get("bodyweight")))
            total = float(params.get("total_kg", params.get("total")))
        else:
            sex, bw, total = _parse_query(query)

        result = round(_dots(sex, bw, total), 2)
        meta = {
            "sex": sex.upper()[:1],
            "bodyweight_kg": bw,
            "total_kg": total,
            "formula": "IPF Dots (2019)",
            "formatted": f"{sex} {bw}kg {total}kg -> Dots {result}",
        }

        if SkillOutput:
            return SkillOutput(result=result, success=True, metadata=meta)
        return {"result": result, "success": True, "error": None, "metadata": meta}

    except Exception as exc:
        if SkillOutput:
            return SkillOutput.failure(str(exc))
        return {"result": None, "success": False, "error": str(exc), "metadata": {}}
