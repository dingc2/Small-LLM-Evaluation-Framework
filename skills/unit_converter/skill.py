"""
Unit converter skill — converts between common measurement units.

Supports: length, weight/mass, temperature, volume, area, speed, time, and data storage.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Skill metadata
# ---------------------------------------------------------------------------

SKILL_META = {
    "name": "unit_converter",
    "description": (
        "Use for converting between measurement units: "
        "length (km, miles, feet, inches, cm), weight (kg, pounds, ounces, grams), "
        "temperature (Celsius, Fahrenheit, Kelvin), volume (liters, gallons, cups), "
        "and clinical lab values (creatinine, glucose, cholesterol, calcium, BUN, "
        "vitamin D, hemoglobin, triglycerides — mg/dL ↔ mmol/L, g/dL ↔ g/L, etc.). "
        "Use this whenever the user asks 'how many X in Y' or 'convert X to Y'."
    ),
    "trigger_patterns": [
        r"\bconvert\b",
        r"\bconversion\b",
        r"\bhow\s+many\b.*\bin\b",
        r"\d+\s*(km|miles?|meters?|feet|inches|pounds?|kg|celsius|fahrenheit|gallons?|liters?)\b",
        r"\bto\s+(km|miles?|meters?|feet|inches|pounds?|kg|celsius|fahrenheit|gallons?|liters?)\b",
        r"\b(creatinine|hemoglobin|glucose|cholesterol|calcium|bun|urea|vitamin\s*d|triglycerides)\b",
        r"\b(mg/dL|mmol/L|µmol/L|umol/L|ng/mL|nmol/L|g/dL|g/L)\b",
    ],
    "version": "1.1.0",
    "author": "eval_framework",
    "examples": [
        {"input": "convert 5 km to miles", "expected": 3.10686},
        {"input": "convert 100 F to C", "expected": 37.7778},
        {"input": "convert 1 kg to lb", "expected": 2.20462},
        {"input": "convert 1 gal to L", "expected": 3.78541},
        {"input": "Convert Creatinine 1.5 mg/dL to µmol/L", "expected": 132.6},
        {"input": "Convert Glucose 126 mg/dL to mmol/L", "expected": 7.0},
    ],
}

# ---------------------------------------------------------------------------
# Conversion tables
# ---------------------------------------------------------------------------

# All conversions go through a canonical base unit per category.
# Multiplier means: value_in_base = value * multiplier

_LENGTH_BASE = "m"
_LENGTH: dict[str, float] = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
}

_WEIGHT_BASE = "kg"
_WEIGHT: dict[str, float] = {
    "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gram": 0.001, "grams": 0.001,
    "mg": 0.000001, "milligram": 0.000001, "milligrams": 0.000001,
    "lb": 0.453592, "pound": 0.453592, "pounds": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
    "ton": 907.185, "tons": 907.185,
    "tonne": 1000.0, "tonnes": 1000.0, "metric_ton": 1000.0,
}

_VOLUME_BASE = "L"
_VOLUME: dict[str, float] = {
    "l": 1.0, "liter": 1.0, "liters": 1.0, "litre": 1.0, "litres": 1.0,
    "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
    "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
    "cup": 0.236588, "cups": 0.236588,
    "fl_oz": 0.0295735, "fluid_ounce": 0.0295735,
    "tbsp": 0.0147868, "tablespoon": 0.0147868,
    "tsp": 0.00492892, "teaspoon": 0.00492892,
}

_AREA_BASE = "m2"
_AREA: dict[str, float] = {
    "m2": 1.0, "sq_m": 1.0, "square_meter": 1.0,
    "km2": 1e6, "sq_km": 1e6, "square_kilometer": 1e6,
    "ft2": 0.092903, "sq_ft": 0.092903, "square_foot": 0.092903,
    "acre": 4046.86, "acres": 4046.86,
    "hectare": 10000.0, "hectares": 10000.0, "ha": 10000.0,
}

_SPEED_BASE = "m/s"
_SPEED: dict[str, float] = {
    "m/s": 1.0, "mps": 1.0,
    "km/h": 1 / 3.6, "kph": 1 / 3.6, "kmh": 1 / 3.6,
    "mph": 0.44704, "mi/h": 0.44704,
    "knot": 0.514444, "knots": 0.514444,
}

_TIME_BASE = "s"
_TIME: dict[str, float] = {
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "day": 86400.0, "days": 86400.0,
    "week": 604800.0, "weeks": 604800.0,
    "year": 31557600.0, "years": 31557600.0,  # Julian year
}

_DATA_BASE = "B"
_DATA: dict[str, float] = {
    "b": 1.0, "byte": 1.0, "bytes": 1.0,
    "kb": 1024.0, "kilobyte": 1024.0,
    "mb": 1024.0 ** 2, "megabyte": 1024.0 ** 2,
    "gb": 1024.0 ** 3, "gigabyte": 1024.0 ** 3,
    "tb": 1024.0 ** 4, "terabyte": 1024.0 ** 4,
}

_CATEGORIES: list[tuple[str, dict[str, float]]] = [
    ("length", _LENGTH),
    ("weight", _WEIGHT),
    ("volume", _VOLUME),
    ("area", _AREA),
    ("speed", _SPEED),
    ("time", _TIME),
    ("data", _DATA),
]

# ---------------------------------------------------------------------------
# Clinical lab unit conversions
# ---------------------------------------------------------------------------
# Conversion factors are analyte-dependent (mg/dL → mmol/L differs by
# molecular weight). Ported from the SkillsBench ``lab-unit-harmonization``
# task; re-expressed as a flat lookup table.
# Key: (analyte_canonical, from_unit_normalized, to_unit_normalized)
# Value: multiplier such that ``result_in_to_unit = value * multiplier``

_CLINICAL: dict[tuple[str, str, str], float] = {
    # Creatinine — MW ≈ 113.12
    ("creatinine",    "mg/dl", "umol/l"): 88.4,
    ("creatinine",    "umol/l", "mg/dl"): 1 / 88.4,
    # Hemoglobin — simple dilution factor
    ("hemoglobin",    "g/dl",  "g/l"):    10.0,
    ("hemoglobin",    "g/l",   "g/dl"):   0.1,
    # Glucose — MW ≈ 180.156
    ("glucose",       "mg/dl", "mmol/l"): 1 / 18.0156,
    ("glucose",       "mmol/l", "mg/dl"): 18.0156,
    # Cholesterol — MW ≈ 386.65
    ("cholesterol",   "mg/dl", "mmol/l"): 1 / 38.67,
    ("cholesterol",   "mmol/l", "mg/dl"): 38.67,
    # Calcium — MW ≈ 40.08
    ("calcium",       "mg/dl", "mmol/l"): 0.2495,
    ("calcium",       "mmol/l", "mg/dl"): 1 / 0.2495,
    # Blood Urea Nitrogen — MW ≈ 28.02 for N2
    ("bun",           "mg/dl", "mmol/l"): 1 / 2.8,
    ("bun",           "mmol/l", "mg/dl"): 2.8,
    # Urea — same factor as BUN
    ("urea",          "mg/dl", "mmol/l"): 1 / 2.8,
    ("urea",          "mmol/l", "mg/dl"): 2.8,
    # 25-OH Vitamin D — MW ≈ 400.64
    ("vitamin_d",     "ng/ml", "nmol/l"): 2.496,
    ("vitamin_d",     "nmol/l", "ng/ml"): 1 / 2.496,
    # Triglycerides — MW ≈ 885.4
    ("triglycerides", "mg/dl", "mmol/l"): 1 / 88.57,
    ("triglycerides", "mmol/l", "mg/dl"): 88.57,
}

# Canonical analyte → list of lowercase trigger keywords.
# Order matters: more specific keywords ("25-oh" for Vitamin D) before generic ones.
_ANALYTE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("vitamin_d",     ["25-oh", "25oh", "25-hydroxy", "vitamin d", "vit d", "vitd"]),
    ("cholesterol",   ["cholesterol", "total chol", "chol"]),
    ("triglycerides", ["triglycerides", "triglyceride", "trig"]),
    ("creatinine",    ["creatinine", "creat"]),
    ("hemoglobin",    ["hemoglobin", "haemoglobin", "hgb", "hb"]),
    ("glucose",       ["glucose", "blood sugar"]),
    ("calcium",       ["calcium"]),
    ("bun",           ["bun", "blood urea nitrogen"]),
    ("urea",          ["urea"]),
]


def _identify_analyte(text: str) -> str | None:
    """Return the canonical analyte name if any keyword matches, else None."""
    t = text.lower()
    for canonical, keywords in _ANALYTE_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return canonical
    return None


def _normalize_clinical_unit(u: str) -> str:
    """Normalize a clinical unit for _CLINICAL lookup (lowercase, µ→u)."""
    return u.lower().strip().rstrip(".,;").replace("µ", "u")


def _parse_clinical_query(query: str) -> tuple[str, float, str, str] | None:
    """Extract (analyte, value, from_unit, to_unit) from a clinical query.

    Returns None if no analyte keyword is found or the value/unit structure
    can't be parsed. Handles analyte before or after the value.
    """
    analyte = _identify_analyte(query)
    if analyte is None:
        return None

    # Find a number followed by a unit-like token (letters and optional /).
    m_val = re.search(
        r"(-?\d+(?:\.\d+)?)\s*([A-Za-zµ]+(?:/[A-Za-zµ]+)?)",
        query,
    )
    if m_val is None:
        return None
    value = float(m_val.group(1))
    from_unit = m_val.group(2)

    # Look for a target unit introduced by "to", "in", "into", or "=" after
    # the value.  Doing this on the tail keeps the "in" keyword from matching
    # letters inside the from_unit itself.
    tail = query[m_val.end():]
    m_to = re.search(
        r"\b(?:to|in|into|=)\s*([A-Za-zµ]+(?:/[A-Za-zµ]+)?)",
        tail,
        re.IGNORECASE,
    )
    if m_to is None:
        return None
    to_unit = m_to.group(1)

    return analyte, value, from_unit, to_unit


def _convert_clinical(analyte: str, value: float, from_unit: str, to_unit: str) -> float:
    key = (analyte, _normalize_clinical_unit(from_unit), _normalize_clinical_unit(to_unit))
    factor = _CLINICAL.get(key)
    if factor is None:
        raise ValueError(
            f"No clinical conversion available for {analyte!r}: "
            f"{from_unit!r} -> {to_unit!r}"
        )
    return round(value * factor, 4)


def _find_category(unit: str) -> tuple[str, dict[str, float]] | None:
    unit_lower = unit.lower().strip()
    for name, table in _CATEGORIES:
        if unit_lower in table:
            return name, table
    return None


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    f = from_unit.upper().rstrip("°")
    t = to_unit.upper().rstrip("°")

    # Normalise
    for alias, canonical in [("CELSIUS", "C"), ("FAHRENHEIT", "F"), ("KELVIN", "K")]:
        if f == alias:
            f = canonical
        if t == alias:
            t = canonical

    # To Celsius first
    if f == "C":
        c = value
    elif f == "F":
        c = (value - 32) * 5 / 9
    elif f == "K":
        c = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit!r}")

    # From Celsius to target
    if t == "C":
        return round(c, 6)
    elif t == "F":
        return round(c * 9 / 5 + 32, 6)
    elif t == "K":
        return round(c + 273.15, 6)
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit!r}")


def _is_temperature(unit: str) -> bool:
    u = unit.upper().replace("°", "").strip()
    return u in ("C", "F", "K", "CELSIUS", "FAHRENHEIT", "KELVIN")


def _convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between two units."""
    if _is_temperature(from_unit) or _is_temperature(to_unit):
        return _convert_temperature(value, from_unit, to_unit)

    from_cat = _find_category(from_unit)
    to_cat = _find_category(to_unit)

    if from_cat is None:
        raise ValueError(f"Unknown unit: {from_unit!r}")
    if to_cat is None:
        raise ValueError(f"Unknown unit: {to_unit!r}")
    if from_cat[0] != to_cat[0]:
        raise ValueError(
            f"Cannot convert between {from_cat[0]} ({from_unit}) "
            f"and {to_cat[0]} ({to_unit})"
        )

    table = from_cat[1]
    base_value = value * table[from_unit.lower().strip()]
    result = base_value / table[to_unit.lower().strip()]
    return round(result, 6)


def _parse_query(query: str) -> tuple[float, str, str]:
    """Parse natural language conversion query."""
    query = query.strip()

    # Pattern: "convert X unit to unit"
    m = re.match(
        r"(?:convert\s+)?(-?\d+(?:\.\d+)?)\s+(\S+)\s+(?:to|in|into)\s+(\S+)",
        query,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1)), m.group(2), m.group(3)

    # Pattern: "X unit = ? unit" or "X unit in unit"
    m = re.match(
        r"(-?\d+(?:\.\d+)?)\s+(\S+)\s*(?:=|in|to)\s*\??\s*(\S+)",
        query,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1)), m.group(2), m.group(3)

    # Pattern: "how many unit in X unit"
    m = re.match(
        r"how\s+many\s+(\S+)\s+(?:in|are\s+in)\s+(-?\d+(?:\.\d+)?)\s+(\S+)",
        query,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(2)), m.group(3), m.group(1)

    raise ValueError(f"Could not parse conversion query: {query!r}")


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(input: Any) -> Any:
    """Execute a unit conversion."""
    if hasattr(input, "query"):
        query: str = input.query
        params: dict = getattr(input, "parameters", {})
    else:
        query = input.get("query", "")
        params = input.get("parameters", {})

    try:
        from eval_framework.skills.registry import SkillOutput
    except ImportError:
        SkillOutput = None

    try:
        # Clinical path: an explicit analyte in structured params, or an analyte
        # keyword detected in the query.  Checked first because clinical units
        # (mg/dL, mmol/L, ...) aren't in the general categorical tables.
        analyte: str | None = params.get("analyte")
        if analyte is not None and "value" in params and "from_unit" in params and "to_unit" in params:
            value = float(params["value"])
            from_unit = params["from_unit"]
            to_unit = params["to_unit"]
            result = _convert_clinical(analyte, value, from_unit, to_unit)
            meta = {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "analyte": analyte,
                "category": "clinical",
                "formatted": f"{value} {from_unit} {analyte} = {result} {to_unit}",
            }
            if SkillOutput:
                return SkillOutput(result=result, success=True, metadata=meta)
            return {"result": result, "success": True, "error": None, "metadata": meta}

        clinical_parse = _parse_clinical_query(query) if query else None
        if clinical_parse is not None:
            analyte, value, from_unit, to_unit = clinical_parse
            result = _convert_clinical(analyte, value, from_unit, to_unit)
            meta = {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "analyte": analyte,
                "category": "clinical",
                "formatted": f"{value} {from_unit} {analyte} = {result} {to_unit}",
            }
            if SkillOutput:
                return SkillOutput(result=result, success=True, metadata=meta)
            return {"result": result, "success": True, "error": None, "metadata": meta}

        # If structured params are provided, use them directly
        if "value" in params and "from_unit" in params and "to_unit" in params:
            value = float(params["value"])
            from_unit = params["from_unit"]
            to_unit = params["to_unit"]
        else:
            value, from_unit, to_unit = _parse_query(query)

        result = _convert(value, from_unit, to_unit)
        meta = {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "formatted": f"{value} {from_unit} = {result} {to_unit}",
        }

        if SkillOutput:
            return SkillOutput(result=result, success=True, metadata=meta)
        return {"result": result, "success": True, "error": None, "metadata": meta}

    except Exception as exc:
        if SkillOutput:
            return SkillOutput.failure(str(exc))
        return {"result": None, "success": False, "error": str(exc), "metadata": {}}
