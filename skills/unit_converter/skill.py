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
        "Converts values between measurement units. "
        "Supports length (m, km, mi, ft, in, cm, mm, yd), "
        "weight (kg, g, lb, oz, mg, ton), "
        "temperature (C, F, K), "
        "volume (L, mL, gal, cup, fl_oz, tbsp, tsp), "
        "area (m2, km2, ft2, acre, hectare), "
        "speed (m/s, km/h, mph, knot), "
        "time (s, min, h, day, week, year), "
        "and data (B, KB, MB, GB, TB)."
    ),
    "trigger_patterns": [
        r"\bconvert\b",
        r"\bconversion\b",
        r"\bhow\s+many\b.*\bin\b",
        r"\d+\s*(km|miles?|meters?|feet|inches|pounds?|kg|celsius|fahrenheit|gallons?|liters?)\b",
        r"\bto\s+(km|miles?|meters?|feet|inches|pounds?|kg|celsius|fahrenheit|gallons?|liters?)\b",
    ],
    "version": "1.0.0",
    "author": "eval_framework",
    "examples": [
        {"input": "convert 5 km to miles", "expected": 3.10686},
        {"input": "convert 100 F to C", "expected": 37.7778},
        {"input": "convert 1 kg to lb", "expected": 2.20462},
        {"input": "convert 1 gal to L", "expected": 3.78541},
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
        return round(c, 4)
    elif t == "F":
        return round(c * 9 / 5 + 32, 4)
    elif t == "K":
        return round(c + 273.15, 4)
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
