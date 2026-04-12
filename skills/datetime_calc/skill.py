"""
Date/time calculator skill — performs date arithmetic and formatting.

Supports: days between dates, add/subtract durations, day-of-week lookup,
and date formatting.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Skill metadata
# ---------------------------------------------------------------------------

SKILL_META = {
    "name": "datetime_calc",
    "description": (
        "Use for date and time questions: count days between two dates, "
        "add or subtract days from a date, or find what day of the week a date is. "
        "Use this whenever the user asks about dates, durations, or days of the week."
    ),
    "trigger_patterns": [
        r"\bdays?\s+(between|from|until|since|ago)\b",
        r"\bdate\b.*\b(add|subtract|plus|minus|after|before)\b",
        r"\bday\s+of\s+(the\s+)?week\b",
        r"\bwhat\s+day\b",
        r"\d{4}-\d{2}-\d{2}",
        r"\bhow\s+(long|many\s+days)\b",
    ],
    "version": "1.0.0",
    "author": "eval_framework",
    "examples": [
        {"input": "days between 2024-01-01 and 2024-12-31", "expected": 365},
        {"input": "what day of the week is 2024-07-04", "expected": "Thursday"},
        {"input": "add 30 days to 2024-01-15", "expected": "2024-02-14"},
        {"input": "days between 2024-03-01 and 2024-03-15", "expected": 14},
    ],
}

# ---------------------------------------------------------------------------
# Date operations
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> datetime:
    """Parse a date string in common formats."""
    s = s.strip().strip("'\"")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}. Use YYYY-MM-DD format.")


def _days_between(date1: str, date2: str) -> int:
    """Return absolute number of days between two dates."""
    d1 = _parse_date(date1)
    d2 = _parse_date(date2)
    return abs((d2 - d1).days)


def _add_days(date_str: str, days: int) -> str:
    """Add (or subtract) days from a date, return YYYY-MM-DD."""
    d = _parse_date(date_str)
    result = d + timedelta(days=days)
    return result.strftime("%Y-%m-%d")


def _day_of_week(date_str: str) -> str:
    """Return the day of the week for a date."""
    d = _parse_date(date_str)
    return d.strftime("%A")  # Full day name (Monday, Tuesday, etc.)


def _parse_query(query: str) -> dict[str, Any]:
    """Parse a natural language date query into an operation."""
    query = query.strip()

    # "days between DATE and DATE"
    m = re.search(
        r"days?\s+between\s+(\S+)\s+and\s+(\S+)",
        query,
        re.IGNORECASE,
    )
    if m:
        return {"op": "days_between", "date1": m.group(1), "date2": m.group(2)}

    # "add N days to DATE" or "DATE plus N days"
    m = re.search(
        r"add\s+(\d+)\s+(days?|weeks?)\s+to\s+(\S+)",
        query,
        re.IGNORECASE,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        if unit == "week":
            n *= 7
        return {"op": "add_days", "date": m.group(3), "days": n}

    # "subtract N days from DATE"
    m = re.search(
        r"subtract\s+(\d+)\s+(days?|weeks?)\s+from\s+(\S+)",
        query,
        re.IGNORECASE,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        if unit == "week":
            n *= 7
        return {"op": "add_days", "date": m.group(3), "days": -n}

    # "N days after/before DATE"
    m = re.search(
        r"(\d+)\s+(days?|weeks?)\s+(after|before)\s+(\S+)",
        query,
        re.IGNORECASE,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        direction = m.group(3).lower()
        if unit == "week":
            n *= 7
        if direction == "before":
            n = -n
        return {"op": "add_days", "date": m.group(4), "days": n}

    # "what day of the week is DATE" or "day of week for DATE"
    m = re.search(
        r"(?:what\s+)?day\s+(?:of\s+(?:the\s+)?week\s+)?(?:is\s+|for\s+)?(\d{4}-\d{2}-\d{2})",
        query,
        re.IGNORECASE,
    )
    if m:
        return {"op": "day_of_week", "date": m.group(1)}

    # "what day is DATE"
    m = re.search(r"what\s+day\s+is\s+(\S+)", query, re.IGNORECASE)
    if m:
        return {"op": "day_of_week", "date": m.group(1)}

    raise ValueError(f"Could not parse date query: {query!r}")


# ---------------------------------------------------------------------------
# Public execute
# ---------------------------------------------------------------------------


def execute(input: Any) -> Any:
    """Execute a date/time calculation."""
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
        # Use structured params if available
        if "op" in params:
            parsed = params
        else:
            parsed = _parse_query(query)

        op = parsed["op"]

        if op == "days_between":
            result = _days_between(parsed["date1"], parsed["date2"])
            meta = {"date1": parsed["date1"], "date2": parsed["date2"], "formatted": f"{result} days"}
        elif op == "add_days":
            result = _add_days(parsed["date"], parsed["days"])
            meta = {"original_date": parsed["date"], "days_added": parsed["days"]}
        elif op == "day_of_week":
            result = _day_of_week(parsed["date"])
            meta = {"date": parsed["date"]}
        else:
            raise ValueError(f"Unknown operation: {op!r}")

        if SkillOutput:
            return SkillOutput(result=result, success=True, metadata=meta)
        return {"result": result, "success": True, "error": None, "metadata": meta}

    except Exception as exc:
        if SkillOutput:
            return SkillOutput.failure(str(exc))
        return {"result": None, "success": False, "error": str(exc), "metadata": {}}
