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
    "author": "sLLM_eval_framework",
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
    """Parse a date string in common formats.

    Tries ISO first (cheapest, most reliable for the test suite), then
    common US/EU numeric formats, then natural-language month names.
    Strips wrapping quotes/punctuation that LLMs sometimes leave attached.
    """
    s = s.strip().strip("'\",.;")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y",
                "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}. Use YYYY-MM-DD format.")


# A regex that matches the various date *spans* we accept. We capture the
# entire span so callers can hand it to ``_parse_date`` which already knows
# how to dispatch among formats.
_DATE_SPAN_RE = re.compile(
    r"""(
            \d{4}-\d{1,2}-\d{1,2}                              # ISO 2024-01-15
          | \d{1,2}/\d{1,2}/\d{2,4}                            # 01/15/2024
          | (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec
              |January|February|March|April|June|July|August
              |September|October|November|December)\.?         # month name
            \s+\d{1,2}(?:,)?\s+\d{4}                           # March 11, 2025
          | \d{1,2}\s+
            (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec
              |January|February|March|April|June|July|August
              |September|October|November|December)\.?
            \s+\d{4}                                            # 11 March 2025
        )""",
    re.IGNORECASE | re.VERBOSE,
)


def _find_dates(text: str) -> list[str]:
    """Return all date-like spans in *text*, in order, as raw strings.

    A date is identified by ``_DATE_SPAN_RE``; we trim trailing punctuation
    (commas, periods) that the regex includes only conditionally.
    """
    return [m.group(1).strip().rstrip(",.") for m in _DATE_SPAN_RE.finditer(text)]


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
    """Parse a natural language date query into an operation.

    Strategy
    --------
    The rigid pattern-at-position approach in the first version broke on
    natural prose ("what day of the week *was*…", "days *from*/*to*…",
    "March 11, 2025"). The new approach is: first extract all date spans
    (`_find_dates`), then look at the *keywords* around them to decide the
    operation. This decouples date-format handling from op selection.
    """
    q = query.strip()
    ql = q.lower()
    dates = _find_dates(q)

    # --- Two-date operations (days_between) ---
    # Triggered by a span-linker keyword (between / from-to) OR by two dates
    # separated by "to"/"and" even without a leading verb.
    if len(dates) >= 2:
        # Look for a range-style linker ("between", "from … to/and", "difference")
        has_between = re.search(r"\b(between|from|since)\b", ql) is not None
        has_days_word = re.search(r"\b(days?|weeks?|time)\b", ql) is not None
        # Bare "date to date" / "date and date" also counts as a range query
        # since no other interpretation makes sense for two dates.
        if has_between or has_days_word or re.search(r"\b(to|and|-)\b", ql):
            return {"op": "days_between", "date1": dates[0], "date2": dates[1]}

    # --- Single-date operations ---
    if not dates:
        raise ValueError(f"Could not parse date query: {query!r}")

    date_str = dates[0]

    # add / subtract N days|weeks (keyword-driven, order-insensitive).
    m = re.search(
        r"(add|plus|\+|subtract|minus|-)\s+(\d+)\s*(days?|weeks?)",
        ql,
    )
    if m:
        verb = m.group(1)
        n = int(m.group(2))
        unit = m.group(3).rstrip("s")
        if unit == "week":
            n *= 7
        if verb in ("subtract", "minus", "-"):
            n = -n
        return {"op": "add_days", "date": date_str, "days": n}

    # "N days/weeks after/before DATE"
    m = re.search(r"(\d+)\s*(days?|weeks?)\s+(after|before)", ql)
    if m:
        n = int(m.group(1))
        unit = m.group(2).rstrip("s")
        direction = m.group(3)
        if unit == "week":
            n *= 7
        if direction == "before":
            n = -n
        return {"op": "add_days", "date": date_str, "days": n}

    # Day-of-week intent: explicit "day of week" / "what day" OR a bare date
    # with no other verbs (the most natural use of a lone date string).
    dow_intent = re.search(r"\bday\b.*\bweek\b|\bwhat\s+day\b|\bwhich\s+day\b", ql)
    has_action_verb = re.search(r"\b(add|subtract|plus|minus|days?|weeks?)\b", ql)
    if dow_intent or (len(dates) == 1 and not has_action_verb):
        return {"op": "day_of_week", "date": date_str}

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
        from sLLM_eval_framework.skills.registry import SkillOutput
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
