"""
Unit tests for the skill registry and the built-in calculator skill.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pytest

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval_framework.skills.registry import (
    Skill,
    SkillInput,
    SkillOutput,
    SkillRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _make_registry(skills_dir: Path = SKILLS_DIR) -> SkillRegistry:
    reg = SkillRegistry(skills_dir)
    reg.load()
    return reg


# ---------------------------------------------------------------------------
# Tests: SkillRegistry loading
# ---------------------------------------------------------------------------


def test_registry_loads_calculator():
    reg = _make_registry()
    assert "calculator" in reg
    assert len(reg) >= 1


def test_registry_names_returns_sorted_list():
    reg = _make_registry()
    assert reg.names == sorted(reg.names)


def test_registry_get_returns_skill():
    reg = _make_registry()
    skill = reg.get("calculator")
    assert isinstance(skill, Skill)
    assert skill.name == "calculator"


def test_registry_get_raises_for_unknown():
    reg = _make_registry()
    with pytest.raises(KeyError, match="not registered"):
        reg.get("nonexistent_skill_xyz")


def test_registry_reload_clears_old_skills():
    reg = _make_registry()
    count_first = len(reg)
    reg.load(reload=True)
    assert len(reg) == count_first


def test_registry_repr():
    reg = _make_registry()
    r = repr(reg)
    assert "SkillRegistry" in r
    assert "calculator" in r


# ---------------------------------------------------------------------------
# Tests: Skill.matches()
# ---------------------------------------------------------------------------


def test_skill_matches_trigger_patterns():
    reg = _make_registry()
    calc = reg.get("calculator")
    assert calc.matches("calculate 5 + 3")
    assert calc.matches("What is sqrt(144)?")
    assert calc.matches("compute 2 ** 8")
    assert calc.matches("evaluate sin(pi)")


def test_skill_does_not_match_unrelated_query():
    reg = _make_registry()
    calc = reg.get("calculator")
    assert not calc.matches("Tell me a joke")
    assert not calc.matches("What is the capital of France?")


def test_find_matching_returns_calculator():
    reg = _make_registry()
    matches = reg.find_matching("calculate 10 * 5")
    names = [s.name for s in matches]
    assert "calculator" in names


def test_find_matching_returns_empty_for_no_match():
    reg = _make_registry()
    matches = reg.find_matching("Tell me a bedtime story")
    assert matches == []


# ---------------------------------------------------------------------------
# Tests: Skill.to_tool_definition()
# ---------------------------------------------------------------------------


def test_to_tool_definition_has_required_fields():
    reg = _make_registry()
    calc = reg.get("calculator")
    td = calc.to_tool_definition()
    assert td.name == "calculator"
    assert len(td.description) > 10
    assert "query" in td.parameters


# ---------------------------------------------------------------------------
# Tests: Calculator skill execute()
# ---------------------------------------------------------------------------


def _run_calc(expression: str) -> SkillOutput:
    reg = _make_registry()
    calc = reg.get("calculator")
    si = SkillInput(query=expression, parameters={"expression": expression})
    return asyncio.run(calc.execute(si))


def _run_skill(name: str, query: str, parameters: dict | None = None) -> SkillOutput:
    reg = _make_registry()
    skill = reg.get(name)
    si = SkillInput(query=query, parameters=parameters or {})
    return asyncio.run(skill.execute(si))


def test_calc_basic_addition():
    out = _run_calc("2 + 2")
    assert out.success
    assert out.result == 4


def test_calc_subtraction():
    out = _run_calc("10 - 3")
    assert out.success
    assert out.result == 7


def test_calc_multiplication():
    out = _run_calc("6 * 7")
    assert out.success
    assert out.result == 42


def test_calc_division():
    out = _run_calc("100 / 4")
    assert out.success
    assert out.result == 25.0


def test_calc_floor_division():
    out = _run_calc("7 // 2")
    assert out.success
    assert out.result == 3


def test_calc_modulo():
    out = _run_calc("10 % 3")
    assert out.success
    assert out.result == 1


def test_calc_power():
    out = _run_calc("2 ** 10")
    assert out.success
    assert out.result == 1024


def test_calc_caret_treated_as_power():
    out = _run_calc("2 ^ 10")
    assert out.success
    assert out.result == 1024


def test_calc_sqrt():
    out = _run_calc("sqrt(144)")
    assert out.success
    assert math.isclose(out.result, 12.0)


def test_calc_sin_zero():
    out = _run_calc("sin(0)")
    assert out.success
    assert math.isclose(out.result, 0.0, abs_tol=1e-10)


def test_calc_cos_zero():
    out = _run_calc("cos(0)")
    assert out.success
    assert math.isclose(out.result, 1.0)


def test_calc_pi_constant():
    out = _run_calc("pi")
    assert out.success
    assert math.isclose(out.result, math.pi)


def test_calc_e_constant():
    out = _run_calc("e")
    assert out.success
    assert math.isclose(out.result, math.e)


def test_calc_complex_expression():
    out = _run_calc("(3 + 4) * 5 - 2")
    assert out.success
    assert out.result == 33


def test_calc_math_prefix():
    out = _run_calc("math.sqrt(81)")
    assert out.success
    assert math.isclose(out.result, 9.0)


def test_calc_nested_functions():
    out = _run_calc("sqrt(abs(-25))")
    assert out.success
    assert math.isclose(out.result, 5.0)


def test_calc_division_by_zero():
    out = _run_calc("1 / 0")
    assert not out.success
    assert "zero" in out.error.lower()


def test_calc_invalid_expression():
    out = _run_calc("import os")
    assert not out.success


def test_calc_unknown_function():
    out = _run_calc("evil(42)")
    assert not out.success
    assert out.error is not None


def test_calc_metadata_contains_expression():
    out = _run_calc("2 + 2")
    assert out.success
    assert out.metadata.get("expression") == "2 + 2"


def test_calc_metadata_contains_type():
    out = _run_calc("2 + 2")
    assert "type" in out.metadata


# ---------------------------------------------------------------------------
# Tests: Unit converter skill execute()
# ---------------------------------------------------------------------------


def test_unit_converter_km_to_miles():
    out = _run_skill("unit_converter", "convert 5 km to miles")
    assert out.success
    assert math.isclose(out.result, 3.106856, rel_tol=1e-6)


def test_unit_converter_temperature_f_to_c():
    out = _run_skill("unit_converter", "convert 41 F to C")
    assert out.success
    assert math.isclose(out.result, 5.0, abs_tol=1e-6)


def test_unit_converter_cross_category_fails():
    out = _run_skill("unit_converter", "convert 5 km to kg")
    assert not out.success
    assert out.error is not None
    assert "Cannot convert between" in out.error


def test_unit_converter_structured_params():
    out = _run_skill(
        "unit_converter",
        "",
        parameters={"value": 1, "from_unit": "kg", "to_unit": "lb"},
    )
    assert out.success
    assert math.isclose(out.result, 2.204624, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Dictionary skill execute()
# ---------------------------------------------------------------------------


def test_dictionary_define_word_from_query():
    out = _run_skill("dictionary", "define ephemeral")
    assert out.success
    assert "short time" in out.result
    assert out.metadata.get("word") == "ephemeral"


def test_dictionary_word_from_params():
    out = _run_skill("dictionary", "", parameters={"word": "entropy"})
    assert out.success
    assert "uncertainty" in out.result
    assert out.metadata.get("part_of_speech") == "noun"


def test_dictionary_unknown_word_fails():
    out = _run_skill("dictionary", "define definitely_not_a_real_word")
    assert not out.success
    assert out.error is not None
    assert "not found" in out.error.lower()


# ---------------------------------------------------------------------------
# Tests: DateTime skill execute()
# ---------------------------------------------------------------------------


def test_datetime_days_between():
    out = _run_skill("datetime_calc", "days between 2024-01-01 and 2024-01-31")
    assert out.success
    assert out.result == 30


def test_datetime_add_days():
    out = _run_skill("datetime_calc", "add 10 days to 2024-01-15")
    assert out.success
    assert out.result == "2024-01-25"


def test_datetime_day_of_week():
    out = _run_skill("datetime_calc", "what day of the week is 2024-07-04")
    assert out.success
    assert out.result == "Thursday"


def test_datetime_bad_query_fails():
    out = _run_skill("datetime_calc", "tell me the weather tomorrow")
    assert not out.success
    assert out.error is not None
    assert "could not parse" in out.error.lower()
