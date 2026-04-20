r"""
Dynamic skill registry.

Skills live in self-contained folders::

    skills/
      calculator/
        SKILL.md          ← required: name, description, trigger patterns
        skill.py          ← required: execute(input) -> output + SKILL_META dict
      web_search/
        SKILL.md
        skill.py

The registry scans a directory, imports each ``skill.py``, reads the
``SKILL_META`` dict, and registers the skill.  Adding or removing a skill
folder is the only change required — no framework code touches.

``SKILL_META`` shape (required keys)::

    SKILL_META = {
        "name": "calculator",
        "description": "Evaluates mathematical expressions.",
        "trigger_patterns": ["calculate", "compute", r"\d+\s*[\+\-\*\/]"],
        "version": "1.0.0",          # optional
        "author": "...",             # optional
    }

The ``execute`` function signature::

    async def execute(input: SkillInput) -> SkillOutput: ...
    # OR synchronous:
    def execute(input: SkillInput) -> SkillOutput: ...
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SkillInput(BaseModel):
    """Input passed to a skill's ``execute`` function."""

    query: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class SkillOutput(BaseModel):
    """Output returned by a skill's ``execute`` function."""

    result: Any
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def failure(cls, error: str, **kwargs: Any) -> "SkillOutput":
        return cls(result=None, success=False, error=error, **kwargs)


class SkillExecutionError(Exception):
    """Raised when a skill's execute function fails."""


# ---------------------------------------------------------------------------
# Skill descriptor
# ---------------------------------------------------------------------------


class Skill:
    """
    Runtime representation of a loaded skill.

    Attributes
    ----------
    name:
        Unique identifier.
    description:
        Human-readable description (used for model tool specs).
    trigger_patterns:
        List of regex strings.  ``matches(query)`` returns True when any pattern
        matches, used by the selection benchmark.
    version / author:
        Optional metadata.
    _execute:
        The raw callable loaded from ``skill.py``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        trigger_patterns: list[str],
        execute_fn: Callable[..., Any],
        version: str = "0.1.0",
        author: str = "unknown",
        skill_dir: Optional[Path] = None,
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.trigger_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in trigger_patterns
        ]
        self._execute = execute_fn
        self.version = version
        self.author = author
        self.skill_dir = skill_dir
        self.extra_meta: dict[str, Any] = extra_meta or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def matches(self, query: str) -> bool:
        """Return True if *query* matches any trigger pattern."""
        return any(p.search(query) for p in self.trigger_patterns)

    async def execute(self, input: SkillInput) -> SkillOutput:
        """
        Invoke the skill.  Handles both sync and async execute functions.
        Wraps exceptions in ``SkillOutput.failure``.
        """
        try:
            if asyncio.iscoroutinefunction(self._execute):
                return await self._execute(input)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._execute, input)
        except SkillExecutionError:
            raise
        except Exception as exc:
            logger.exception("Skill %r raised an unexpected error", self.name)
            return SkillOutput.failure(str(exc))

    def to_tool_definition(self) -> "Any":
        """Convert this skill to an adapter ToolDefinition for tool-call prompting."""
        from sLLM_eval_framework.adapters.base import ToolDefinition  # late import

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"query": {"type": "string", "description": "Input query or expression"}},
            required=["query"],
        )

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, version={self.version!r})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """
    Discovers and loads skills from a directory tree.

    Each subdirectory that contains a ``skill.py`` with a ``SKILL_META`` dict
    and an ``execute`` callable is registered as a skill.

    Usage
    -----
    >>> registry = SkillRegistry("/path/to/skills")
    >>> registry.load()
    >>> skill = registry.get("calculator")
    >>> result = await skill.execute(SkillInput(query="2 + 2"))
    """

    def __init__(self, skills_dir: str | Path) -> None:
        self._skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, reload: bool = False) -> int:
        """
        Scan *skills_dir* and register all discovered skills.

        Returns the number of skills successfully loaded.
        """
        if reload:
            self._skills.clear()

        count = 0
        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_py = skill_dir / "skill.py"
            if not skill_py.exists():
                continue
            try:
                skill = self._load_skill(skill_dir, skill_py)
                self._skills[skill.name] = skill
                count += 1
                logger.info("Loaded skill %r from %s", skill.name, skill_dir)
            except Exception as exc:
                logger.warning("Failed to load skill from %s: %s", skill_dir, exc)

        return count

    def _load_skill(self, skill_dir: Path, skill_py: Path) -> Skill:
        """Import ``skill.py`` and build a ``Skill`` from its ``SKILL_META``."""
        module_name = f"_skill_{skill_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, skill_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {skill_py}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        meta: dict[str, Any] = getattr(module, "SKILL_META", None)
        if meta is None:
            raise ValueError(f"{skill_py} must define SKILL_META dict")

        execute_fn = getattr(module, "execute", None)
        if execute_fn is None or not callable(execute_fn):
            raise ValueError(f"{skill_py} must define a callable `execute`")

        required = {"name", "description", "trigger_patterns"}
        missing = required - meta.keys()
        if missing:
            raise ValueError(f"SKILL_META missing keys: {missing}")

        extra = {k: v for k, v in meta.items() if k not in required | {"version", "author"}}

        return Skill(
            name=meta["name"],
            description=meta["description"],
            trigger_patterns=meta["trigger_patterns"],
            execute_fn=execute_fn,
            version=meta.get("version", "0.1.0"),
            author=meta.get("author", "unknown"),
            skill_dir=skill_dir,
            extra_meta=extra,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, name: str) -> Skill:
        """Return the named skill, raising ``KeyError`` if not found."""
        if name not in self._skills:
            raise KeyError(f"Skill {name!r} not registered. Available: {self.names}")
        return self._skills[name]

    def find_matching(self, query: str) -> list[Skill]:
        """Return all skills whose trigger patterns match *query*, sorted by name."""
        return sorted(
            [s for s in self._skills.values() if s.matches(query)],
            key=lambda s: s.name,
        )

    def all(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={self.names})"
