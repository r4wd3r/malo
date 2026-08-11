"""Goal validation interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from MaLO.challenges.models import Goal


class GoalValidator(Protocol):
    """Protocol for goal validation."""

    def validate(self, answer: str | list[str]) -> tuple[bool, str]:
        """Validate an answer and return (is_valid, feedback)."""


@dataclass(frozen=True)
class DefaultGoalValidator:
    """Goal validator backed by a Goal model."""

    goal: Goal

    def validate(self, answer: str | list[str]) -> tuple[bool, str]:
        """Validate using the Goal model."""
        return self.goal.validate_answer(answer)
