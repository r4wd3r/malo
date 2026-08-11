"""Challenge and goal models for MaLO LangGraph."""

from __future__ import annotations

import ast
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GoalType(str, Enum):
    """Supported goal validation types."""

    EXACT_MATCH = "exact_match"
    LIST_MATCH = "list_match"
    REGEX = "regex"


class BenchmarkType(str, Enum):
    """Supported benchmark formats."""

    MALO_BENCHMARK = "malo-benchmark"
    NYU_CTF = "nyuctf"
    INTERCTF = "interctf"


class Goal(BaseModel):
    """Validation goal for a challenge."""

    model_config = ConfigDict(extra="forbid")

    goal_type: GoalType
    description: str
    validation_data: dict[str, Any]
    metadata: dict[str, Any] | None = None

    def validate_answer(self, answer: str | list[str]) -> tuple[bool, str]:
        """Validates an answer against this goal.

        Args:
            answer: The answer to validate (string or list).

        Returns:
            Tuple of (is_valid, feedback_message).
        """
        if self.goal_type == GoalType.EXACT_MATCH:
            return self._validate_exact_match(answer)
        if self.goal_type == GoalType.LIST_MATCH:
            return self._validate_list_match(answer)
        if self.goal_type == GoalType.REGEX:
            return self._validate_regex(answer)
        return False, f"Unknown goal type: {self.goal_type}"

    def _validate_exact_match(
        self,
        answer: str | list[str],
    ) -> tuple[bool, str]:
        """Validates exact string match."""
        expected = self.validation_data.get("expected", "")
        case_sensitive = self.validation_data.get("case_sensitive", True)

        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        if case_sensitive:
            is_valid = answer == expected
        else:
            is_valid = answer.lower() == expected.lower()

        if is_valid:
            return True, "Answer accepted."
        return False, "Answer rejected."

    def _validate_list_match(
        self,
        answer: str | list[str],
    ) -> tuple[bool, str]:
        """Validates list comparison."""
        expected = self.validation_data.get("expected", [])
        order_matters = self.validation_data.get("order_matters", False)
        case_sensitive = self.validation_data.get("case_sensitive", True)

        expected_list = [str(item).strip() for item in expected]
        answer_list = _normalize_list_answer(answer, expected_list)

        if not case_sensitive:
            expected_list = [item.lower() for item in expected_list]
            answer_list = [item.lower() for item in answer_list]

        if order_matters:
            is_valid = answer_list == expected_list
        else:
            is_valid = set(answer_list) == set(expected_list)

        if is_valid:
            return True, "Answer accepted."
        return False, "Answer rejected."

    def _validate_regex(self, answer: str | list[str]) -> tuple[bool, str]:
        """Validates regex pattern match."""
        pattern = self.validation_data.get("pattern", "")
        flags_str = self.validation_data.get("flags", "")

        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        regex_flags = 0
        if "IGNORECASE" in flags_str.upper():
            regex_flags |= re.IGNORECASE
        if "MULTILINE" in flags_str.upper():
            regex_flags |= re.MULTILINE

        try:
            if re.match(pattern, answer, regex_flags):
                return True, "Answer accepted."
            return False, "Answer rejected."
        except re.error as err:
            return False, f"Invalid regex pattern: {err}"

    def get_summary(self) -> str:
        """Returns a human-readable summary of the goal."""
        if self.goal_type == GoalType.EXACT_MATCH:
            return f"Exact match validation"
        if self.goal_type == GoalType.LIST_MATCH:
            count = len(self.validation_data.get("expected", []))
            return f"List match ({count} items)"
        if self.goal_type == GoalType.REGEX:
            pattern = self.validation_data.get("pattern", "")
            return f"Regex: {pattern[:30]}..."
        return f"Unknown: {self.goal_type}"


def _normalize_list_answer(
    answer: str | list[str],
    expected: list[str] | None = None,
) -> list[str]:
    """Normalizes a list-style answer into a list of strings."""
    if isinstance(answer, dict):
        for key in (
            "answer",
            "answers",
            "items",
            "list",
            "values",
            "ips",
            "ip",
            "channels",
        ):
            if key in answer:
                return _normalize_list_answer(answer[key], expected)
        return [
            str(item).strip()
            for item in answer.values()
            if str(item).strip()
        ]

    if isinstance(answer, list):
        return [str(item).strip() for item in answer if str(item).strip()]

    raw = (answer or "").strip()
    if not raw:
        return []

    # Strip code fences if present.
    if "```" in raw:
        fenced = re.findall(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
        if fenced:
            raw = fenced[0].strip()
        else:
            raw = raw.replace("```", "")

    # Strip common bullet/numbering prefixes from multi-line inputs.
    raw = re.sub(r"(?m)^[\s]*[-*•]\s+", "", raw)
    raw = re.sub(r"(?m)^[\s]*\d+[\).\:\-]\s+", "", raw)

    if expected and _looks_like_ipv4_list(expected):
        extracted = _extract_ipv4_list(raw)
        if extracted:
            return extracted

    # Try JSON list if it looks like one.
    if raw.startswith("[") and raw.endswith("]"):
        try:
            import json

            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except json.JSONDecodeError:
            pass

        # Try Python-style list literals.
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except (ValueError, SyntaxError):
            pass

    # Fallback: split on common separators.
    tokens = re.split(r"[,;\n]+", raw)
    if len(tokens) > 1:
        return [_strip_list_prefix(t) for t in tokens if t.strip()]

    # If no separators, optionally split on whitespace when safe.
    if expected is not None and any(" " in item for item in expected):
        return [raw.strip().strip("'\"")] if raw.strip() else []

    if re.search(r"\s+", raw):
        tokens = re.split(r"\s+", raw)
        return [_strip_list_prefix(t) for t in tokens if t.strip()]

    return [_strip_list_prefix(raw)] if raw.strip() else []


def _strip_list_prefix(value: str) -> str:
    cleaned = value.strip().strip("'\"")
    cleaned = re.sub(r"^[\s]*\d+[\).\:\-]\s+", "", cleaned)
    return cleaned.strip()


def _looks_like_ipv4_list(values: list[str]) -> bool:
    return bool(values) and all(_is_ipv4(value) for value in values)


def _extract_ipv4_list(text: str) -> list[str]:
    candidates = re.findall(r"(?:\d{1,3}\.){3}\d{1,3}", text)
    return [ip for ip in candidates if _is_ipv4(ip)]


def _is_ipv4(value: str) -> bool:
    parts = value.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return False
    return all(0 <= num <= 255 for num in nums)


class ChallengeData(BaseModel):
    """Challenge metadata and configuration."""

    model_config = ConfigDict(extra="allow")

    challenge_id: str = Field(alias="id")
    path: str
    name: str
    category: str
    description: str
    goal: Goal
    date: str | None = None
    origin: str | None = None
    tags: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    container: str | None = None
    compose: str | None = None
    benchmark: BenchmarkType | None = None
    base_path: str | None = None
    canonical_name: str | None = None
    # Set when a challenge container is started; used in objective for service endpoint.
    service_hostname: str | None = None
    service_port: int | None = None

    @field_validator("goal", mode="before")
    @classmethod
    def parse_goal(cls, value: dict[str, Any] | Goal) -> Goal:
        """Parses goal from dict if needed."""
        if isinstance(value, Goal):
            return value
        return Goal(**value)
