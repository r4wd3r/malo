"""Shared utilities for orchestrator graph implementations."""

from __future__ import annotations

import json
from typing import Any


def format_state_json(state: dict[str, Any]) -> str:
    """Formats orchestrator state as JSON string."""
    default_state = {
        "phase": "reconnaissance",
        "findings": [],
        "attempted": [],
        "current_hypothesis": "",
        "next_tasks": [],
    }
    merged = {**default_state, **state}
    return json.dumps(merged, indent=2)


def merge_state(
    current: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    """Merges state update into current state."""
    result = current.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], list) and isinstance(value, list):
            # Append to lists instead of replacing
            result[key] = result[key] + value
        else:
            result[key] = value

    return result


def extract_tasks(orch_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extracts tasks from orchestrator output.

    Expects "tasks" (array) only. For a single task use "tasks": [ {...} ].

    Args:
        orch_output: Parsed JSON output from orchestrator.

    Returns:
        List of task dictionaries.
    """
    tasks = orch_output.get("tasks", [])
    if isinstance(tasks, list):
        return tasks
    return []


def extract_tasks_from_plan(proposed_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Extracts tasks from approved proposed_plan.

    Expects "tasks" (array) only. For a single task use "tasks": [ {...} ].
    """
    tasks = proposed_plan.get("tasks", [])
    if isinstance(tasks, list):
        return tasks
    return []


def extract_worker_token_updates(
    state: dict[str, Any],
    response: Any,
) -> dict[str, int]:
    """Extracts token updates for worker state.

    Args:
        state: Current worker state.
        response: LLM response object.

    Returns:
        Dictionary with token count updates.
    """
    prompt_tokens = getattr(response, "prompt_tokens", 0)
    completion_tokens = getattr(response, "completion_tokens", 0)
    total_tokens = getattr(response, "total_tokens", 0)

    return {
        "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tokens,
        "completion_tokens": state.get("completion_tokens", 0) + completion_tokens,
        "total_tokens": state.get("total_tokens", 0) + total_tokens,
    }


def reduce_worker_reports(left: list[str], right: list[str] | None) -> list[str]:
    """Reducer: append unless explicitly reset with an empty list."""
    if right is None:
        return left
    if right == []:
        return []
    return left + right
