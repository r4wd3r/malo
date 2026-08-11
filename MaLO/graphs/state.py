"""Shared graph state types."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def _reduce_termination_reason(left: str, right: str) -> str:
    """Reducer: keep first non-empty termination reason."""
    return left if left else right


class BaseState(TypedDict):
    """Base state fields shared by all graph architectures."""

    objective: str
    turn_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_call_count: int
    done: bool
    termination_reason: str
    needs_human: bool


class MessageState(BaseState):
    """State with message history for chat-based graphs."""

    messages: Annotated[list[dict[str, Any]], add_messages]


class OrchestratorState(BaseState):
    """Base state for orchestrator graphs with reducer fields."""

    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    total_tokens: Annotated[int, operator.add]
    tool_call_count: Annotated[int, operator.add]
    done: Annotated[bool, operator.or_]
    termination_reason: Annotated[str, _reduce_termination_reason]
    needs_human: Annotated[bool, operator.or_]


class WorkerState(TypedDict):
    """State for a single worker branch."""

    objective: str
    task: dict[str, Any]
    worker_id: int
    worker_role: str
    worker_messages: list[dict[str, Any]]
    worker_turn_count: int
    worker_report: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_call_count: int
    done: bool
    termination_reason: str
    needs_human: bool
