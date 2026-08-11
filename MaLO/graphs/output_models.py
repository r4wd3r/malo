"""Typed output models for JSON-producing agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReasonerCommand(BaseModel):
    """Command emitted by the reasoner."""

    model_config = ConfigDict(extra="allow")

    type: Literal["bash", "python", "submit", "give_up", "error"]
    script: str | None = None
    answer: str | list[str] | None = None
    reason: str | None = None
    error: str | None = None


class ReasonerOutput(BaseModel):
    """Reasoner output with thought and command."""

    model_config = ConfigDict(extra="allow")

    thought: str = ""
    command: ReasonerCommand


class OrchestratorTask(BaseModel):
    """Task emitted by orchestrator for workers."""

    model_config = ConfigDict(extra="allow")

    type: str = ""
    description: str = ""
    details: str | None = None
    role: str | None = None
    answer: str | None = None


class OrchestratorOutput(BaseModel):
    """Orchestrator output containing state update and tasks."""

    model_config = ConfigDict(extra="allow")

    state_update: dict[str, Any] = Field(default_factory=dict)
    tasks: list[OrchestratorTask] = Field(default_factory=list)


class OrchestratorCriticOutput(BaseModel):
    """Orchestrator output for critic workflow."""

    model_config = ConfigDict(extra="allow")

    state_update: dict[str, Any] = Field(default_factory=dict)
    proposed_plan: dict[str, Any] = Field(default_factory=dict)


class CriticOutput(BaseModel):
    """Critic verdict output."""

    model_config = ConfigDict(extra="allow")

    verdict: Literal["APPROVE", "REJECT"] = "APPROVE"
    feedback: str | None = None
    suggestion: str | None = None

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.upper()
        return value
