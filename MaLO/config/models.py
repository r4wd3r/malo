"""Pydantic models for MaLO LangGraph configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ArchitectureType = Literal[
    "single_react",
    "reasoner_executor",
    "orchestrator_workers",
    "orchestrator_critic_workers",
]


class EnvironmentConfig(BaseModel):
    """Configuration for the Docker-based environment."""

    model_config = ConfigDict(extra="forbid")

    docker_image: str
    setup_commands: list[str] = Field(default_factory=list)
    timeout_seconds: int = 600
    # When set, the agent container is attached to this network (e.g. ctfnet).
    docker_network: str | None = None


class LimitsConfig(BaseModel):
    """Global runtime limits for an experiment."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int
    max_cost_usd: float
    observation_truncation_chars: int
    max_message_chars: int | None = None
    recursion_limit: int = 100
    max_workers: int = 3
    worker_max_turns: int = 10
    worker_timeout_seconds: int | None = None
    challenge_timeout_seconds: int | None = None
    random_seed: int | None = None


class BaseAgentConfig(BaseModel):
    """Shared configuration for a single agent."""

    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float
    prompts_file: str
    provider: str | None = None
    base_url: str | None = None


class CriticConfig(BaseAgentConfig):
    """Configuration for the critic agent."""

    strictness: str


class AgentsConfig(BaseModel):
    """Configuration for all agents in an experiment."""

    model_config = ConfigDict(extra="forbid")

    # Single agent mode (single_react)
    agent: BaseAgentConfig | None = None

    # Multi-agent mode (orchestrator architectures)
    orchestrator: BaseAgentConfig | None = None
    critic: CriticConfig | None = None
    worker: BaseAgentConfig | None = None


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    log_to_wandb: bool
    wandb_project: str | None = None
    environment: EnvironmentConfig
    architecture: ArchitectureType
    limits: LimitsConfig
    agents: AgentsConfig

    @model_validator(mode="after")
    def _validate_worker_timeout_for_orchestrator_architectures(self) -> "ExperimentConfig":
        """Require worker_timeout_seconds for orchestrator_workers and orchestrator_critic_workers."""
        if self.architecture not in (
            "orchestrator_workers",
            "orchestrator_critic_workers",
        ):
            return self
        wt = self.limits.worker_timeout_seconds
        if wt is None:
            raise ValueError(
                f"{self.architecture!r} requires limits.worker_timeout_seconds to be set."
            )
        if wt <= 0:
            raise ValueError(
                f"{self.architecture!r} requires limits.worker_timeout_seconds > 0 "
                f"(got {wt})."
            )
        return self


class PromptConfig(BaseModel):
    """Prompt configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    instruction_template: str | None = None
