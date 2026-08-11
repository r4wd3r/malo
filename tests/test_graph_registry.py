from __future__ import annotations

from dataclasses import dataclass

from MaLO.config.models import AgentsConfig, BaseAgentConfig, EnvironmentConfig, ExperimentConfig, LimitsConfig
from MaLO.graphs.registry import GRAPH_BUILDERS


@dataclass
class _Dummy:
    pass


def _config_for_arch(arch: str, agents: AgentsConfig) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test",
        log_to_wandb=False,
        environment=EnvironmentConfig(docker_image="test", setup_commands=[]),
        architecture=arch,  # type: ignore[arg-type]
        limits=LimitsConfig(
            max_turns=5,
            max_cost_usd=1.0,
            observation_truncation_chars=1000,
        ),
        agents=agents,
    )


def test_graph_registry_contains_known_architectures() -> None:
    assert set(GRAPH_BUILDERS.keys()) == {
        "single_react",
        "reasoner_executor",
        "orchestrator_workers",
        "orchestrator_critic_workers",
    }


def test_graph_registry_required_agents_metadata() -> None:
    assert GRAPH_BUILDERS["single_react"].required_agents == ["agent"]
    assert GRAPH_BUILDERS["reasoner_executor"].required_agents == [
        "orchestrator",
        "worker",
    ]
    assert GRAPH_BUILDERS["orchestrator_workers"].required_agents == [
        "orchestrator",
        "worker",
    ]
    assert GRAPH_BUILDERS["orchestrator_critic_workers"].required_agents == [
        "orchestrator",
        "critic",
        "worker",
    ]
