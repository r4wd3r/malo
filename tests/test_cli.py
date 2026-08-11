from __future__ import annotations

from pathlib import Path

import yaml

from MaLO.cli import _build_parser
from MaLO.config.loader import load_experiment_config_layered
from MaLO.config.models import (
    AgentsConfig,
    BaseAgentConfig,
    EnvironmentConfig,
    ExperimentConfig,
    LimitsConfig,
)


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test",
        log_to_wandb=False,
        environment=EnvironmentConfig(
            docker_image="malo:latest",
            setup_commands=[],
            timeout_seconds=10,
        ),
        architecture="single_react",  # type: ignore[arg-type]
        limits=LimitsConfig(
            max_turns=3,
            max_cost_usd=1.0,
            observation_truncation_chars=1000,
            recursion_limit=50,
            max_workers=1,
            worker_timeout_seconds=5,
            challenge_timeout_seconds=30,
        ),
        agents=AgentsConfig(
            agent=BaseAgentConfig(
                model="test-model",
                temperature=0.0,
                prompts_file="prompts/single_react/agent.yaml",
            ),
        ),
    )


def test_cli_challenge_short_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            "config/single_react/architecture/single_react.yaml",
            "-c",
            "path/to/challenge",
        ]
    )
    assert args.challenge == "path/to/challenge"


def test_cli_batch_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            "config/single_react/architecture/single_react.yaml",
            "--batch",
            "benchmarks/malo-benchmark",
        ]
    )
    assert args.batch == "benchmarks/malo-benchmark"


def test_config_layering_applies_overrides(tmp_path: Path) -> None:
    base = _make_config()
    base_dict = base.model_dump()
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base_dict), encoding="utf-8")

    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        yaml.safe_dump({"limits": {"max_turns": 7}}),
        encoding="utf-8",
    )

    layered = load_experiment_config_layered(
        base_path=base_path,
        override_paths=[override_path],
        set_overrides=["limits.max_cost_usd=2.5"],
        model_override="new-model",
        provider_override="openrouter",
    )

    assert layered.limits.max_turns == 7
    assert layered.limits.max_cost_usd == 2.5
    assert layered.agents.agent is not None
    assert layered.agents.agent.model == "new-model"
    assert layered.agents.agent.provider == "openrouter"
