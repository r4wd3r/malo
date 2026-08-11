from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import MaLO.run_helpers as run_helpers
from MaLO.challenges.models import Goal, GoalType, ChallengeData
from MaLO.config.models import AgentsConfig, BaseAgentConfig, EnvironmentConfig, ExperimentConfig, LimitsConfig


@dataclass
class _FakeEnv:
    teardown_called: bool = False

    def teardown(self) -> None:
        self.teardown_called = True


def _build_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test",
        log_to_wandb=False,
        environment=EnvironmentConfig(docker_image="test", setup_commands=[]),
        architecture="single_react",
        limits=LimitsConfig(
            max_turns=5,
            max_cost_usd=1.0,
            observation_truncation_chars=1000,
        ),
        agents=AgentsConfig(
            agent=BaseAgentConfig(
                model="test", temperature=0.0, prompts_file=""
            )
        ),
    )


def _build_challenge() -> ChallengeData:
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "flag"},
    )
    return ChallengeData(
        id="c1",
        path="/tmp",
        name="c",
        category="pwn",
        description="d",
        goal=goal,
    )


def test_challenge_execution_context_teardown(monkeypatch) -> None:
    config = _build_config()
    challenge = _build_challenge()
    env = _FakeEnv()
    stop_called: dict[str, Any] = {"called": False}

    def _fake_prepare(*_args: Any, **_kwargs: Any) -> tuple[_FakeEnv, str, bool]:
        return env, "/ctf", True

    def _fake_stop(_challenge: Any) -> None:
        stop_called["called"] = True

    monkeypatch.setattr(run_helpers, "prepare_challenge_environment", _fake_prepare)
    import MaLO.environment.challenge_container as challenge_container

    monkeypatch.setattr(challenge_container, "stop_challenge_container", _fake_stop)

    with run_helpers.challenge_execution_context(
        config=config,
        challenge=challenge,
        skip_env=False,
    ) as (_env_manager, workdir, tool_executor):
        assert workdir == "/ctf"
        assert tool_executor is not None

    assert env.teardown_called is True
    assert stop_called["called"] is True
