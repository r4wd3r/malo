from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from MaLO.challenges.models import BenchmarkType, ChallengeData, Goal, GoalType
from MaLO.config.models import EnvironmentConfig, ExperimentConfig, LimitsConfig, AgentsConfig, BaseAgentConfig
from MaLO.runner import BatchRunner, BatchResult


def _make_config(architecture: str = "single_react") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test",
        log_to_wandb=False,
        environment=EnvironmentConfig(
            docker_image="malo:latest",
            setup_commands=[],
            timeout_seconds=10,
        ),
        architecture=architecture,  # type: ignore[arg-type]
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


def test_batch_runner_writes_results_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    challenge = ChallengeData(
        id="MALO-0001-24",
        path=str(tmp_path),
        name="Example",
        category="pwn",
        description="test",
        goal=Goal(
            goal_type=GoalType.EXACT_MATCH,
            description="flag",
            validation_data={"expected": "flag"},
        ),
        benchmark=BenchmarkType.MALO_BENCHMARK,
    )

    def _fake_load_challenges(_path: str, batch: bool = False):
        return [challenge]

    def _fake_run_graph_for_objective(*_args, **_kwargs):
        return {
            "termination_reason": "success",
            "turn_count": 2,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    monkeypatch.setattr("MaLO.runner.load_challenges", _fake_load_challenges)
    monkeypatch.setattr(
        "MaLO.runner.run_graph_for_objective",
        _fake_run_graph_for_objective,
    )

    config = _make_config()
    runner = BatchRunner(
        config=config,
        config_path=Path("/tmp/config.yaml"),
        skip_env=True,
        verbose=False,
    )

    output_path = tmp_path / "results.json"
    result = runner.run_batch(str(tmp_path), str(output_path))

    assert result.total_challenges == 1
    assert result.successful == 1
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["total_challenges"] == 1
    assert data["success_rate"] == 1.0
    assert data["results"][0]["termination_reason"] == "success"


def test_cli_batch_mode_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from MaLO import cli as cli_mod

    config = _make_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test", encoding="utf-8")

    class _FakeRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_batch(self, *_args, **_kwargs) -> BatchResult:
            return BatchResult(
                total_challenges=1,
                successful=1,
                failed=0,
                errors=0,
                results=[],
                start_time="",
                end_time="",
            )

    monkeypatch.setattr(cli_mod, "BatchRunner", _FakeRunner)

    code = cli_mod._run_batch_mode(
        config=config,
        config_path=config_path,
        challenges_dir=str(tmp_path),
        output_path=None,
        skip_env=True,
        disable_wandb=False,
        verbose=False,
        console_handler=None,
    )

    assert code == 0


def test_cli_batch_mode_passes_disable_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from MaLO import cli as cli_mod

    config = _make_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test", encoding="utf-8")

    captured: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

        def run_batch(self, *_args, **_kwargs) -> BatchResult:
            return BatchResult(
                total_challenges=1,
                successful=1,
                failed=0,
                errors=0,
                results=[],
                start_time="",
                end_time="",
            )

    monkeypatch.setattr(cli_mod, "BatchRunner", _FakeRunner)

    code = cli_mod._run_batch_mode(
        config=config,
        config_path=config_path,
        challenges_dir=str(tmp_path),
        output_path=None,
        skip_env=True,
        disable_wandb=True,
        verbose=False,
        console_handler=None,
    )

    assert code == 0
    assert captured.get("disable_wandb") is True
