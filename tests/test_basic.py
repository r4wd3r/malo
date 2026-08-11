from __future__ import annotations

from pathlib import Path

import pytest

from MaLO.challenges import load_challenge
from MaLO.config.loader import load_experiment_config
from MaLO.environment.environment_manager import EnvironmentManager
from MaLO.config.models import EnvironmentConfig


def test_load_malo_challenge(tmp_path: Path) -> None:
    challenge_dir = tmp_path / "challenge"
    src_dir = challenge_dir / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "artifact.bin").write_text("data", encoding="utf-8")

    (challenge_dir / "challenge.yml").write_text(
        """
ID: MALO-0001-24
name: Example Challenge
date: 2024
origin: Test
category: pwn
description: Simple test challenge
flags:
  - CTF{test_flag}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    challenge = load_challenge(str(challenge_dir))

    assert challenge.challenge_id == "MALO-0001-24"
    assert challenge.name == "Example Challenge"
    assert challenge.category == "pwn"
    assert challenge.goal.goal_type.value == "exact_match"
    assert "src/artifact.bin" in challenge.files


def test_load_nyu_challenge(tmp_path: Path) -> None:
    # Path format: <year>/<event>/<category>/<challenge>
    challenge_dir = tmp_path / "2023" / "CSAW" / "pwn" / "sample"
    challenge_dir.mkdir(parents=True)

    (challenge_dir / "challenge.json").write_text(
        """
{
  "name": "Sample",
  "description": "NYU test",
  "category": "pwn",
  "flag": "CTF{nyu}",
  "files": ["binary"]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    challenge = load_challenge(str(challenge_dir))

    assert challenge.benchmark.value == "nyuctf"
    assert challenge.category == "pwn"
    assert challenge.goal.goal_type.value == "exact_match"
    assert challenge.files == ["binary"]


class _FakeContainer:
    def __init__(self) -> None:
        self.short_id = "deadbeef"

    def exec_run(self, *_args, **_kwargs):
        return 0, (b"", b"")

    def kill(self) -> None:
        return None

    def remove(self, force: bool = False) -> None:
        return None


class _FakeImages:
    def __init__(self, should_exist: bool) -> None:
        self.should_exist = should_exist
        self.pulled = False

    def get(self, _name: str) -> None:
        if not self.should_exist:
            from MaLO.environment import environment_manager as env_mod

            raise env_mod.docker.errors.ImageNotFound("missing")

    def pull(self, _name: str) -> None:
        self.pulled = True


class _FakeContainers:
    def __init__(self) -> None:
        self.ran_with = None

    def run(self, _image: str, **kwargs):
        self.ran_with = kwargs
        return _FakeContainer()


class _FakeDockerClient:
    def __init__(self, images: _FakeImages, containers: _FakeContainers) -> None:
        self.images = images
        self.containers = containers


def test_environment_manager_setup_pulls_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    images = _FakeImages(should_exist=False)
    containers = _FakeContainers()
    client = _FakeDockerClient(images, containers)

    from MaLO.environment import environment_manager as env_mod

    monkeypatch.setattr(env_mod.docker, "from_env", lambda: client)

    cfg = EnvironmentConfig(
        docker_image="malo:latest",
        setup_commands=[],
        timeout_seconds=10,
    )
    manager = EnvironmentManager(cfg)
    manager.setup()

    assert images.pulled is True
    assert containers.ran_with is not None
    assert containers.ran_with.get("privileged") is True
    assert containers.ran_with.get("working_dir") == "/ctf"

    manager.teardown()


def test_load_experiment_config(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
experiment_name: "basic"
log_to_wandb: false

environment:
  docker_image: "malo:latest"
  setup_commands: []
  timeout_seconds: 30

architecture: "single_react"

limits:
  max_turns: 5
  max_cost_usd: 1.0
  observation_truncation_chars: 1000
  recursion_limit: 100
  challenge_timeout_seconds: 60
  worker_timeout_seconds: 10

agents:
  agent:
    model: "openai/gpt-4o"
    temperature: 0.1
    prompts_file: "prompts/single_react/agent.yaml"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.architecture == "single_react"
    assert config.limits.max_turns == 5
    assert config.environment.docker_image == "malo:latest"
    assert config.agents.agent is not None
