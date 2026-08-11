from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from MaLO.config.loader import load_experiment_config
from MaLO.config.prompt_loader import load_prompt_config


def test_orchestrator_requires_worker_timeout(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.yaml"
    config_path.write_text(
        """
experiment_name: "orch"
log_to_wandb: false

environment:
  docker_image: "malo:latest"
  setup_commands: []
  timeout_seconds: 30

architecture: "orchestrator_workers"

limits:
  max_turns: 5
  max_cost_usd: 1.0
  observation_truncation_chars: 1000
  recursion_limit: 100
  max_workers: 2
  challenge_timeout_seconds: 60

agents:
  orchestrator:
    model: "openai/gpt-4o"
    temperature: 0.1
    prompts_file: "prompts/orchestrator_workers/orchestrator.yaml"
  worker:
    model: "openai/gpt-4o"
    temperature: 0.1
    prompts_file: "prompts/orchestrator_workers/worker.yaml"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_experiment_config(config_path)


def test_prompt_config_requires_system_prompt(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        """
# Missing system_prompt
instruction_template: "{objective}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_prompt_config(prompt_path)
