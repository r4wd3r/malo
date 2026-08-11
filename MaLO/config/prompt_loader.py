"""Prompt configuration loader."""

from __future__ import annotations

from pathlib import Path

from MaLO.config.models import PromptConfig
from MaLO.utils.yaml_loader import load_yaml


def load_prompt_config(path: str | Path) -> PromptConfig:
    """Loads and validates a prompt configuration.

    Args:
        path: Path to the prompt YAML file.

    Returns:
        Parsed and validated prompt configuration.
    """
    raw_config = load_yaml(path)
    return PromptConfig.model_validate(raw_config)
