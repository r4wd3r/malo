"""Experiment configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from MaLO.config.models import ExperimentConfig
from MaLO.utils.yaml_loader import load_yaml


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Loads and validates an experiment configuration.

    Args:
        path: Path to the experiment YAML file.

    Returns:
        Parsed and validated experiment configuration.
    """
    raw_config = load_yaml(path)
    return ExperimentConfig.model_validate(raw_config)


def _deep_merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge two dicts, replacing lists and scalars."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_set_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    """Apply dotted-path overrides in-place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override {item!r}; expected KEY=VALUE format."
            )
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(
                f"Invalid override {item!r}; key must not be empty."
            )
        value = load_yaml_value(raw_value)
        _set_by_dotted_path(config, key, value)


def _set_by_dotted_path(
    config: dict[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    parts = dotted_path.split(".")
    current: dict[str, Any] = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            raise ValueError(
                f"Cannot set {dotted_path!r}; {part!r} is not a mapping."
            )
        current = current[part]
    current[parts[-1]] = value


def _apply_agent_override(
    config: dict[str, Any],
    field: str,
    value: str | None,
) -> None:
    if value is None:
        return
    agents = config.get("agents")
    if not isinstance(agents, dict):
        return
    for agent_config in agents.values():
        if isinstance(agent_config, dict):
            agent_config[field] = value


def load_experiment_config_layered(
    base_path: str | Path,
    override_paths: list[str | Path] | None = None,
    set_overrides: list[str] | None = None,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> ExperimentConfig:
    """Loads and validates a layered experiment configuration."""
    raw_config = load_experiment_dict_layered(
        base_path=base_path,
        override_paths=override_paths,
        set_overrides=set_overrides,
        model_override=model_override,
        provider_override=provider_override,
    )
    return ExperimentConfig.model_validate(raw_config)


def load_experiment_dict_layered(
    base_path: str | Path,
    override_paths: list[str | Path] | None = None,
    set_overrides: list[str] | None = None,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> dict[str, Any]:
    """Loads a layered experiment configuration as a dict."""
    raw_config = load_yaml(base_path)
    for override_path in override_paths or []:
        override = load_yaml(override_path)
        raw_config = _deep_merge_dicts(raw_config, override)
    if set_overrides:
        _apply_set_overrides(raw_config, set_overrides)
    _apply_agent_override(raw_config, "model", model_override)
    _apply_agent_override(raw_config, "provider", provider_override)
    return raw_config


def load_yaml_value(raw_value: str) -> Any:
    """Parse a scalar override value using YAML rules."""
    loaded = load_yaml_from_string(raw_value)
    return loaded


def load_yaml_from_string(raw_value: str) -> Any:
    """Loads YAML from a scalar string."""
    import yaml

    return yaml.safe_load(raw_value)
