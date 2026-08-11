"""Command-line interface for MaLO LangGraph."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from MaLO.config.loader import (
    load_experiment_config_layered,
    load_experiment_dict_layered,
)
from MaLO.config.models import ExperimentConfig
from MaLO.observability.console_handler import ConsoleHandler
from MaLO.observability.metrics import RunMetrics
from MaLO.observability.wandb_handler import WandBHandler
from MaLO.run_helpers import (
    build_objective_from_challenge,
    challenge_execution_context,
    compute_run_stats,
    get_primary_model,
    run_graph_for_objective,
)
from MaLO.runner import BatchRunner
from MaLO.utils.weave_utils import get_weave
from MaLO.challenges import load_challenge


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaLO CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a challenge or batch")
    _add_common_config_args(run_parser)

    target_group = run_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "-c",
        "--challenge",
        default=None,
        help="Path to a single challenge folder.",
    )
    target_group.add_argument(
        "--batch",
        default=None,
        help="Path to directory containing multiple challenges (batch mode).",
    )
    run_parser.add_argument(
        "--output",
        default=None,
        help="Path to save batch results JSON.",
    )
    run_parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Skip Docker environment startup.",
    )
    run_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging even if configured.",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output including full tool args and reasoning.",
    )
    run_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Show only final result line.",
    )

    list_parser = subparsers.add_parser(
        "list", help="List available architecture configs"
    )
    list_parser.add_argument(
        "--root",
        default="config",
        help="Root directory to scan for architecture configs.",
    )

    config_parser = subparsers.add_parser(
        "config", help="Inspect or validate configs"
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", required=True
    )

    show_parser = config_subparsers.add_parser(
        "show", help="Show resolved configuration"
    )
    _add_common_config_args(show_parser)

    validate_parser = config_subparsers.add_parser(
        "validate", help="Validate configuration"
    )
    _add_common_config_args(validate_parser)

    return parser


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    base_group = parser.add_mutually_exclusive_group(required=True)
    base_group.add_argument(
        "--config",
        default=None,
        help="Experiment YAML file.",
    )
    base_group.add_argument(
        "--arch",
        default=None,
        help="Architecture name mapped to config/<arch>/architecture/<arch>.yaml.",
    )
    parser.add_argument(
        "--overrides",
        action="append",
        default=[],
        help="Override YAML file (repeatable).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Inline override in dotted KEY=VALUE format (repeatable).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model for all configured agents.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override provider for all configured agents.",
    )


def _resolve_base_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    if args.arch:
        arch = args.arch
        return Path("config") / arch / "architecture" / f"{arch}.yaml"
    raise ValueError("Either --config or --arch is required.")


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    base_path = _resolve_base_config_path(args)
    return load_experiment_config_layered(
        base_path=base_path,
        override_paths=args.overrides,
        set_overrides=args.set,
        model_override=args.model,
        provider_override=args.provider,
    )


def _load_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    base_path = _resolve_base_config_path(args)
    return load_experiment_dict_layered(
        base_path=base_path,
        override_paths=args.overrides,
        set_overrides=args.set,
        model_override=args.model,
        provider_override=args.provider,
    )


def _list_architecture_configs(root: str) -> list[Path]:
    root_path = Path(root)
    return sorted(root_path.glob("**/architecture/*.yaml"))


def _run_single_challenge(
    config: ExperimentConfig,
    config_path: Path,
    challenge_path: str,
    skip_env: bool,
    disable_wandb: bool,
    console_handler: ConsoleHandler,
) -> dict[str, Any]:
    challenge = load_challenge(challenge_path)

    console_handler.on_challenge_start(
        name=challenge.name,
        category=challenge.category,
        architecture=config.architecture,
        max_turns=config.limits.max_turns,
    )

    primary_model = get_primary_model(config)

    use_wandb = config.log_to_wandb and not disable_wandb
    wandb_handler = WandBHandler(
        project=config.wandb_project or "malo-experiments",
        experiment_name=f"{config.experiment_name}_{challenge.name}",
        config={
            "architecture": config.architecture,
            "model": primary_model,
            "challenge": challenge.name,
            "category": challenge.category,
        },
        enabled=use_wandb,
    )
    wandb_handler.start()

    metrics = RunMetrics()

    with challenge_execution_context(
        config=config,
        challenge=challenge,
        skip_env=skip_env,
        console_handler=console_handler,
    ) as (_env_manager, workdir, tool_executor):
        objective = build_objective_from_challenge(
            challenge=challenge,
            workdir=workdir,
            include_hints=False,
        )
        final_state = run_graph_for_objective(
            config=config,
            config_path=config_path,
            tool_executor=tool_executor,
            objective=objective,
            console_handler=console_handler,
        )

        stats = compute_run_stats(
            final_state=final_state,
            primary_model=primary_model,
            max_cost_usd=config.limits.max_cost_usd,
            console_handler=console_handler,
        )
        metrics.turn_count = stats["turn_count"]
        metrics.total_prompt_tokens = stats["prompt_tokens"]
        metrics.total_completion_tokens = stats["completion_tokens"]
        metrics.tool_calls = final_state.get("tool_call_count", 0)
        metrics.finalize(
            success=stats["success"],
            termination_reason=stats["termination_reason"],
        )

        console_handler.on_run_end(
            success=stats["success"],
            termination_reason=stats["termination_reason"],
            turn_count=stats["turn_count"],
            total_tokens=stats["total_tokens"],
            cost_usd=stats["cost_usd"],
        )

        wandb_handler.finish(metrics)

        return final_state


def _run_batch_mode(
    config: ExperimentConfig,
    config_path: Path,
    challenges_dir: str,
    output_path: str | None,
    skip_env: bool,
    disable_wandb: bool,
    verbose: bool = False,
    console_handler: ConsoleHandler | None = None,
) -> int:
    runner = BatchRunner(
        config=config,
        config_path=config_path,
        wandb_project=config.wandb_project,
        skip_env=skip_env,
        disable_wandb=disable_wandb,
        verbose=verbose,
        timeout_seconds=config.environment.timeout_seconds,
        console_handler=console_handler,
    )

    batch_result = runner.run_batch(challenges_dir, output_path)

    return 0 if batch_result.errors == 0 else 1


def _init_weave(
    config: ExperimentConfig,
    console_handler: ConsoleHandler,
) -> bool:
    weave = get_weave()
    if config.log_to_wandb and getattr(weave, "is_available", False):
        project = config.wandb_project or "malo-langgraph"
        weave.init(project)
        console_handler.print_info(f"Weave tracing enabled for project: {project}")
        return True
    return False


def _weave_flush_delay(weave_initialized: bool) -> None:
    if weave_initialized:
        time.sleep(2)


def _handle_run(args: argparse.Namespace) -> int:
    config = _load_config(args)
    config_path = _resolve_base_config_path(args)

    console_handler = ConsoleHandler(
        verbose=args.verbose,
        quiet=args.quiet,
    )

    weave_initialized = False
    if not args.no_wandb:
        weave_initialized = _init_weave(config, console_handler)

    if args.batch:
        result = _run_batch_mode(
            config=config,
            config_path=config_path,
            challenges_dir=args.batch,
            output_path=args.output,
            skip_env=args.skip_env,
            disable_wandb=args.no_wandb,
            verbose=args.verbose,
            console_handler=console_handler,
        )
        _weave_flush_delay(weave_initialized)
        return result

    final_state = _run_single_challenge(
        config=config,
        config_path=config_path,
        challenge_path=args.challenge,
        skip_env=args.skip_env,
        disable_wandb=args.no_wandb,
        console_handler=console_handler,
    )

    termination_reason = final_state.get("termination_reason", "")
    _weave_flush_delay(weave_initialized)
    return 0 if termination_reason == "success" else 1


def _handle_list(args: argparse.Namespace) -> int:
    configs = _list_architecture_configs(args.root)
    for config_path in configs:
        alias = config_path.stem
        print(f"{alias}\t{config_path}")
    return 0


def _handle_config_show(args: argparse.Namespace) -> int:
    config_dict = _load_config_dict(args)
    print(yaml.safe_dump(config_dict, sort_keys=False).rstrip())
    return 0


def _handle_config_validate(args: argparse.Namespace) -> int:
    _load_config(args)
    print("Config is valid.")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return _handle_run(args)
    if args.command == "list":
        return _handle_list(args)
    if args.command == "config":
        if args.config_command == "show":
            return _handle_config_show(args)
        if args.config_command == "validate":
            return _handle_config_validate(args)

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit("MaLO CLI is available via the `malo` command only.")
