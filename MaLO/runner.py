"""Batch runner for executing multiple challenges."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from MaLO.challenges import ChallengeData, load_challenges
from MaLO.config.models import ExperimentConfig
from MaLO.observability.batch_serializer import batch_result_to_dict
from MaLO.observability.console_handler import ConsoleHandler
from MaLO.run_helpers import (
    build_objective_from_challenge,
    challenge_execution_context,
    compute_run_stats,
    get_primary_model,
    run_graph_for_objective,
)
logger = logging.getLogger(__name__)


@dataclass
class ChallengeResult:
    """Result of running a single challenge."""

    challenge_id: str
    challenge_name: str
    success: bool
    termination_reason: str
    turn_count: int
    total_tokens: int
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class BatchResult:
    """Aggregated results from a batch run."""

    total_challenges: int
    successful: int
    failed: int
    errors: int
    results: list[ChallengeResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Converts batch result to dictionary."""
        return batch_result_to_dict(self)

    def save(self, path: str) -> None:
        """Saves batch results to a JSON file.

        Args:
            path: Path to save the results.
        """
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class BatchRunner:
    """Runs multiple challenges in batch mode."""

    def __init__(
        self,
        config: ExperimentConfig,
        config_path: Path,
        wandb_project: str | None = None,
        skip_env: bool = False,
        disable_wandb: bool = False,
        verbose: bool = False,
        timeout_seconds: int | None = None,
        console_handler: ConsoleHandler | None = None,
    ) -> None:
        """Initializes the batch runner.

        Args:
            config: Experiment configuration (architecture, agents, limits).
            config_path: Path to the experiment config file (for resolving
                prompts).
            wandb_project: Optional WandB project name.
            skip_env: Whether to skip Docker environment.
            disable_wandb: Whether to disable WandB logging.
            verbose: Whether to show detailed output per challenge.
            timeout_seconds: Optional command timeout for environment (seconds).
            console_handler: Optional console handler for same CLI output as
                single-challenge mode (used when verbose to show agent/tool
                output).
        """
        self._config = config
        self._config_path = config_path
        self._limits_config = config.limits
        self._wandb_project = wandb_project
        self._skip_env = skip_env
        self._disable_wandb = disable_wandb
        self._verbose = verbose
        self._timeout_seconds = timeout_seconds
        self._console_handler = console_handler
        self._console = Console()

    def run_batch(
        self,
        challenges_path: str,
        output_path: str | None = None,
    ) -> BatchResult:
        """Runs all challenges in a directory.

        Args:
            challenges_path: Path to challenges directory.
            output_path: Optional path to save results JSON.

        Returns:
            Aggregated batch results.
        """
        challenges = load_challenges(challenges_path, batch=True)

        batch_result = BatchResult(
            total_challenges=len(challenges),
            successful=0,
            failed=0,
            errors=0,
            start_time=datetime.now().isoformat(),
        )

        self._console.print(
            f"\n[bold]Running {len(challenges)} challenges...[/bold]\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Challenges",
                total=len(challenges),
            )

            for challenge in challenges:
                progress.update(
                    task,
                    description=f"[cyan]{challenge.name}",
                )

                result = self._run_single_challenge(challenge)
                batch_result.results.append(result)

                if result.error:
                    batch_result.errors += 1
                    status = "[red]ERROR[/red]"
                    if self._verbose:
                        self._console.print(
                            "  [red]{name}: ERROR - {error}[/red]".format(
                                name=challenge.name,
                                error=escape(result.error),
                            )
                        )
                elif result.success:
                    batch_result.successful += 1
                    status = "[green]SUCCESS[/green]"
                else:
                    batch_result.failed += 1
                    status = "[yellow]FAILED[/yellow]"

                # Show result if verbose
                if self._verbose:
                    self._console.print(
                        f"  {challenge.name}: {status} "
                        f"({result.turn_count} turns)"
                    )

                progress.advance(task)

        batch_result.end_time = datetime.now().isoformat()

        if output_path:
            batch_result.save(output_path)
            self._console.print(
                f"\n[dim]Results saved to {output_path}[/dim]"
            )

        self._print_summary(batch_result)
        return batch_result

    def _run_single_challenge(
        self,
        challenge: ChallengeData,
    ) -> ChallengeResult:
        """Runs a single challenge.

        Args:
            challenge: Challenge data.

        Returns:
            Challenge result.
        """
        try:
            if self._console_handler is not None:
                self._console_handler.on_challenge_start(
                    name=challenge.name,
                    category=challenge.category,
                    architecture=self._config.architecture,
                    max_turns=self._limits_config.max_turns,
                )

            with challenge_execution_context(
                config=self._config,
                challenge=challenge,
                skip_env=self._skip_env,
                timeout_seconds=self._timeout_seconds,
                console_handler=self._console_handler if self._verbose else None,
            ) as (_env_manager, workdir, tool_executor):
                objective = build_objective_from_challenge(
                    challenge=challenge,
                    workdir=workdir,
                    include_hints=True,
                )
                final_state = run_graph_for_objective(
                    config=self._config,
                    config_path=self._config_path,
                    tool_executor=tool_executor,
                    objective=objective,
                    console_handler=(
                        self._console_handler if self._verbose else None
                    ),
                )

            stats = compute_run_stats(
                final_state=final_state,
                primary_model=get_primary_model(self._config),
                max_cost_usd=self._limits_config.max_cost_usd,
                console_handler=self._console_handler,
            )

            if self._console_handler is not None:
                self._console_handler.on_run_end(
                    success=stats["success"],
                    termination_reason=stats["termination_reason"],
                    turn_count=stats["turn_count"],
                    total_tokens=stats["total_tokens"],
                    cost_usd=stats["cost_usd"],
                )

            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                challenge_name=challenge.name,
                success=stats["success"],
                termination_reason=stats["termination_reason"],
                turn_count=stats["turn_count"],
                total_tokens=stats["total_tokens"],
                cost_usd=stats["cost_usd"],
            )

        except Exception as err:
            return ChallengeResult(
                challenge_id=challenge.challenge_id,
                challenge_name=challenge.name,
                success=False,
                termination_reason="error",
                turn_count=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(err),
            )

    def _print_summary(self, batch_result: BatchResult) -> None:
        """Prints a summary of the batch run using Rich table."""
        self._console.print()

        table = Table(title="Batch Run Summary", show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total challenges", str(batch_result.total_challenges))
        table.add_row(
            "Successful",
            f"[green]{batch_result.successful}[/green]",
        )
        table.add_row(
            "Failed",
            f"[yellow]{batch_result.failed}[/yellow]",
        )
        table.add_row(
            "Errors",
            f"[red]{batch_result.errors}[/red]",
        )

        if batch_result.total_challenges > 0:
            rate = batch_result.successful / batch_result.total_challenges
            color = "green" if rate >= 0.7 else "yellow" if rate >= 0.4 else "red"
            table.add_row(
                "Success rate",
                f"[{color}]{rate:.1%}[/{color}]",
            )

        self._console.print(table)
