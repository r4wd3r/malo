"""Shared helpers for building graphs and initial state from experiment config.

Used by both CLI (single-challenge and legacy modes) and BatchRunner so that
batch mode runs the same architecture as the experiment config.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import contextvars
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager

logger = logging.getLogger(__name__)

from MaLO.challenges.models import ChallengeData, Goal
from MaLO.config.models import EnvironmentConfig, ExperimentConfig
from MaLO.environment.challenge_container import (
    ensure_network,
    needs_challenge_container,
    start_challenge_container,
)
from MaLO.environment.constants import DEFAULT_DOCKER_NETWORK
from MaLO.environment.environment_manager import EnvironmentManager
from MaLO.graphs.registry import GraphBuildContext, build_graph_from_registry
from MaLO.observability.console_handler import ConsoleHandler
from MaLO.observability.metrics import estimate_cost
from MaLO.tools.executor import ToolExecutor
from MaLO.utils.llm_client import LLMClient, build_llm_client


def resolve_prompt_path(config_path: Path, prompt_path: str) -> Path:
    """Resolves a prompt path relative to the config location.

    Args:
        config_path: Path to the experiment config.
        prompt_path: Prompt path from the configuration.

    Returns:
        Absolute path to the prompt file.
    """
    candidate = Path(prompt_path)
    if candidate.is_absolute():
        return candidate
    return config_path.parent / candidate


def get_llm_client() -> LLMClient:
    """Builds a default LLM client from environment configuration."""
    return build_llm_client()


def prepare_challenge_environment(
    config: ExperimentConfig,
    challenge: ChallengeData,
    skip_env: bool,
    timeout_seconds: int | None = None,
    console_handler: ConsoleHandler | None = None,
) -> tuple[EnvironmentManager | None, str, bool]:
    """Prepares the environment for a challenge.

    Returns (env_manager, workdir, challenge_container_started).
    """
    workdir = "/ctf"
    challenge_container_started = False
    if skip_env:
        return None, workdir, challenge_container_started

    challenge_path_resolved = str(Path(challenge.path).resolve())
    network = config.environment.docker_network or DEFAULT_DOCKER_NETWORK

    if needs_challenge_container(challenge):
        ensure_network(network)
        hostname, port = start_challenge_container(challenge, network)
        challenge.service_hostname = hostname
        challenge.service_port = port
        challenge_container_started = True
        env_config = EnvironmentConfig(
            docker_image=config.environment.docker_image,
            setup_commands=config.environment.setup_commands,
            timeout_seconds=timeout_seconds
            or config.environment.timeout_seconds,
            docker_network=network,
        )
        env_manager = EnvironmentManager(env_config)
        env_manager.setup()
        workdir = env_manager.workdir
        files_to_copy = [
            f for f in (challenge.files or []) if "flag" not in f.lower()
        ]
        if files_to_copy:
            env_manager.copy_files(challenge_path_resolved, files_to_copy)
        if console_handler is not None:
            console_handler.print_info(
                f"Challenge files at: {workdir}; service at {hostname}:{port}"
            )
        return env_manager, workdir, challenge_container_started

    env_config = EnvironmentConfig(
        docker_image=config.environment.docker_image,
        setup_commands=config.environment.setup_commands,
        timeout_seconds=timeout_seconds or config.environment.timeout_seconds,
    )
    env_manager = EnvironmentManager(env_config)
    env_manager.setup()
    workdir = env_manager.workdir
    if challenge.files:
        env_manager.copy_files(challenge_path_resolved, challenge.files)
    if console_handler is not None:
        console_handler.print_info(f"Challenge files available at: {workdir}")
    return env_manager, workdir, challenge_container_started


def build_objective_from_challenge(
    challenge: ChallengeData,
    workdir: str = "/ctf",
    include_hints: bool = True,
) -> str:
    """Builds the objective string for a challenge."""
    parts = [
        f"Challenge: {challenge.name}",
        f"Category: {challenge.category}",
        f"Description: {challenge.description}",
    ]
    if include_hints and challenge.hints:
        parts.append(f"Hints: {', '.join(challenge.hints)}")
    parts.append(f"Goal: {challenge.goal.description}")

    if challenge.files:
        files_list = "\n".join(f"  - {f}" for f in challenge.files)
        parts.append(f"\nChallenge files (in {workdir}):\n{files_list}")

    if challenge.service_hostname and challenge.service_port is not None:
        parts.append(
            f"\nThe challenge service is running at "
            f"{challenge.service_hostname}:{challenge.service_port}. "
            f"Use this endpoint to interact with the challenge "
            f"(e.g. connect with nc or pwntools)."
        )

    return "\n".join(parts)


def build_tool_executor_for_goal(
    env_manager: EnvironmentManager | None,
    truncation_chars: int,
    goal: Goal | None,
) -> ToolExecutor:
    """Builds a tool executor with goal validation configured."""
    from MaLO.validation.goal_validator import DefaultGoalValidator

    goal_validator = DefaultGoalValidator(goal) if goal is not None else None
    return ToolExecutor(
        env_manager=env_manager,
        truncation_chars=truncation_chars,
        goal_validator=goal_validator,
    )


@contextmanager
def challenge_execution_context(
    *,
    config: ExperimentConfig,
    challenge: ChallengeData,
    skip_env: bool,
    timeout_seconds: int | None = None,
    console_handler: ConsoleHandler | None = None,
) -> tuple[EnvironmentManager | None, str, ToolExecutor]:
    """Context manager for challenge environment and tool executor lifecycle."""
    env_manager: EnvironmentManager | None = None
    workdir = "/ctf"
    challenge_container_started = False
    try:
        env_manager, workdir, challenge_container_started = (
            prepare_challenge_environment(
                config=config,
                challenge=challenge,
                skip_env=skip_env,
                timeout_seconds=timeout_seconds,
                console_handler=console_handler,
            )
        )
        tool_executor = build_tool_executor_for_goal(
            env_manager=env_manager,
            truncation_chars=config.limits.observation_truncation_chars,
            goal=challenge.goal,
        )
        yield env_manager, workdir, tool_executor
    finally:
        if challenge_container_started:
            try:
                from MaLO.environment.challenge_container import (
                    stop_challenge_container,
                )

                stop_challenge_container(challenge)
            except Exception as err:
                logger.warning("Failed to stop challenge container: %s", err)
        if env_manager is not None:
            try:
                env_manager.teardown()
            except Exception as err:
                logger.warning("Failed to teardown environment: %s", err)


def run_graph_for_objective(
    config: ExperimentConfig,
    config_path: Path,
    tool_executor: ToolExecutor,
    objective: str,
    console_handler: ConsoleHandler | None = None,
) -> dict[str, Any]:
    """Builds and runs the graph for a given objective."""
    graph = build_graph_for_architecture(
        config=config,
        config_path=config_path,
        tool_executor=tool_executor,
        console_handler=console_handler,
    )
    initial_state = get_initial_state_for_architecture(
        config.architecture, objective
    )
    return invoke_graph_with_timeout(
        graph=graph,
        initial_state=initial_state,
        recursion_limit=config.limits.recursion_limit,
        timeout_seconds=config.limits.challenge_timeout_seconds,
    )


def compute_run_stats(
    final_state: dict[str, Any],
    primary_model: str,
    max_cost_usd: float,
    console_handler: ConsoleHandler | None = None,
) -> dict[str, Any]:
    """Computes summary stats for a run and warns on cost overages."""
    prompt_tokens = final_state.get("prompt_tokens", 0)
    completion_tokens = final_state.get("completion_tokens", 0)
    total_tokens = final_state.get("total_tokens", 0)
    turn_count = final_state.get("turn_count", 0)
    termination_reason = final_state.get("termination_reason", "")
    success = termination_reason == "success"
    cost_usd = estimate_cost(primary_model, prompt_tokens, completion_tokens)
    if cost_usd > max_cost_usd and console_handler is not None:
        console_handler.print_info(
            "WARNING: Estimated cost "
            f"${cost_usd:.3f} exceeds max_cost_usd "
            f"${max_cost_usd:.3f}."
        )
    return {
        "turn_count": turn_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "termination_reason": termination_reason,
        "success": success,
        "cost_usd": cost_usd,
    }


def build_graph_for_architecture(
    config: ExperimentConfig,
    config_path: Path,
    tool_executor: ToolExecutor,
    console_handler: ConsoleHandler | None = None,
) -> Any:
    """Builds the appropriate graph based on architecture."""
    ctx = GraphBuildContext(
        config=config,
        config_path=config_path,
        tool_executor=tool_executor,
        console_handler=console_handler,
    )
    return build_graph_from_registry(ctx)


def get_initial_state_for_architecture(
    architecture: str,
    objective: str,
) -> dict[str, Any]:
    """Creates the initial state for the given architecture.

    Args:
        architecture: Architecture type.
        objective: Challenge objective.

    Returns:
        Initial state dictionary.
    """
    base_state = {
        "objective": objective,
        "turn_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tool_call_count": 0,
        "done": False,
        "termination_reason": "",
        "needs_human": False,
    }

    if architecture == "single_react":
        return {**base_state, "messages": []}

    elif architecture == "reasoner_executor":
        return {
            **base_state,
            "messages": [],
            "reasoner_command": {},
        }

    elif architecture == "orchestrator_workers":
        return {
            **base_state,
            "orchestrator_state": {},
            "orchestrator_messages": [],
            "worker_report": "",
            "current_tasks": [],
            "worker_reports": [],
            "last_submit_feedback": "",
        }

    elif architecture == "orchestrator_critic_workers":
        return {
            **base_state,
            "orchestrator_state": {},
            "orchestrator_messages": [],
            "proposed_plan": {},
            "critic_verdict": "",
            "critic_feedback": "",
            "rejection_count": 0,
            "current_tasks": [],
            "worker_reports": [],
            "worker_report": "",
            "last_submit_feedback": "",
        }

    else:
        return {**base_state, "messages": []}


def get_primary_model(config: ExperimentConfig) -> str:
    """Gets the primary model for cost estimation.

    Args:
        config: Experiment configuration.

    Returns:
        Model identifier string.
    """
    if config.agents.agent:
        return config.agents.agent.model
    if config.agents.orchestrator:
        return config.agents.orchestrator.model
    if config.agents.worker:
        return config.agents.worker.model
    return "unknown"


def invoke_graph_with_timeout(
    graph: Any,
    initial_state: dict[str, Any],
    recursion_limit: int,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    """Invokes a graph with an optional overall timeout.

    Args:
        graph: Compiled LangGraph workflow.
        initial_state: Initial state dictionary.
        recursion_limit: Recursion limit for graph execution.
        timeout_seconds: Optional timeout in seconds. If None, no timeout.

    Returns:
        Final state dictionary. On timeout, returns state with
        termination_reason set to "challenge_timeout" and done set to True.
    """
    if timeout_seconds is None:
        return graph.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
        )

    ctx = contextvars.copy_context()

    def _run() -> dict[str, Any]:
        return ctx.run(
            graph.invoke,
            initial_state,
            {"recursion_limit": recursion_limit},
        )

    executor = ThreadPoolExecutor(max_workers=1)
    timed_out = False
    try:
        future = executor.submit(_run)
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        timed_out = True
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.warning(
            f"Challenge timeout exceeded ({timeout_seconds}s). "
            "Terminating workflow."
        )
        state = dict(initial_state)
        state.update(
            {
                "done": True,
                "termination_reason": "challenge_timeout",
            }
        )
        return state
    finally:
        if not timed_out:
            executor.shutdown(wait=True)
