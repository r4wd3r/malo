"""Graph builder registry for architecture selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from MaLO.config.models import BaseAgentConfig, ExperimentConfig
from MaLO.config.prompt_loader import load_prompt_config
from MaLO.graphs.orchestrator_critic_workers import (
    build_orchestrator_critic_workers_graph,
)
from MaLO.graphs.orchestrator_workers import build_orchestrator_workers_graph
from MaLO.graphs.reasoner_executor import build_reasoner_executor_graph
from MaLO.graphs.single_agent import build_single_agent_graph
from MaLO.tools.executor import ToolExecutor
from MaLO.utils.llm_client import LLMClient, build_llm_client


@dataclass
class GraphBuildContext:
    config: ExperimentConfig
    config_path: Path
    tool_executor: ToolExecutor
    console_handler: Any | None = None


@dataclass
class GraphBuilderSpec:
    required_agents: list[str]
    build: Callable[[GraphBuildContext], Any]


def _build_client(agent_config: BaseAgentConfig) -> LLMClient:
    return build_llm_client(
        provider=agent_config.provider,
        base_url=agent_config.base_url,
    )


def _validate_required_agents(config: ExperimentConfig, required: list[str]) -> None:
    for agent_name in required:
        if getattr(config.agents, agent_name, None) is None:
            raise ValueError(
                f"Missing required agent configuration: {agent_name!r}"
            )


def _resolve_prompt_path(config_path: Path, prompt_path: str) -> Path:
    candidate = Path(prompt_path)
    if candidate.is_absolute():
        return candidate
    return config_path.parent / candidate


def _build_single_react(ctx: GraphBuildContext) -> Any:
    config = ctx.config
    if config.agents.agent is not None:
        agent_config = config.agents.agent
    elif config.agents.worker is not None:
        agent_config = config.agents.worker
    else:
        raise ValueError("No agent configuration found (need 'agent' or 'worker').")

    prompt_path = _resolve_prompt_path(
        ctx.config_path, agent_config.prompts_file
    )
    prompt_config = load_prompt_config(prompt_path)
    client = _build_client(agent_config)

    return build_single_agent_graph(
        prompt_config=prompt_config,
        agent_config=agent_config,
        limits_config=config.limits,
        tool_executor=ctx.tool_executor,
        client=client,
        console_handler=ctx.console_handler,
    )


def _build_reasoner_executor(ctx: GraphBuildContext) -> Any:
    config = ctx.config
    _validate_required_agents(config, ["orchestrator", "worker"])

    reasoner_config = config.agents.orchestrator
    executor_config = config.agents.worker
    assert reasoner_config is not None and executor_config is not None

    reasoner_prompt_path = _resolve_prompt_path(
        ctx.config_path, reasoner_config.prompts_file
    )
    executor_prompt_path = _resolve_prompt_path(
        ctx.config_path, executor_config.prompts_file
    )

    reasoner_prompt = load_prompt_config(reasoner_prompt_path)
    executor_prompt = load_prompt_config(executor_prompt_path)

    reasoner_client = _build_client(reasoner_config)
    executor_client = _build_client(executor_config)

    return build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_config,
        executor_config=executor_config,
        limits_config=config.limits,
        tool_executor=ctx.tool_executor,
        reasoner_client=reasoner_client,
        executor_client=executor_client,
        console_handler=ctx.console_handler,
    )


def _build_orchestrator_workers(ctx: GraphBuildContext) -> Any:
    config = ctx.config
    _validate_required_agents(config, ["orchestrator", "worker"])

    orchestrator_config = config.agents.orchestrator
    worker_config = config.agents.worker
    assert orchestrator_config is not None and worker_config is not None

    orchestrator_prompt_path = _resolve_prompt_path(
        ctx.config_path, orchestrator_config.prompts_file
    )
    worker_prompt_path = _resolve_prompt_path(
        ctx.config_path, worker_config.prompts_file
    )

    orchestrator_prompt = load_prompt_config(orchestrator_prompt_path)
    worker_prompt = load_prompt_config(worker_prompt_path)

    orchestrator_client = _build_client(orchestrator_config)
    worker_client = _build_client(worker_config)

    return build_orchestrator_workers_graph(
        orchestrator_prompt=orchestrator_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orchestrator_config,
        worker_config=worker_config,
        limits_config=config.limits,
        tool_executor=ctx.tool_executor,
        orchestrator_client=orchestrator_client,
        worker_client=worker_client,
        console_handler=ctx.console_handler,
    )


def _build_orchestrator_critic_workers(ctx: GraphBuildContext) -> Any:
    config = ctx.config
    _validate_required_agents(config, ["orchestrator", "critic", "worker"])

    orchestrator_config = config.agents.orchestrator
    critic_config = config.agents.critic
    worker_config = config.agents.worker
    assert (
        orchestrator_config is not None
        and critic_config is not None
        and worker_config is not None
    )

    orchestrator_prompt_path = _resolve_prompt_path(
        ctx.config_path, orchestrator_config.prompts_file
    )
    critic_prompt_path = _resolve_prompt_path(
        ctx.config_path, critic_config.prompts_file
    )
    worker_prompt_path = _resolve_prompt_path(
        ctx.config_path, worker_config.prompts_file
    )

    orchestrator_prompt = load_prompt_config(orchestrator_prompt_path)
    critic_prompt = load_prompt_config(critic_prompt_path)
    worker_prompt = load_prompt_config(worker_prompt_path)

    orchestrator_client = _build_client(orchestrator_config)
    critic_client = _build_client(critic_config)
    worker_client = _build_client(worker_config)

    return build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orchestrator_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orchestrator_config,
        critic_config=critic_config,
        worker_config=worker_config,
        limits_config=config.limits,
        tool_executor=ctx.tool_executor,
        orchestrator_client=orchestrator_client,
        critic_client=critic_client,
        worker_client=worker_client,
        console_handler=ctx.console_handler,
    )


GRAPH_BUILDERS: dict[str, GraphBuilderSpec] = {
    "single_react": GraphBuilderSpec(
        required_agents=["agent"],
        build=_build_single_react,
    ),
    "reasoner_executor": GraphBuilderSpec(
        required_agents=["orchestrator", "worker"],
        build=_build_reasoner_executor,
    ),
    "orchestrator_workers": GraphBuilderSpec(
        required_agents=["orchestrator", "worker"],
        build=_build_orchestrator_workers,
    ),
    "orchestrator_critic_workers": GraphBuilderSpec(
        required_agents=["orchestrator", "critic", "worker"],
        build=_build_orchestrator_critic_workers,
    ),
}


def build_graph_from_registry(ctx: GraphBuildContext) -> Any:
    spec = GRAPH_BUILDERS.get(ctx.config.architecture)
    if spec is None:
        raise ValueError(f"Unsupported architecture: {ctx.config.architecture}")
    _validate_required_agents(ctx.config, spec.required_agents)
    return spec.build(ctx)
