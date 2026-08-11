"""Single-agent ReAct LangGraph implementation."""

from __future__ import annotations

import logging
from typing import Any, Literal

from MaLO.graphs.engine import (
    execute_tools_update,
    run_with_timeout,
    run_llm_call,
    turn_limit_update,
)

logger = logging.getLogger(__name__)

from langgraph.graph import END, StateGraph

from MaLO.config.models import BaseAgentConfig, LimitsConfig, PromptConfig
from MaLO.graphs.state import MessageState
from MaLO.graphs.utils import (
    check_turn_limit,
    format_system_prompt,
    get_tool_calls,
    should_terminate,
)
from MaLO.observability.console_handler import AgentRole, ConsoleHandler
from MaLO.tools.executor import ToolExecutor
from MaLO.tools.schemas import get_agent_tool_schemas
from MaLO.utils.llm_client import LLMClient


class AgentState(MessageState):
    """State for the single-agent ReAct graph."""

    pass


def build_single_agent_graph(
    prompt_config: PromptConfig,
    agent_config: BaseAgentConfig,
    limits_config: LimitsConfig,
    tool_executor: ToolExecutor,
    client: LLMClient,
    console_handler: ConsoleHandler | None = None,
) -> StateGraph:
    """Builds a single-agent ReAct LangGraph workflow.

    Args:
        prompt_config: Prompt configuration.
        agent_config: Agent configuration for the LLM.
        limits_config: Global limits configuration.
        tool_executor: Tool executor instance.
        client: LLM client instance.
        console_handler: Optional console handler for CLI output.

    Returns:
        Compiled LangGraph workflow.
    """
    tools = get_agent_tool_schemas()

    def _run_agent_turn(state: AgentState) -> dict[str, Any]:
        """Runs a single agent turn without wall-clock timeout."""
        turn = state.get("turn_count", 0) + 1
        max_turns = limits_config.max_turns

        system_prompt = format_system_prompt(
            prompt_config.system_prompt,
            state.get("objective", ""),
        )

        timeout = (
            float(limits_config.worker_timeout_seconds)
            if limits_config.worker_timeout_seconds is not None
            else None
        )
        result = run_llm_call(
            client=client,
            model=agent_config.model,
            temperature=agent_config.temperature,
            system_prompt=system_prompt,
            messages=list(state.get("messages", [])),
            tools=tools,
            timeout_seconds=timeout,
            seed=limits_config.random_seed,
            max_message_chars=limits_config.max_message_chars,
            console_handler=console_handler,
            agent_role=AgentRole.AGENT,
            turn=turn,
            max_turns=max_turns,
            token_update_mode="cumulative",
            state=state,
        )

        return {
            "messages": [result["assistant_message"]],
            **result["token_updates"],
            "turn_count": turn,
        }

    def agent_node(state: AgentState) -> dict[str, Any]:
        """Agent node: generates the next action or response.

        Applies a per-turn wall-clock timeout when limits.worker_timeout_seconds
        is set to a positive value. On timeout, the run is terminated with
        termination_reason set to \"llm_timeout\".
        """
        return run_with_timeout(
            fn=lambda: _run_agent_turn(state),
            timeout_seconds=limits_config.worker_timeout_seconds,
            timeout_result={
                "done": True,
                "termination_reason": "llm_timeout",
            },
            log_message=(
                f"Agent turn timed out after "
                f"{limits_config.worker_timeout_seconds}s. "
                "Terminating challenge."
            ),
            logger=logger,
        )

    def _run_tool_turn(state: AgentState) -> dict[str, Any]:
        """Runs tool execution without wall-clock timeout."""
        messages = state.get("messages", [])
        if not messages:
            return {"done": True, "termination_reason": "no_messages"}

        last_message = messages[-1]
        tool_calls = get_tool_calls(last_message)
        if not tool_calls:
            return {}

        turn = state.get("turn_count", 0)
        max_turns = limits_config.max_turns

        return execute_tools_update(
            state=state,
            tool_calls=tool_calls,
            tool_executor=tool_executor,
            truncation_chars=limits_config.observation_truncation_chars,
            console_handler=console_handler,
            agent_role=AgentRole.AGENT,
            turn=turn,
            max_turns=max_turns,
            timeout_seconds=limits_config.worker_timeout_seconds,
        )

    def tool_node(state: AgentState) -> dict[str, Any]:
        """Tool node: executes tool calls and returns results.

        Applies a per-turn wall-clock timeout when limits.worker_timeout_seconds
        is set to a positive value. On timeout, the run is terminated with
        termination_reason set to \"tool_timeout\".
        """
        return run_with_timeout(
            fn=lambda: _run_tool_turn(state),
            timeout_seconds=limits_config.worker_timeout_seconds,
            timeout_result={"done": True, "termination_reason": "tool_timeout"},
            log_message=(
                f"Tool execution timed out after "
                f"{limits_config.worker_timeout_seconds}s. "
                "Terminating challenge."
            ),
            logger=logger,
        )

    def end_turn_limit_node(_: AgentState) -> dict[str, Any]:
        """Sets termination_reason to turn_limit before ending."""
        return turn_limit_update()

    def should_continue(state: AgentState) -> Literal["tools", "end", "end_turn_limit"]:
        """Determines whether to continue to tools or end."""
        if should_terminate(state):
            return "end"

        if check_turn_limit(state, limits_config.max_turns):
            return "end_turn_limit"

        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if get_tool_calls(last_message):
            return "tools"

        return "end"

    def after_tools(state: AgentState) -> Literal["agent", "end"]:
        """Determines whether to loop back to agent or end."""
        if should_terminate(state):
            return "end"
        return "agent"

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("end_turn_limit", end_turn_limit_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END, "end_turn_limit": "end_turn_limit"},
    )
    graph.add_edge("end_turn_limit", END)
    graph.add_conditional_edges(
        "tools",
        after_tools,
        {"agent": "agent", "end": END},
    )

    return graph.compile()
