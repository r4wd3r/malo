"""Reasoner-Executor two-agent LangGraph implementation.

This architecture separates reasoning from execution:
- Reasoner: Analyzes the situation and outputs structured JSON commands
- Executor: Parses commands and executes tools

The key benefit is preventing "tool hallucination" - the Reasoner has no
direct tool access, forcing it to think before acting.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from MaLO.graphs.engine import run_with_timeout, run_llm_call, turn_limit_update

from langgraph.graph import END, StateGraph
from MaLO.config.models import BaseAgentConfig, LimitsConfig, PromptConfig
from MaLO.graphs.output_models import ReasonerOutput
from MaLO.graphs.state import MessageState
from MaLO.graphs.utils import (
    check_turn_limit,
    execute_tool_calls,
    format_system_prompt,
    parse_typed_json,
    should_terminate,
    ToolExecutionResult,
)
from MaLO.observability.console_handler import AgentRole, ConsoleHandler
from MaLO.tools.executor import ToolExecutor
from MaLO.tools.schemas import get_executor_tool_schemas
from MaLO.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ReasonerExecutorState(MessageState):
    """State for the reasoner-executor graph."""

    reasoner_command: dict[str, Any]  # Parsed JSON command from reasoner


def build_reasoner_executor_graph(
    reasoner_prompt: PromptConfig,
    executor_prompt: PromptConfig,
    reasoner_config: BaseAgentConfig,
    executor_config: BaseAgentConfig,
    limits_config: LimitsConfig,
    tool_executor: ToolExecutor,
    reasoner_client: LLMClient,
    executor_client: LLMClient,
    console_handler: ConsoleHandler | None = None,
) -> StateGraph:
    """Builds a reasoner-executor LangGraph workflow.

    Args:
        reasoner_prompt: Prompt configuration for the Reasoner.
        executor_prompt: Prompt configuration for the Executor.
        reasoner_config: Agent configuration for the Reasoner LLM.
        executor_config: Agent configuration for the Executor LLM.
        limits_config: Global limits configuration.
        tool_executor: Tool executor instance.
        reasoner_client: LLM client instance for the reasoner.
        executor_client: LLM client instance for the executor.
        console_handler: Optional console handler for CLI output.

    Returns:
        Compiled LangGraph workflow.
    """

    def _run_reasoner_turn(state: ReasonerExecutorState) -> dict[str, Any]:
        """Runs a single reasoner turn without wall-clock timeout.

        The Reasoner has NO tool access - it outputs structured JSON
        that the Executor will parse and execute.
        """
        turn = state.get("turn_count", 0) + 1
        max_turns = limits_config.max_turns

        system_prompt = format_system_prompt(
            reasoner_prompt.system_prompt,
            state.get("objective", ""),
        )

        # Reasoner does NOT get tool schemas - pure reasoning
        http_timeout = (
            float(limits_config.worker_timeout_seconds)
            if limits_config.worker_timeout_seconds is not None
            else None
        )
        result = run_llm_call(
            client=reasoner_client,
            model=reasoner_config.model,
            temperature=reasoner_config.temperature,
            system_prompt=system_prompt,
            messages=list(state.get("messages", [])),
            tools=None,
            timeout_seconds=http_timeout,
            seed=limits_config.random_seed,
            max_message_chars=limits_config.max_message_chars,
            console_handler=console_handler,
            agent_role=AgentRole.REASONER,
            turn=turn,
            max_turns=max_turns,
            token_update_mode="cumulative",
            state=state,
        )
        response = result["response"]

        # Parse JSON command from response
        parsed, _error = parse_typed_json(
            response.content or "", ReasonerOutput
        )
        if parsed is None:
            reasoner_command = {
                "thought": response.content or "",
                "command": {"type": "error", "error": "Failed to parse JSON"},
            }
        else:
            reasoner_command = parsed.model_dump()

        return {
            "messages": [result["assistant_message"]],
            "reasoner_command": reasoner_command,
            **result["token_updates"],
            "turn_count": turn,
        }

    def reasoner_node(state: ReasonerExecutorState) -> dict[str, Any]:
        """Reasoner node: analyzes state and outputs JSON command.

        Applies a per-turn wall-clock timeout when limits.worker_timeout_seconds
        is set to a positive value. On timeout, terminates with
        termination_reason \"llm_timeout\".
        """
        return run_with_timeout(
            fn=lambda: _run_reasoner_turn(state),
            timeout_seconds=limits_config.worker_timeout_seconds,
            timeout_result={
                "done": True,
                "termination_reason": "llm_timeout",
            },
            log_message=(
                f"Reasoner timed out after "
                f"{limits_config.worker_timeout_seconds}s. "
                "Terminating challenge."
            ),
            logger=logger,
        )

    def _run_executor_turn(state: ReasonerExecutorState) -> dict[str, Any]:
        """Executes a single executor turn without wall-clock timeout.

        The Executor calls an LLM with tool access to execute the Reasoner's
        command, then runs any tool calls produced.
        """
        turn = state.get("turn_count", 0)
        max_turns = limits_config.max_turns
        reasoner_command = state.get("reasoner_command", {})
        command_json = json.dumps(reasoner_command, indent=2)

        system_prompt = format_system_prompt(
            executor_prompt.system_prompt,
            state.get("objective", ""),
        )
        instruction = ""
        if executor_prompt.instruction_template:
            instruction = executor_prompt.instruction_template.replace(
                "{worker_report}", command_json
            )
        else:
            instruction = f"## Command from Reasoner\n\n```json\n{command_json}\n```"

        http_timeout = (
            float(limits_config.worker_timeout_seconds)
            if limits_config.worker_timeout_seconds is not None
            else None
        )
        llm_result = run_llm_call(
            client=executor_client,
            model=executor_config.model,
            temperature=executor_config.temperature,
            system_prompt=system_prompt,
            messages=list(state.get("messages", [])),
            tools=get_executor_tool_schemas(),
            timeout_seconds=http_timeout,
            seed=limits_config.random_seed,
            max_message_chars=limits_config.max_message_chars,
            console_handler=console_handler,
            agent_role=AgentRole.EXECUTOR,
            turn=turn,
            max_turns=max_turns,
            instruction=instruction,
            token_update_mode="cumulative",
            state=state,
            log_response=False,
        )
        response = llm_result["response"]

        tool_calls = []
        if response.tool_calls:
            for tc in response.tool_calls:
                name = getattr(tc.function, "name", "") if hasattr(tc, "function") else ""
                if name:
                    tool_calls.append(tc)

        tool_result = ToolExecutionResult(
            messages=[],
            calls_executed=0,
            done=False,
            termination_reason="",
            needs_human=False,
        )
        if tool_calls:
            tool_result = execute_tool_calls(
                tool_calls=tool_calls,
                tool_executor=tool_executor,
                truncation_chars=limits_config.observation_truncation_chars,
                console_handler=console_handler,
                agent_role=AgentRole.EXECUTOR,
                turn=turn,
                max_turns=max_turns,
                timeout_seconds=limits_config.worker_timeout_seconds,
            )
        else:
            # Fallback: execute the reasoner command directly if the executor
            # did not emit a valid tool call.
            command = reasoner_command.get("command", {})
            command_type = command.get("type", "")
            timeout_seconds = limits_config.worker_timeout_seconds
            calls_executed = 0

            if command_type in ("bash", "python"):
                script = command.get("script", "")
                result = tool_executor.execute(
                    "execute_script",
                    {"language": command_type, "script": script},
                    timeout_seconds=timeout_seconds,
                )
                calls_executed = 1
                tool_result = ToolExecutionResult(
                    messages=[
                        {
                            "role": "tool",
                            "tool_call_id": "executor_fallback",
                            "content": result.output,
                        }
                    ],
                    calls_executed=calls_executed,
                    done=result.terminate,
                    termination_reason=result.termination_reason or "",
                    needs_human=result.needs_human,
                )
            elif command_type == "submit":
                answer = command.get("answer", "")
                result = tool_executor.execute("submit_goal", {"answer": answer})
                calls_executed = 1
                tool_result = ToolExecutionResult(
                    messages=[
                        {
                            "role": "tool",
                            "tool_call_id": "executor_fallback",
                            "content": result.output,
                        }
                    ],
                    calls_executed=calls_executed,
                    done=result.terminate,
                    termination_reason=result.termination_reason or "",
                    needs_human=result.needs_human,
                )
            elif command_type == "give_up":
                reason = command.get("reason", "No reason provided")
                result = tool_executor.execute(
                    "give_up", {"reason": reason}, timeout_seconds=timeout_seconds
                )
                calls_executed = 1
                tool_result = ToolExecutionResult(
                    messages=[
                        {
                            "role": "tool",
                            "tool_call_id": "executor_fallback",
                            "content": result.output,
                        }
                    ],
                    calls_executed=calls_executed,
                    done=True,
                    termination_reason="give_up",
                    needs_human=result.needs_human,
                )
            else:
                tool_result = ToolExecutionResult(
                    messages=[
                        {
                            "role": "tool",
                            "tool_call_id": "executor_fallback",
                            "content": f"Unknown command type: {command_type}",
                        }
                    ],
                    calls_executed=0,
                    done=False,
                    termination_reason="",
                    needs_human=False,
                )

        if tool_result.termination_reason == "task_complete":
            tool_result = ToolExecutionResult(
                messages=tool_result.messages,
                calls_executed=tool_result.calls_executed,
                done=False,
                termination_reason="",
                needs_human=tool_result.needs_human,
            )

        tool_output = ""
        if tool_result.messages:
            tool_output = tool_result.messages[-1].get("content", "")

        new_tool_call_count = state.get("tool_call_count", 0) + tool_result.calls_executed

        if console_handler:
            console_handler.on_agent_response(
                _turn=turn,
                _max_turns=max_turns,
                agent=AgentRole.EXECUTOR,
                content=tool_output[:100] if tool_output else response.content,
                _has_tool_calls=bool(response.tool_calls),
            )

        return {
            "messages": [llm_result["assistant_message"]] + tool_result.messages,
            "tool_call_count": new_tool_call_count,
            "done": tool_result.done,
            "termination_reason": tool_result.termination_reason,
            "needs_human": tool_result.needs_human,
            **llm_result["token_updates"],
        }

    def executor_node(state: ReasonerExecutorState) -> dict[str, Any]:
        """Executor node: parses reasoner command and executes tools.

        Applies a per-turn wall-clock timeout when limits.worker_timeout_seconds
        is set to a positive value. On timeout, returns a synthetic execution
        result but does not terminate the overall workflow.
        """
        timeout_seconds = limits_config.worker_timeout_seconds

        def _timeout_result() -> dict[str, Any]:
            tool_output = (
                f"[EXECUTOR TIMED OUT after {timeout_seconds} seconds]"
            )
            result_message = {
                "role": "user",
                "content": f"## Execution Result\n\n{tool_output}",
            }
            if console_handler:
                console_handler.on_agent_response(
                    _turn=state.get("turn_count", 0),
                    _max_turns=limits_config.max_turns,
                    agent=AgentRole.EXECUTOR,
                    content=tool_output,
                    _has_tool_calls=False,
                )
            return {
                "messages": [result_message],
                "tool_call_count": state.get("tool_call_count", 0),
                "done": False,
                "termination_reason": "",
                "needs_human": False,
            }

        return run_with_timeout(
            fn=lambda: _run_executor_turn(state),
            timeout_seconds=timeout_seconds,
            timeout_result=_timeout_result,
            log_message=(
                f"Executor timed out after {timeout_seconds}s. "
                "Continuing with reasoner."
            ),
            logger=logger,
        )

    def end_turn_limit_node(_: ReasonerExecutorState) -> dict[str, Any]:
        """Sets termination_reason to turn_limit before ending."""
        return turn_limit_update()

    def should_continue(
        state: ReasonerExecutorState,
    ) -> Literal["executor", "end", "end_turn_limit"]:
        """Determines whether to continue to executor or end."""
        if should_terminate(state):
            return "end"

        if check_turn_limit(state, limits_config.max_turns):
            return "end_turn_limit"

        # Check if reasoner produced a valid command
        reasoner_command = state.get("reasoner_command", {})
        if not reasoner_command:
            return "end"

        return "executor"

    def after_executor(
        state: ReasonerExecutorState,
    ) -> Literal["reasoner", "end"]:
        """Determines whether to loop back to reasoner or end."""
        if should_terminate(state):
            return "end"
        return "reasoner"

    graph = StateGraph(ReasonerExecutorState)

    graph.add_node("reasoner", reasoner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("end_turn_limit", end_turn_limit_node)

    graph.set_entry_point("reasoner")

    graph.add_conditional_edges(
        "reasoner",
        should_continue,
        {
            "executor": "executor",
            "end": END,
            "end_turn_limit": "end_turn_limit",
        },
    )
    graph.add_edge("end_turn_limit", END)
    graph.add_conditional_edges(
        "executor",
        after_executor,
        {"reasoner": "reasoner", "end": END},
    )

    return graph.compile()
