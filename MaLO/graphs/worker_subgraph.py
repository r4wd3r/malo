"""Shared worker subgraph builder for orchestrator workflows."""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Sequence

from langgraph.graph import END, StateGraph

from MaLO.config.models import BaseAgentConfig, LimitsConfig, PromptConfig
from MaLO.graphs.engine import ensure_system_message, execute_tools_update, run_llm_call, run_with_timeout
from MaLO.graphs.utils import format_system_prompt, get_message_attr, get_tool_calls
from MaLO.observability.console_handler import AgentRole, ConsoleHandler
from MaLO.tools.executor import ToolExecutor
from MaLO.tools.schemas import get_worker_tool_schemas
from MaLO.utils.llm_client import LLMClient
from MaLO.utils.text_utils import truncate_text


def _get_task_desc(task: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = task.get(key)
        if value:
            return str(value)
    return ""


def build_worker_subgraph(
    *,
    worker_state_type: Any,
    worker_prompt: PromptConfig,
    worker_config: BaseAgentConfig,
    limits_config: LimitsConfig,
    tool_executor: ToolExecutor,
    worker_client: LLMClient,
    console_handler: ConsoleHandler | None = None,
    task_desc_keys: Sequence[str] = ("description",),
    default_instruction_uses_task_content: bool = False,
    suppress_tool_termination: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[Any, Callable[[Any], dict[str, Any]]]:
    """Builds the worker subgraph and returns (compiled_worker, wrapper_node)."""
    tools = get_worker_tool_schemas()
    worker_max_turns = limits_config.worker_max_turns
    report_truncation = limits_config.observation_truncation_chars
    node_logger = logger or logging.getLogger(__name__)

    def _build_instruction(task: dict[str, Any]) -> str:
        task_desc = _get_task_desc(task, task_desc_keys)
        task_details = task.get("details", "")
        task_content = f"**Task:** {task_desc}\n\n**Details:** {task_details}"
        if worker_prompt.instruction_template:
            return worker_prompt.instruction_template.replace(
                "{assigned_task}", task_content
            )
        if default_instruction_uses_task_content:
            return f"## Assigned Task\n\n{task_content}"
        return f"## Assigned Task\n\n{task_desc}\n\n{task_details}"

    def worker_node(state: Any) -> dict[str, Any]:
        """Worker node: executes task using tools with dynamic role."""
        worker_turn = state.get("worker_turn_count", 0) + 1
        worker_role = state.get("worker_role", "Security Specialist")
        worker_id = state.get("worker_id")

        task = state.get("task", {})
        system_prompt = format_system_prompt(
            worker_prompt.system_prompt,
            state.get("objective", ""),
        )
        system_prompt = system_prompt.replace("{worker_role}", worker_role)

        instruction = _build_instruction(task)

        existing_messages = list(state.get("worker_messages", []))
        had_system = bool(
            existing_messages
            and get_message_attr(existing_messages[0], "role") == "system"
        )
        messages = ensure_system_message(existing_messages, system_prompt)
        if not had_system:
            messages.append({"role": "user", "content": instruction})

        timeout = (
            float(limits_config.worker_timeout_seconds)
            if limits_config.worker_timeout_seconds is not None
            else None
        )
        llm_result = run_llm_call(
            client=worker_client,
            model=worker_config.model,
            temperature=worker_config.temperature,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            timeout_seconds=timeout,
            seed=limits_config.random_seed,
            max_message_chars=limits_config.max_message_chars,
            console_handler=console_handler,
            agent_role=AgentRole.WORKER,
            turn=worker_turn,
            max_turns=worker_max_turns,
            worker_id=worker_id,
            token_update_mode="cumulative",
            state=state,
            log_response=False,
        )
        response = llm_result["response"]

        if console_handler:
            console_handler.on_agent_response(
                _turn=worker_turn,
                _max_turns=worker_max_turns,
                agent=AgentRole.WORKER,
                content=f"[{worker_role}] {response.content or ''}",
                _has_tool_calls=bool(response.tool_calls),
                _worker_id=worker_id,
            )

        return {
            "worker_messages": messages + [llm_result["assistant_message"]],
            "worker_turn_count": worker_turn,
            **llm_result["token_updates"],
        }

    def worker_tools_node(state: Any) -> dict[str, Any]:
        """Executes tools for a worker."""
        messages = state.get("worker_messages", [])
        if not messages:
            return {}

        last_message = messages[-1]
        tool_calls = get_tool_calls(last_message)
        if not tool_calls:
            return {}

        worker_turn = state.get("worker_turn_count", 0)
        worker_id = state.get("worker_id")

        tool_update = execute_tools_update(
            state=state,
            tool_calls=tool_calls,
            tool_executor=tool_executor,
            truncation_chars=report_truncation,
            console_handler=console_handler,
            agent_role=AgentRole.WORKER,
            turn=worker_turn,
            max_turns=worker_max_turns,
            worker_id=worker_id,
            timeout_seconds=limits_config.worker_timeout_seconds,
        )

        updated_messages = messages + tool_update.get("messages", [])

        if worker_turn >= 2 and not tool_update.get("done"):
            updated_messages = updated_messages + [
                {
                    "role": "user",
                    "content": (
                        "If the task is complete, call task_complete(findings) "
                        "now with your summary. Otherwise continue with another "
                        "tool call."
                    ),
                }
            ]

        done = tool_update.get("done", False)
        termination_reason = tool_update.get("termination_reason", "")
        if suppress_tool_termination:
            is_give_up = termination_reason == "give_up"
            is_task_complete = termination_reason == "task_complete"
            should_continue = is_give_up or is_task_complete
            done = done and not should_continue
            termination_reason = "" if should_continue else termination_reason

        return {
            "worker_messages": updated_messages,
            "tool_call_count": tool_update.get(
                "tool_call_count", state.get("tool_call_count", 0)
            ),
            "done": done,
            "termination_reason": termination_reason,
            "needs_human": tool_update.get("needs_human", False),
        }

    def compile_single_worker_report(state: Any) -> dict[str, Any]:
        """Compiles a single worker's results into a report."""
        messages = state.get("worker_messages", [])
        task = state.get("task", {})
        worker_role = state.get("worker_role", "Worker")
        worker_id = state.get("worker_id")
        task_title = task.get("description", "Task")

        if worker_id is not None:
            heading = f"### Worker #{worker_id} ({worker_role}) - {task_title}"
        else:
            heading = f"### {worker_role} - {task_title}"

        report_parts = [heading]
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant" and content:
                report_parts.append(
                    f"**Response:** {truncate_text(content, report_truncation)}"
                )
            elif role == "tool" and content:
                report_parts.append(
                    f"**Tool Output:** {truncate_text(content, report_truncation)}"
                )

        worker_report = "\n\n".join(report_parts[-5:])
        return {"worker_report": worker_report}

    def route_worker_internal(state: Any) -> Literal["worker_tools", "report", "__end__"]:
        if state.get("done", False):
            return "report"
        messages = state.get("worker_messages", [])
        if not messages:
            return "report"
        if get_tool_calls(messages[-1]):
            return "worker_tools"
        return "report"

    def route_after_worker_tools_internal(
        state: Any,
    ) -> Literal["worker", "report", "__end__"]:
        if state.get("done", False):
            return "report"
        if state.get("worker_turn_count", 0) >= worker_max_turns:
            return "report"
        return "worker"

    worker_subgraph = StateGraph(worker_state_type)
    worker_subgraph.add_node("worker", worker_node)
    worker_subgraph.add_node("worker_tools", worker_tools_node)
    worker_subgraph.add_node("report", compile_single_worker_report)
    worker_subgraph.set_entry_point("worker")
    worker_subgraph.add_conditional_edges(
        "worker",
        route_worker_internal,
        {"worker_tools": "worker_tools", "report": "report", "__end__": END},
    )
    worker_subgraph.add_conditional_edges(
        "worker_tools",
        route_after_worker_tools_internal,
        {"worker": "worker", "report": "report", "__end__": END},
    )
    worker_subgraph.add_edge("report", END)

    compiled_worker = worker_subgraph.compile()

    def worker_subgraph_node(state: Any) -> dict[str, Any]:
        """Runs worker subgraph with wall-clock timeout and aggregates report."""
        timeout_seconds = limits_config.worker_timeout_seconds
        assert timeout_seconds is not None and timeout_seconds > 0

        def _timeout_result() -> dict[str, Any]:
            task = state.get("task", {})
            worker_role = state.get("worker_role", "Worker")
            task_desc = task.get("description", "Task")
            node_logger.warning(
                f"Worker '{worker_role}' timed out after "
                f"{timeout_seconds}s on task: {task_desc}. "
                "Continuing with orchestrator."
            )
            report = f"### {worker_role} - {task_desc}\n\n**[WORKER TIMED OUT]**"
            return {
                "worker_reports": [report],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "tool_call_count": 0,
                "done": False,
                "termination_reason": "",
            }

        result = run_with_timeout(
            fn=lambda: compiled_worker.invoke(state),
            timeout_seconds=timeout_seconds,
            timeout_result=_timeout_result,
            log_message="",
            logger=node_logger,
        )

        if "worker_reports" in result and "worker_report" not in result:
            return result

        termination_reason = result.get("termination_reason", "")
        is_give_up = termination_reason == "give_up"
        is_task_complete = termination_reason == "task_complete"
        is_tool_timeout = termination_reason == "tool_timeout"
        should_continue = is_give_up or is_task_complete or is_tool_timeout

        report = result.get("worker_report", "")
        if is_give_up:
            report = f"{report}\n\n**[WORKER GAVE UP]**"
        elif is_task_complete:
            report = f"{report}\n\n**[TASK COMPLETED]**"

        return {
            "worker_reports": [report],
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "tool_call_count": result.get("tool_call_count", 0),
            "done": result.get("done", False) and not should_continue,
            "termination_reason": "" if should_continue else termination_reason,
            "needs_human": result.get("needs_human", False),
        }

    return compiled_worker, worker_subgraph_node
