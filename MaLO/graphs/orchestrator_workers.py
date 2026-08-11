"""Orchestrator-Workers hierarchical LangGraph implementation.

This architecture implements a Map-Reduce style workflow:
- Orchestrator: Plans tasks using structured JSON state (not chat history)
- Workers: Execute tasks in parallel mini ReAct loops and report back

The key design is that the Orchestrator maintains a JSON state object
instead of relying on chat history, enabling better strategic planning.
Workers can be dynamically assigned roles by the Orchestrator.
"""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, Literal

logger = logging.getLogger(__name__)

from langgraph.constants import Send
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from MaLO.config.models import BaseAgentConfig, LimitsConfig, PromptConfig
from MaLO.graphs.output_models import OrchestratorOutput
from MaLO.graphs.engine import (
    ensure_system_message,
    execute_tools_update,
    run_with_timeout,
    run_llm_call,
    turn_limit_update,
)
from MaLO.graphs.orchestrator_utils import (
    extract_tasks as _extract_tasks,
    format_state_json as _format_state_json,
    merge_state as _merge_state,
    reduce_worker_reports as _reduce_worker_reports,
)
from MaLO.graphs.state import OrchestratorState as OrchestratorStateBase
from MaLO.graphs.state import WorkerState as WorkerStateBase
from MaLO.graphs.worker_subgraph import build_worker_subgraph
from MaLO.graphs.utils import (
    check_turn_limit,
    format_system_prompt,
    parse_typed_json,
    should_terminate,
)
from MaLO.observability.console_handler import AgentRole, ConsoleHandler
from MaLO.tools.executor import ToolExecutor
from MaLO.tools.schemas import get_orchestrator_tool_schemas
from MaLO.utils.llm_client import LLMClient


class OrchestratorWorkersState(OrchestratorStateBase):
    """State for the orchestrator-workers graph."""

    # Orchestrator uses JSON state, not chat history
    orchestrator_state: dict[str, Any]
    orchestrator_messages: Annotated[list[dict[str, Any]], add_messages]
    # Multiple tasks for parallel workers
    current_tasks: list[dict[str, Any]]
    # Aggregated reports from parallel workers (reducer collects them)
    worker_reports: Annotated[list[str], _reduce_worker_reports]
    # Combined report for orchestrator consumption
    worker_report: str
    # Feedback from last failed submit (so orchestrator can retry)
    last_submit_feedback: str


class WorkerState(WorkerStateBase):
    """State for a single worker branch."""


def build_orchestrator_workers_graph(
    orchestrator_prompt: PromptConfig,
    worker_prompt: PromptConfig,
    orchestrator_config: BaseAgentConfig,
    worker_config: BaseAgentConfig,
    limits_config: LimitsConfig,
    tool_executor: ToolExecutor,
    orchestrator_client: LLMClient,
    worker_client: LLMClient,
    console_handler: ConsoleHandler | None = None,
) -> StateGraph:
    """Builds an orchestrator-workers LangGraph workflow with parallel workers.

    Args:
        orchestrator_prompt: Prompt configuration for the Orchestrator.
        worker_prompt: Prompt configuration for the Worker.
        orchestrator_config: Agent configuration for the Orchestrator LLM.
        worker_config: Agent configuration for the Worker LLM.
        limits_config: Global limits configuration.
        tool_executor: Tool executor instance.
        orchestrator_client: LLM client instance for the orchestrator.
        worker_client: LLM client instance for the workers.
        console_handler: Optional console handler for CLI output.

    Returns:
        Compiled LangGraph workflow.
    """
    orchestrator_tools = get_orchestrator_tool_schemas()
    max_workers = limits_config.max_workers
    def _run_orchestrator_turn(
        state: OrchestratorWorkersState,
    ) -> dict[str, Any]:
        """Runs a single orchestrator turn without wall-clock timeout."""
        turn = state.get("turn_count", 0) + 1
        max_turns = limits_config.max_turns

        # Build system prompt with current state
        orch_state = state.get("orchestrator_state", {})
        state_json = _format_state_json(orch_state)
        worker_report = state.get("worker_report", "")

        system_prompt = format_system_prompt(
            orchestrator_prompt.system_prompt,
            state.get("objective", ""),
        )

        # Build instruction with worker report
        instruction = ""
        if orchestrator_prompt.instruction_template and worker_report:
            instruction = orchestrator_prompt.instruction_template.replace(
                "{worker_report}", worker_report
            )
        elif worker_report:
            instruction = f"## Worker Reports\n\n{worker_report}"

        # Build messages for orchestrator
        messages = ensure_system_message(
            list(state.get("orchestrator_messages", [])),
            system_prompt,
        )

        # Add current state context and worker report
        context_msg = f"## Current State\n\n```json\n{state_json}\n```"
        if instruction:
            context_msg += f"\n\n{instruction}"

        messages.append({"role": "user", "content": context_msg})

        http_timeout = (
            float(limits_config.worker_timeout_seconds)
            if limits_config.worker_timeout_seconds is not None
            else None
        )
        llm_result = run_llm_call(
            client=orchestrator_client,
            model=orchestrator_config.model,
            temperature=orchestrator_config.temperature,
            system_prompt=system_prompt,
            messages=messages,
            tools=orchestrator_tools,
            timeout_seconds=http_timeout,
            seed=limits_config.random_seed,
            max_message_chars=limits_config.max_message_chars,
            console_handler=console_handler,
            agent_role=AgentRole.ORCHESTRATOR,
            turn=turn,
            max_turns=max_turns,
            token_update_mode="delta",
            log_response=False,
        )
        response = llm_result["response"]

        # Execute tool calls (submit_goal / give_up) if present.
        done = False
        termination_reason = ""
        last_submit_feedback = ""
        needs_human = False
        if response.tool_calls:
            tool_calls_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
            tool_update = execute_tools_update(
                state=state,
                tool_calls=tool_calls_dicts,
                tool_executor=tool_executor,
                truncation_chars=limits_config.observation_truncation_chars,
                console_handler=console_handler,
                agent_role=AgentRole.ORCHESTRATOR,
                turn=turn,
                max_turns=max_turns,
                timeout_seconds=limits_config.worker_timeout_seconds,
            )
            if tool_update.get("done"):
                done = True
                termination_reason = (
                    tool_update.get("termination_reason") or "terminated"
                )
            needs_human = tool_update.get("needs_human", False)

        # Parse orchestrator output (state_update, tasks)
        parsed, _error = parse_typed_json(
            response.content or "", OrchestratorOutput
        )
        if parsed is None:
            tasks = []
            if not response.tool_calls:
                tasks = [{"type": "error", "description": "Failed to parse"}]
            orch_output = {
                "state_update": {},
                "analysis": response.content or "",
                "tasks": tasks,
            }
        else:
            orch_output = parsed.model_dump()

        # Update orchestrator state (ignore attempted from model output)
        state_update = orch_output.get("state_update", {})
        if "attempted" in state_update:
            state_update = {**state_update}
            state_update.pop("attempted", None)
        new_orch_state = _merge_state(orch_state, state_update)

        # Extract tasks for workers (backward compatible with single task)
        tasks = _extract_tasks(orch_output)

        # Log response
        if console_handler:
            task_count = len(tasks)
            first_task_desc = (
                tasks[0].get("description", "Unknown")[:40] if tasks else "None"
            )
            console_handler.on_agent_response(
                _turn=turn,
                _max_turns=max_turns,
                agent=AgentRole.ORCHESTRATOR,
                content=f"Tasks ({task_count}): {first_task_desc}...",
                _has_tool_calls=bool(response.tool_calls),
            )

        submitted_answers: list[Any] = []

        if response.tool_calls:
            for tc in response.tool_calls:
                if getattr(tc.function, "name", None) == "submit_goal":
                    try:
                        args = json.loads(getattr(tc.function, "arguments", "{}"))
                        submitted = args.get("answer")
                        if submitted is not None:
                            submitted_answers.append(submitted)
                    except (json.JSONDecodeError, TypeError):
                        continue

        # If no tool-call submit yet, check for submit task in JSON.
        # Only allow submit after at least one worker round (turn > 1).
        if not done and turn > 1:
            for task in tasks:
                if task.get("type") == "submit":
                    answer = (
                        task.get("answer")
                        or task.get("details")
                        or task.get("description")
                        or ""
                    )
                    if answer:
                        result = tool_executor.execute(
                            "submit_goal", {"answer": answer}
                        )
                        submitted_answers.append(answer)
                        if result.terminate:
                            done = True
                            termination_reason = (
                                result.termination_reason or "terminated"
                            )
                            break
                        else:
                            last_submit_feedback = (
                                result.output or "Submission rejected."
                            )

        if submitted_answers:
            attempted = list(new_orch_state.get("attempted") or [])
            for submitted in submitted_answers:
                attempted.append(
                    str(submitted) if not isinstance(submitted, str) else submitted
                )
            new_orch_state = {**new_orch_state, "attempted": attempted}

        return {
            "orchestrator_messages": [llm_result["assistant_message"]],
            "orchestrator_state": new_orch_state,
            "current_tasks": tasks,
            "worker_reports": [],  # Reset for new batch
            "worker_report": "",
            **llm_result["token_updates"],
            "turn_count": turn,
            "done": done,
            "termination_reason": termination_reason,
            "last_submit_feedback": last_submit_feedback,
            "needs_human": needs_human,
        }

    def orchestrator_node(state: OrchestratorWorkersState) -> dict[str, Any]:
        """Orchestrator node: plans using JSON state, generates tasks.

        Applies a per-turn wall-clock timeout when limits.worker_timeout_seconds
        is set to a positive value. On timeout, terminates with
        termination_reason \"llm_timeout\".
        """
        return run_with_timeout(
            fn=lambda: _run_orchestrator_turn(state),
            timeout_seconds=limits_config.worker_timeout_seconds,
            timeout_result={
                "done": True,
                "termination_reason": "llm_timeout",
            },
            log_message=(
                f"Orchestrator timed out after "
                f"{limits_config.worker_timeout_seconds}s. "
                "Terminating challenge."
            ),
            logger=logger,
        )

    _compiled_worker, worker_subgraph_node = build_worker_subgraph(
        worker_state_type=WorkerState,
        worker_prompt=worker_prompt,
        worker_config=worker_config,
        limits_config=limits_config,
        tool_executor=tool_executor,
        worker_client=worker_client,
        console_handler=console_handler,
        task_desc_keys=("description",),
        default_instruction_uses_task_content=False,
        suppress_tool_termination=False,
        logger=logger,
    )

    # --- Main graph nodes ---

    def end_turn_limit_node(state: OrchestratorWorkersState) -> dict[str, Any]:
        """Sets termination_reason to turn_limit before ending."""
        return turn_limit_update()

    def route_to_workers(
        state: OrchestratorWorkersState,
    ) -> list[Send] | Literal["end", "retry", "end_turn_limit"]:
        """Fan-out: dispatch tasks to parallel workers."""
        if should_terminate(state):
            return "end"

        if check_turn_limit(state, limits_config.max_turns):
            return "end_turn_limit"

        tasks = state.get("current_tasks", [])

        # Filter out submit/error tasks (already handled by orchestrator)
        valid_tasks = [
            t for t in tasks
            if t.get("type") not in ("submit", "error") and t.get("description")
        ]

        if not valid_tasks:
            # Give orchestrator another turn: with submit feedback, or a
            # "no tasks" prompt so it can output tasks or submit.
            return "retry"

        # Limit to max_workers
        tasks_to_dispatch = valid_tasks[:max_workers]

        sends: list[Send] = []
        for index, task in enumerate(tasks_to_dispatch, start=1):
            if console_handler is not None:
                worker_id = console_handler.spawn_worker()
            else:
                worker_id = index

            sends.append(
                Send(
                    "worker_subgraph",
                    {
                        "objective": state.get("objective", ""),
                        "task": task,
                        "worker_id": worker_id,
                        "worker_role": task.get("role", "Security Specialist"),
                        "worker_messages": [],
                        "worker_turn_count": 0,
                        "worker_report": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "tool_call_count": 0,
                        "done": False,
                        "termination_reason": "",
                        "needs_human": False,
                    },
                )
            )

        return sends

    def collect_reports(state: OrchestratorWorkersState) -> dict[str, Any]:
        """Fan-in: aggregate all worker reports."""
        reports = state.get("worker_reports", [])
        combined_report = "\n\n---\n\n".join(r for r in reports if r)
        return {
            "worker_report": combined_report,
            "done": state.get("done", False),
            "termination_reason": state.get("termination_reason", ""),
        }

    def inject_submit_feedback(
        state: OrchestratorWorkersState,
    ) -> dict[str, Any]:
        """Injects feedback as worker_report so orchestrator gets another turn.

        Uses last submit rejection if present; otherwise prompts for next tasks
        when the orchestrator output no dispatchable tasks (avoids stopping
        when tasks is empty).
        """
        feedback = state.get("last_submit_feedback", "").strip()
        if feedback:
            worker_report = f"## Last submission was rejected\n\n{feedback}"
        else:
            worker_report = (
                "## No tasks were dispatched this round\n\n"
                "You output no tasks (or only submit/error tasks). "
                "Provide next tasks or a submit task to continue."
            )
        return {
            "worker_report": worker_report,
            "last_submit_feedback": "",
        }

    def route_after_collect(
        state: OrchestratorWorkersState,
    ) -> Literal["orchestrator", "end", "end_turn_limit"]:
        """Routes after collecting worker reports."""
        if should_terminate(state):
            return "end"

        if check_turn_limit(state, limits_config.max_turns):
            return "end_turn_limit"

        return "orchestrator"

    # --- Build main graph ---
    graph = StateGraph(OrchestratorWorkersState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("worker_subgraph", worker_subgraph_node)
    graph.add_node("collect", collect_reports)
    graph.add_node("inject_submit_feedback", inject_submit_feedback)
    graph.add_node("end_turn_limit", end_turn_limit_node)

    graph.set_entry_point("orchestrator")

    # Fan-out with Send(), or retry orchestrator with submit feedback.
    graph.add_conditional_edges(
        "orchestrator",
        route_to_workers,
        {
            "end": END,
            "retry": "inject_submit_feedback",
            "end_turn_limit": "end_turn_limit",
        },
    )
    graph.add_edge("end_turn_limit", END)
    graph.add_edge("inject_submit_feedback", "orchestrator")

    # Fan-in: workers return to collect node
    graph.add_edge("worker_subgraph", "collect")

    # Back to orchestrator
    graph.add_conditional_edges(
        "collect",
        route_after_collect,
        {
            "orchestrator": "orchestrator",
            "end": END,
            "end_turn_limit": "end_turn_limit",
        },
    )

    return graph.compile()
