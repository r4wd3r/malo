"""Orchestrator-Critic-Workers LangGraph implementation.

This architecture adds a Critic review step to the hierarchical workflow:
- Orchestrator: Proposes plans using structured JSON (with submit_goal/give_up)
- Critic: Reviews plans against challenge description and findings
- Workers: Execute approved tasks in parallel mini ReAct loops

The Critic evaluates safety, relevance to the challenge, and alignment
with current findings (System 2 thinking).
"""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, Literal

from langgraph.constants import Send
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from MaLO.config.models import (
    BaseAgentConfig,
    CriticConfig,
    LimitsConfig,
    PromptConfig,
)
from MaLO.graphs.engine import (
    ensure_system_message,
    execute_tools_update,
    run_with_timeout,
    run_llm_call,
    turn_limit_update,
)
from MaLO.graphs.orchestrator_utils import (
    extract_tasks_from_plan as _extract_tasks_from_plan,
    format_state_json as _format_state_json,
    merge_state as _merge_state,
    reduce_worker_reports as _reduce_worker_reports,
)
from MaLO.graphs.state import OrchestratorState as OrchestratorStateBase
from MaLO.graphs.state import WorkerState as WorkerStateBase
from MaLO.graphs.worker_subgraph import build_worker_subgraph
from MaLO.graphs.output_models import CriticOutput, OrchestratorCriticOutput
from MaLO.graphs.utils import (
    check_turn_limit,
    extract_json_from_response,
    format_system_prompt,
    parse_typed_json,
    should_terminate,
)
from MaLO.observability.console_handler import AgentRole, ConsoleHandler
from MaLO.tools.executor import ToolExecutor
from MaLO.tools.schemas import get_orchestrator_tool_schemas
from MaLO.utils.llm_client import LLMClient
from MaLO.utils.text_utils import truncate_text


logger = logging.getLogger(__name__)


class OrchestratorCriticWorkersState(OrchestratorStateBase):
    """State for the orchestrator-critic-workers graph."""

    # Orchestrator state
    orchestrator_state: dict[str, Any]
    orchestrator_messages: Annotated[list[dict[str, Any]], add_messages]
    proposed_plan: dict[str, Any]
    # Critic state
    critic_verdict: str
    critic_feedback: str
    rejection_count: int
    # Parallel workers (aligned with orchestrator_workers)
    current_tasks: list[dict[str, Any]]
    worker_reports: Annotated[list[str], _reduce_worker_reports]
    worker_report: str
    last_submit_feedback: str

class WorkerState(WorkerStateBase):
    """State for a single worker branch."""


MAX_REJECTIONS = 3


def build_orchestrator_critic_workers_graph(
    orchestrator_prompt: PromptConfig,
    critic_prompt: PromptConfig,
    worker_prompt: PromptConfig,
    orchestrator_config: BaseAgentConfig,
    critic_config: CriticConfig,
    worker_config: BaseAgentConfig,
    limits_config: LimitsConfig,
    tool_executor: ToolExecutor,
    orchestrator_client: LLMClient,
    critic_client: LLMClient,
    worker_client: LLMClient,
    console_handler: ConsoleHandler | None = None,
) -> StateGraph:
    """Builds an orchestrator-critic-workers LangGraph workflow.

    Flow: Orchestrator -> Critic -> (REJECT -> Orchestrator | APPROVE ->
    prepare_approved_tasks -> route_to_workers -> worker_subgraph(s) ->
    collect -> Orchestrator).
    """
    orchestrator_tools = get_orchestrator_tool_schemas()
    max_workers = limits_config.max_workers

    # --- Orchestrator node ---
    def _run_orchestrator_turn(
        state: OrchestratorCriticWorkersState,
    ) -> dict[str, Any]:
        """Runs a single orchestrator turn without wall-clock timeout."""
        turn = state.get("turn_count", 0) + 1
        max_turns = limits_config.max_turns

        orch_state = state.get("orchestrator_state", {})
        state_json = _format_state_json(orch_state)
        worker_report = state.get("worker_report", "")
        critic_feedback = state.get("critic_feedback", "")
        last_submit_feedback = state.get("last_submit_feedback", "")

        system_prompt = format_system_prompt(
            orchestrator_prompt.system_prompt,
            state.get("objective", ""),
        )

        instruction_parts = []
        if critic_feedback:
            instruction_parts.append(
                f"## Critic Feedback (Plan Rejected)\n\n{critic_feedback}"
            )
        if last_submit_feedback:
            instruction_parts.append(
                f"## Last submission was rejected\n\n{last_submit_feedback}"
            )
        if worker_report:
            if orchestrator_prompt.instruction_template:
                instruction_parts.append(
                    orchestrator_prompt.instruction_template.replace(
                        "{worker_report}", worker_report
                    )
                )
            else:
                instruction_parts.append(f"## Worker Reports\n\n{worker_report}")

        messages = ensure_system_message(
            list(state.get("orchestrator_messages", [])),
            system_prompt,
        )

        context_msg = f"## Current State\n\n```json\n{state_json}\n```"
        if instruction_parts:
            context_msg += "\n\n" + "\n\n".join(instruction_parts)
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
        done = False
        termination_reason = ""
        new_last_submit_feedback = ""

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
            else:
                # Submit failed or gave up — feed back to orchestrator (same
                # as orchestrator_workers: feedback is the tool output; here
                # it lives in the last tool message, not on ToolExecutionResult).
                _content = ""
                if tool_update.get("messages"):
                    _content = (
                        tool_update.get("messages", [])[-1].get("content", "")
                        or ""
                    ).strip()
                new_last_submit_feedback = (
                    _content if _content else "Submission rejected."
                )
            needs_human = tool_update.get("needs_human", False)

        raw_output = extract_json_from_response(response.content or "")
        parsed, _error = parse_typed_json(
            response.content or "", OrchestratorCriticOutput
        )
        if parsed is None:
            tasks = []
            objective = "Tool call issued"
            if not response.tool_calls:
                tasks = [{"type": "error", "description": "Failed to parse"}]
                objective = "Parse error"
            orch_output = {
                "analysis": response.content or "",
                "proposed_plan": {"objective": objective, "tasks": tasks},
            }
        else:
            orch_output = parsed.model_dump()

        proposed_plan = orch_output.get("proposed_plan", orch_output)
        if not proposed_plan and raw_output:
            proposed_plan = raw_output
        state_update = orch_output.get("state_update", {})
        if "attempted" in state_update:
            state_update = {**state_update}
            state_update.pop("attempted", None)
        new_orch_state = _merge_state(orch_state, state_update)

        if response.tool_calls:
            submitted_answers: list[Any] = []
            for tc in response.tool_calls:
                if getattr(tc.function, "name", None) == "submit_goal":
                    try:
                        args = json.loads(
                            getattr(tc.function, "arguments", "{}")
                        )
                        submitted = args.get("answer")
                        if submitted is not None:
                            submitted_answers.append(submitted)
                    except (json.JSONDecodeError, TypeError):
                        pass
            if submitted_answers:
                attempted = list(new_orch_state.get("attempted") or [])
                for submitted in submitted_answers:
                    attempted.append(
                        str(submitted)
                        if not isinstance(submitted, str)
                        else submitted
                    )
                new_orch_state = {
                    **new_orch_state,
                    "attempted": attempted,
                }
        if console_handler:
            plan_obj = proposed_plan.get("objective", "Unknown")[:50]
            console_handler.on_agent_response(
                _turn=turn,
                _max_turns=max_turns,
                agent=AgentRole.ORCHESTRATOR,
                content=f"Proposed: {plan_obj}",
                _has_tool_calls=bool(response.tool_calls),
            )

        return {
            "orchestrator_messages": [llm_result["assistant_message"]],
            "orchestrator_state": new_orch_state,
            "proposed_plan": proposed_plan,
            "critic_feedback": "",
            "last_submit_feedback": new_last_submit_feedback,
            **llm_result["token_updates"],
            "turn_count": turn,
            "done": done,
            "termination_reason": termination_reason,
            "needs_human": needs_human,
        }

    def orchestrator_node(
        state: OrchestratorCriticWorkersState,
    ) -> dict[str, Any]:
        """Orchestrator: proposes plans and can call submit_goal/give_up.

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

    # --- Critic node: challenge + findings + proposed plan ---
    def _run_critic_turn(
        state: OrchestratorCriticWorkersState,
    ) -> dict[str, Any]:
        """Runs a single critic turn without wall-clock timeout."""
        turn = state.get("turn_count", 0)
        max_turns = limits_config.max_turns

        objective = state.get("objective", "")
        orch_state = state.get("orchestrator_state", {})
        worker_report = state.get("worker_report", "")
        proposed_plan = state.get("proposed_plan", {})

        # Build structured context for critic: challenge, findings, plan
        findings_json = json.dumps(orch_state, indent=2)
        plan_json = json.dumps(proposed_plan, indent=2)
        report_truncation = limits_config.observation_truncation_chars
        worker_report_truncated = truncate_text(worker_report, report_truncation)

        context_block = (
            "## Challenge and goal\n\n"
            f"{objective}\n\n"
            "## Current findings (from state and workers)\n\n"
            f"```json\n{findings_json}\n```\n\n"
            "Latest worker report:\n\n"
            f"{worker_report_truncated}\n\n"
            "## Proposed plan from orchestrator\n\n"
            f"```json\n{plan_json}\n```"
        )

        system_prompt = format_system_prompt(
            critic_prompt.system_prompt,
            objective,
        )
        if critic_prompt.instruction_template:
            instruction = critic_prompt.instruction_template.replace(
                "{evaluation_context}", context_block
            )
        else:
            instruction = context_block

        http_timeout = (
            float(limits_config.worker_timeout_seconds)
            if limits_config.worker_timeout_seconds is not None
            else None
        )
        llm_result = run_llm_call(
            client=critic_client,
            model=critic_config.model,
            temperature=critic_config.temperature,
            system_prompt=system_prompt,
            messages=[],
            tools=None,
            timeout_seconds=http_timeout,
            seed=limits_config.random_seed,
            max_message_chars=limits_config.max_message_chars,
            console_handler=console_handler,
            agent_role=AgentRole.CRITIC,
            turn=turn,
            max_turns=max_turns,
            instruction=instruction,
            token_update_mode="delta",
            log_response=False,
        )
        response = llm_result["response"]

        parsed, _error = parse_typed_json(response.content or "", CriticOutput)
        if parsed is None:
            critic_output: dict[str, Any] = {
                "verdict": "APPROVE",
                "feedback": "Unable to parse - defaulting to approve",
            }
        else:
            critic_output = parsed.model_dump()

        verdict = critic_output.get("verdict", "APPROVE")
        feedback = critic_output.get("feedback", "") or ""
        suggestion = critic_output.get("suggestion", "") or ""
        if suggestion:
            feedback = f"{feedback}\n\nSuggestion: {suggestion}"

        rejection_count = state.get("rejection_count", 0)
        if verdict == "REJECT":
            rejection_count += 1
        else:
            rejection_count = 0
        if rejection_count >= MAX_REJECTIONS:
            verdict = "APPROVE"
            feedback = f"Forced approval after {MAX_REJECTIONS} rejections."

        if console_handler:
            reason = feedback[:30] if verdict == "REJECT" else None
            console_handler.on_critique(
                turn=turn,
                max_turns=max_turns,
                verdict=verdict,
                reason=reason,
            )

        return {
            "critic_verdict": verdict,
            "critic_feedback": feedback if verdict == "REJECT" else "",
            "rejection_count": rejection_count,
            **llm_result["token_updates"],
        }

    def critic_node(state: OrchestratorCriticWorkersState) -> dict[str, Any]:
        """Critic: reviews plan against challenge description and findings.

        Applies a per-turn wall-clock timeout when limits.worker_timeout_seconds
        is set to a positive value. On timeout, treats as approve and continues
        so the workflow does not hang.
        """
        timeout_seconds = limits_config.worker_timeout_seconds
        return run_with_timeout(
            fn=lambda: _run_critic_turn(state),
            timeout_seconds=timeout_seconds,
            timeout_result={
                "critic_verdict": "APPROVE",
                "critic_feedback": "",
                "rejection_count": state.get("rejection_count", 0),
            },
            log_message=(
                f"Critic timed out after {timeout_seconds}s. "
                "Treating as approve and continuing."
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
        task_desc_keys=("description", "script"),
        default_instruction_uses_task_content=True,
        suppress_tool_termination=True,
        logger=logger,
    )

    # --- prepare_approved_tasks: set current_tasks from proposed_plan ---
    def prepare_approved_tasks(
        state: OrchestratorCriticWorkersState,
    ) -> dict[str, Any]:
        """Sets current_tasks from approved proposed_plan for dispatch."""
        proposed_plan = state.get("proposed_plan", {})
        current_tasks = _extract_tasks_from_plan(proposed_plan)

        submitted_answers: list[Any] = []
        last_submit_feedback = ""
        done = False
        termination_reason = ""

        turn = state.get("turn_count", 0)
        if turn > 1:
            for task in current_tasks:
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
                        last_submit_feedback = (
                            result.output or "Submission rejected."
                        )

        if submitted_answers:
            orch_state = state.get("orchestrator_state", {})
            attempted = list(orch_state.get("attempted") or [])
            for submitted in submitted_answers:
                attempted.append(
                    str(submitted) if not isinstance(submitted, str) else submitted
                )
            orch_state = {**orch_state, "attempted": attempted}
        else:
            orch_state = state.get("orchestrator_state", {})

        non_submit_tasks = [
            t for t in current_tasks if t.get("type") != "submit"
        ]

        return {
            "current_tasks": non_submit_tasks,
            "worker_reports": [],
            "last_submit_feedback": last_submit_feedback,
            "orchestrator_state": orch_state,
            "done": done,
            "termination_reason": termination_reason,
        }

    def end_turn_limit_node(
        state: OrchestratorCriticWorkersState,
    ) -> dict[str, Any]:
        """Sets termination_reason to turn_limit before ending."""
        return turn_limit_update()

    # --- route_to_workers: retry or Send list ---
    def route_to_workers(
        state: OrchestratorCriticWorkersState,
    ) -> list[Send] | Literal["end", "retry", "end_turn_limit"]:
        if should_terminate(state):
            return "end"
        if check_turn_limit(state, limits_config.max_turns):
            return "end_turn_limit"
        tasks = state.get("current_tasks", [])
        valid_tasks = [
            t for t in tasks
            if t.get("type") not in ("submit", "error")
            and (t.get("description") or t.get("script"))
        ]
        if not valid_tasks:
            return "retry"
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

    def collect_reports(state: OrchestratorCriticWorkersState) -> dict[str, Any]:
        reports = state.get("worker_reports", [])
        combined = "\n\n---\n\n".join(r for r in reports if r)
        return {
            "worker_report": combined,
            "done": state.get("done", False),
            "termination_reason": state.get("termination_reason", ""),
        }

    def inject_submit_feedback(
        state: OrchestratorCriticWorkersState,
    ) -> dict[str, Any]:
        feedback = state.get("last_submit_feedback", "").strip()
        if feedback:
            worker_report = f"## Last submission was rejected\n\n{feedback}"
        else:
            worker_report = (
                "## No tasks were dispatched this round\n\n"
                "You output no tasks (or only submit/error tasks). "
                "Provide next tasks or a submit task to continue."
            )
        return {"worker_report": worker_report, "last_submit_feedback": ""}

    def route_after_collect(
        state: OrchestratorCriticWorkersState,
    ) -> Literal["orchestrator", "end", "end_turn_limit"]:
        if should_terminate(state):
            return "end"
        if check_turn_limit(state, limits_config.max_turns):
            return "end_turn_limit"
        return "orchestrator"

    def route_after_orchestrator(
        state: OrchestratorCriticWorkersState,
    ) -> Literal["critic", "end", "end_turn_limit"]:
        if should_terminate(state):
            return "end"
        if check_turn_limit(state, limits_config.max_turns):
            return "end_turn_limit"
        if not state.get("proposed_plan"):
            return "end"
        return "critic"

    def route_after_critic(
        state: OrchestratorCriticWorkersState,
    ) -> Literal["orchestrator", "prepare_approved_tasks", "end"]:
        if should_terminate(state):
            return "end"
        if state.get("critic_verdict", "APPROVE") == "REJECT":
            return "orchestrator"
        return "prepare_approved_tasks"

    # --- Build main graph ---
    graph = StateGraph(OrchestratorCriticWorkersState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("prepare_approved_tasks", prepare_approved_tasks)
    graph.add_node("worker_subgraph", worker_subgraph_node)
    graph.add_node("collect", collect_reports)
    graph.add_node("inject_submit_feedback", inject_submit_feedback)
    graph.add_node("end_turn_limit", end_turn_limit_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "critic": "critic",
            "end": END,
            "end_turn_limit": "end_turn_limit",
        },
    )
    graph.add_edge("end_turn_limit", END)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "orchestrator": "orchestrator",
            "prepare_approved_tasks": "prepare_approved_tasks",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "prepare_approved_tasks",
        route_to_workers,
        {
            "end": END,
            "retry": "inject_submit_feedback",
            "end_turn_limit": "end_turn_limit",
        },
    )
    graph.add_edge("inject_submit_feedback", "orchestrator")
    graph.add_edge("worker_subgraph", "collect")
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
