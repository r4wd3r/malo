from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from MaLO.challenges.models import Goal, GoalType
from MaLO.config.models import BaseAgentConfig, LimitsConfig, PromptConfig
from MaLO.graphs import (
    build_orchestrator_critic_workers_graph,
    build_orchestrator_workers_graph,
    build_reasoner_executor_graph,
    build_single_agent_graph,
)
from MaLO.graphs.utils import get_message_attr
from MaLO.tools.core_tools import ToolResult
from MaLO.run_helpers import get_initial_state_for_architecture
from MaLO.tools.core_tools import give_up, request_human_help, submit_goal, task_complete
from MaLO.tools.executor import ToolExecutor
from MaLO.validation.goal_validator import DefaultGoalValidator
from MaLO.utils.openrouter_client import ChatResponse


class _NoToolClient:
    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            content="ok",
            tool_calls=[],
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


class _OrchestratorTurnLimitClient:
    def __init__(self) -> None:
        self._count = 0

    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        self._count += 1
        if model == "orch":
            # Always emit a dispatchable task; with max_turns=1 this will hit
            # turn_limit after the worker round.
            payload = {
                "state_update": {},
                "analysis": "keep going",
                "tasks": [
                    {"description": "inspect", "type": "explore", "role": "Worker"}
                ],
            }
            return ChatResponse(
                content=json.dumps(payload),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        if model == "worker":
            return ChatResponse(
                content="work",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        raise AssertionError(f"Unexpected model: {model}")


class _GiveUpToolExecutor(ToolExecutor):
    def __init__(self) -> None:
        super().__init__(env_manager=None, truncation_chars=1000, goal_validator=None)

    def execute(
        self, tool_name: str, arguments: dict[str, Any], timeout_seconds: int | None = None
    ) -> ToolResult:  # type: ignore[override]
        if tool_name == "give_up":
            return ToolResult(
                output="gave up",
                success=False,
                terminate=True,
                termination_reason="give_up",
            )
        return ToolResult(output="ok", success=True)


@dataclass
class _ToolFunction:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _ToolFunction


class _OrchestratorCriticTurnLimitClient:
    def __init__(self) -> None:
        self._count = 0

    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        self._count += 1
        if model == "orch":
            payload = {
                "state_update": {},
                "analysis": "plan",
                "proposed_plan": {
                    "objective": "inspect",
                    "tasks": [
                        {
                            "description": "inspect",
                            "type": "explore",
                            "role": "Worker",
                        }
                    ],
                },
            }
            return ChatResponse(
                content=json.dumps(payload),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        if model == "critic":
            verdict = {"verdict": "APPROVE", "feedback": "ok"}
            return ChatResponse(
                content=json.dumps(verdict),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        if model == "worker":
            return ChatResponse(
                content="work",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        raise AssertionError(f"Unexpected model: {model}")


def _limits(max_turns: int) -> LimitsConfig:
    return LimitsConfig(
        max_turns=max_turns,
        max_cost_usd=1.0,
        observation_truncation_chars=1000,
        recursion_limit=50,
        max_workers=1,
        worker_timeout_seconds=5,
        challenge_timeout_seconds=30,
    )


def test_goal_exact_match() -> None:
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "CTF{ok}", "case_sensitive": True},
    )
    assert goal.validate_answer("CTF{ok}")[0] is True
    assert goal.validate_answer("ctf{ok}")[0] is False


def test_goal_list_match_order() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={"expected": ["a", "b"], "order_matters": True},
    )
    assert goal.validate_answer(["a", "b"])[0] is True
    assert goal.validate_answer(["b", "a"])[0] is False


def test_goal_list_match_unordered() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={"expected": ["a", "b"], "order_matters": False},
    )
    assert goal.validate_answer(["b", "a"])[0] is True


def test_goal_list_match_parses_json_list() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={"expected": ["a", "b"], "order_matters": False},
    )
    assert goal.validate_answer('["a", "b"]')[0] is True


def test_goal_list_match_parses_python_list() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={"expected": ["a", "b"], "order_matters": False},
    )
    assert goal.validate_answer("['a', 'b']")[0] is True


def test_goal_list_match_parses_newlines() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={"expected": ["a", "b"], "order_matters": False},
    )
    assert goal.validate_answer("a\nb")[0] is True


def test_goal_list_match_case_insensitive() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={
            "expected": ["Aa", "Bb"],
            "order_matters": False,
            "case_sensitive": False,
        },
    )
    assert goal.validate_answer(["aa", "bb"])[0] is True


def test_goal_list_match_whitespace_split_for_simple_tokens() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={"expected": ["1.1.1.1", "2.2.2.2"], "order_matters": False},
    )
    assert goal.validate_answer("1.1.1.1 2.2.2.2")[0] is True


def test_goal_list_match_preserves_spaces_when_expected_has_spaces() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={
            "expected": ["a b", "c d"],
            "order_matters": False,
        },
    )
    assert goal.validate_answer("a b\nc d")[0] is True


def test_goal_list_match_extracts_ipv4_with_labels() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={
            "expected": ["192.168.1.140", "192.168.2.50"],
            "order_matters": False,
        },
    )
    answer = "Malicious IPs: 192.168.1.140, 192.168.2.50"
    assert goal.validate_answer(answer)[0] is True


def test_goal_list_match_strips_numbered_prefixes() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={"expected": ["alpha", "beta"], "order_matters": False},
    )
    assert goal.validate_answer("1. alpha\n2) beta")[0] is True


def test_goal_list_match_parses_dict_answer() -> None:
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="list",
        validation_data={"expected": ["a", "b"], "order_matters": False},
    )
    assert goal.validate_answer({"answers": ["a", "b"]})[0] is True


def test_goal_regex() -> None:
    goal = Goal(
        goal_type=GoalType.REGEX,
        description="regex",
        validation_data={"pattern": r"CTF\{.+\}"},
    )
    assert goal.validate_answer("CTF{ok}")[0] is True
    assert goal.validate_answer("nope")[0] is False


def test_submit_goal_tool_exact_match() -> None:
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "CTF{ok}"},
    )
    result = submit_goal("CTF{ok}", goal)
    assert result.terminate is True
    assert result.termination_reason == "success"


def test_submit_goal_tool_reject() -> None:
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "CTF{ok}"},
    )
    result = submit_goal("bad", goal)
    assert result.terminate is False


def test_give_up_tool() -> None:
    result = give_up("no")
    assert result.terminate is True
    assert result.termination_reason == "give_up"


def test_task_complete_tool() -> None:
    result = task_complete("done")
    assert result.terminate is True
    assert result.termination_reason == "task_complete"


def test_request_human_help_tool() -> None:
    result = request_human_help("help")
    assert result.needs_human is True
    assert result.terminate is False


def test_single_react_request_human_help_sets_state_flag() -> None:
    class _HumanHelpClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if self._count == 1:
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "request_human_help",
                            json.dumps({"query": "Need input"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="help",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            return ChatResponse(
                content="ok",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _HumanHelpClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    prompt = PromptConfig(system_prompt="{objective}")
    agent_cfg = BaseAgentConfig(model="agent", temperature=0.0, prompts_file="")

    limits = _limits(3)

    graph = build_single_agent_graph(
        prompt_config=prompt,
        agent_config=agent_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        client=client,
    )
    state = get_initial_state_for_architecture("single_react", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("needs_human") is True


def test_single_react_turn_limit_sets_reason() -> None:
    client = _NoToolClient()
    prompt = PromptConfig(system_prompt="{objective}")
    agent_cfg = BaseAgentConfig(model="test-agent", temperature=0.0, prompts_file="")
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)

    graph = build_single_agent_graph(
        prompt_config=prompt,
        agent_config=agent_cfg,
        limits_config=_limits(0),
        tool_executor=tool_executor,
        client=client,
    )
    state = get_initial_state_for_architecture("single_react", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 20})

    assert final_state.get("termination_reason") == "turn_limit"


def test_reasoner_executor_turn_limit_sets_reason() -> None:
    client = _NoToolClient()
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(system_prompt="{objective}", instruction_template="{worker_report}")
    reasoner_cfg = BaseAgentConfig(model="test-reasoner", temperature=0.0, prompts_file="")
    executor_cfg = BaseAgentConfig(model="test-executor", temperature=0.0, prompts_file="")
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=_limits(0),
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 20})

    assert final_state.get("termination_reason") == "turn_limit"


def test_single_react_give_up_propagates() -> None:
    class _GiveUpClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            tool_calls = [
                _ToolCall(
                    "1",
                    _ToolFunction("give_up", json.dumps({"reason": "nope"})),
                )
            ]
            return ChatResponse(
                content="giving up",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _GiveUpClient()
    tool_executor = _GiveUpToolExecutor()
    prompt = PromptConfig(system_prompt="{objective}")
    agent_cfg = BaseAgentConfig(model="agent", temperature=0.0, prompts_file="")

    graph = build_single_agent_graph(
        prompt_config=prompt,
        agent_config=agent_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        client=client,
    )
    state = get_initial_state_for_architecture("single_react", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") == "give_up"


def test_reasoner_executor_give_up_propagates() -> None:
    class _ReasonerGiveUpClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "reasoner":
                cmd = {"thought": "done", "command": {"type": "give_up", "reason": "nope"}}
                return ChatResponse(
                    content=json.dumps(cmd),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "executor":
                # Executor LLM can emit give_up tool call directly.
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction("give_up", json.dumps({"reason": "nope"})),
                    )
                ]
                return ChatResponse(
                    content="giving up",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _ReasonerGiveUpClient()
    tool_executor = _GiveUpToolExecutor()
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(system_prompt="{objective}", instruction_template="{worker_report}")
    reasoner_cfg = BaseAgentConfig(model="reasoner", temperature=0.0, prompts_file="")
    executor_cfg = BaseAgentConfig(model="executor", temperature=0.0, prompts_file="")

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") == "give_up"


def test_orchestrator_workers_turn_limit_sets_reason() -> None:
    client = _OrchestratorTurnLimitClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(1),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state.get("termination_reason") == "turn_limit"


def test_orchestrator_critic_workers_turn_limit_sets_reason() -> None:
    client = _OrchestratorCriticTurnLimitClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(1),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 150})

    assert final_state.get("termination_reason") == "turn_limit"


def test_orchestrator_workers_give_up_propagates() -> None:
    class _OrchGiveUpClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "orch":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "give_up",
                            json.dumps({"reason": "nope"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="giving up",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="work",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchGiveUpClient()
    tool_executor = _GiveUpToolExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state.get("termination_reason") == "give_up"


def test_orchestrator_critic_workers_give_up_propagates() -> None:
    class _OrchCriticGiveUpClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "orch":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction("give_up", json.dumps({"reason": "nope"})),
                    )
                ]
                return ChatResponse(
                    content="giving up",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="work",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchCriticGiveUpClient()
    tool_executor = _GiveUpToolExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state.get("termination_reason") == "give_up"


def test_single_react_success_propagates() -> None:
    class _SubmitClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            tool_calls = [
                _ToolCall(
                    "1",
                    _ToolFunction("submit_goal", json.dumps({"answer": "flag"})),
                )
            ]
            return ChatResponse(
                content="submit",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _SubmitClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "flag"},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    prompt = PromptConfig(system_prompt="{objective}")
    agent_cfg = BaseAgentConfig(model="agent", temperature=0.0, prompts_file="")

    graph = build_single_agent_graph(
        prompt_config=prompt,
        agent_config=agent_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        client=client,
    )
    state = get_initial_state_for_architecture("single_react", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") in {"success", "turn_limit"}


def test_reasoner_executor_success_propagates() -> None:
    class _ReasonerSuccessClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "reasoner":
                cmd = {"thought": "submit", "command": {"type": "submit", "answer": "flag"}}
                return ChatResponse(
                    content=json.dumps(cmd),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "executor":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction("submit_goal", json.dumps({"answer": "flag"})),
                    )
                ]
                return ChatResponse(
                    content="submit",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _ReasonerSuccessClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "flag"},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(system_prompt="{objective}", instruction_template="{worker_report}")
    reasoner_cfg = BaseAgentConfig(model="reasoner", temperature=0.0, prompts_file="")
    executor_cfg = BaseAgentConfig(model="executor", temperature=0.0, prompts_file="")

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") in {"success", "turn_limit"}


def test_orchestrator_workers_success_propagates() -> None:
    class _OrchSuccessClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                if self._count == 1:
                    payload = {
                        "state_update": {},
                        "analysis": "plan",
                        "tasks": [
                            {"description": "inspect", "type": "explore", "role": "Worker"}
                        ],
                    }
                else:
                    payload = {
                        "state_update": {},
                        "analysis": "submit",
                        "tasks": [
                            {"description": "submit", "type": "submit", "answer": "flag"}
                        ],
                    }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="done",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchSuccessClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "flag"},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state.get("termination_reason") in {"success", "turn_limit"}


def test_orchestrator_critic_workers_success_propagates() -> None:
    class _OrchCriticSuccessClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                if self._count == 1:
                    payload = {
                        "state_update": {},
                        "analysis": "plan",
                        "proposed_plan": {
                            "objective": "inspect",
                            "tasks": [
                                {
                                    "description": "inspect",
                                    "type": "explore",
                                    "role": "Worker",
                                }
                            ],
                        },
                    }
                else:
                    payload = {
                        "state_update": {},
                        "analysis": "submit",
                        "proposed_plan": {
                            "objective": "submit",
                            "tasks": [
                                {
                                    "description": "submit",
                                    "type": "submit",
                                    "answer": "flag",
                                }
                            ],
                        },
                    }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="done",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchCriticSuccessClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "flag"},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 150})

    assert final_state.get("termination_reason") in {"success", "turn_limit"}


def test_orchestrator_workers_worker_submit_stops_run() -> None:
    class _OrchWorkerSubmitClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                payload = {
                    "state_update": {},
                    "analysis": "plan",
                    "tasks": [
                        {"description": "submit", "type": "explore", "role": "Worker"}
                    ],
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "submit_goal",
                            json.dumps({"answer": "flag"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="submit",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchWorkerSubmitClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "flag"},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state.get("termination_reason") == "success"


def test_orchestrator_critic_workers_worker_submit_stops_run() -> None:
    class _OrchCriticWorkerSubmitClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                payload = {
                    "state_update": {},
                    "analysis": "plan",
                    "proposed_plan": {
                        "objective": "submit",
                        "tasks": [
                            {
                                "description": "submit",
                                "type": "explore",
                                "role": "Worker",
                            }
                        ],
                    },
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "submit_goal",
                            json.dumps({"answer": "flag"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="submit",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchCriticWorkerSubmitClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "flag"},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 150})

    assert final_state.get("termination_reason") == "success"


def test_orchestrator_critic_workers_submit_task_stops_run() -> None:
    class _SubmitPlanClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "orch":
                self.calls += 1
                plan = {
                    "proposed_plan": {
                        "objective": "submit",
                        "tasks": [
                            {
                                "description": "submit",
                                "type": "submit",
                                "answer": "flag",
                            }
                        ],
                    }
                }
                return ChatResponse(
                    content=json.dumps(plan),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="noop",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _SubmitPlanClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "flag"},
    )
    tool_executor = ToolExecutor(
        env_manager=None,
        truncation_chars=1000,
        goal_validator=DefaultGoalValidator(goal),
    )
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(max_turns=5),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 200})

    assert final_state.get("termination_reason") == "success"


def test_single_react_list_match_submit_succeeds() -> None:
    class _SubmitListClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            tool_calls = [
                _ToolCall(
                    "1",
                    _ToolFunction(
                        "submit_goal",
                        json.dumps({"answer": '["1.1.1.1","2.2.2.2"]'}),
                    ),
                )
            ]
            return ChatResponse(
                content="submit",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _SubmitListClient()
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="ips",
        validation_data={"expected": ["1.1.1.1", "2.2.2.2"], "order_matters": False},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    prompt = PromptConfig(system_prompt="{objective}")
    agent_cfg = BaseAgentConfig(model="agent", temperature=0.0, prompts_file="")

    graph = build_single_agent_graph(
        prompt_config=prompt,
        agent_config=agent_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        client=client,
    )
    state = get_initial_state_for_architecture("single_react", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") == "success"


def test_reasoner_executor_list_match_submit_succeeds() -> None:
    class _ReasonerListClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "reasoner":
                cmd = {
                    "thought": "submit",
                    "command": {"type": "submit", "answer": "1.1.1.1\n2.2.2.2"},
                }
                return ChatResponse(
                    content=json.dumps(cmd),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "executor":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "submit_goal",
                            json.dumps({"answer": "1.1.1.1\n2.2.2.2"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="submit",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _ReasonerListClient()
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="ips",
        validation_data={"expected": ["1.1.1.1", "2.2.2.2"], "order_matters": False},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(system_prompt="{objective}", instruction_template="{worker_report}")
    reasoner_cfg = BaseAgentConfig(model="reasoner", temperature=0.0, prompts_file="")
    executor_cfg = BaseAgentConfig(model="executor", temperature=0.0, prompts_file="")

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") == "success"


def test_orchestrator_workers_list_match_submit_succeeds() -> None:
    class _OrchListClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "submit_goal",
                            json.dumps({"answer": "1.1.1.1, 2.2.2.2"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="submit",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="work",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchListClient()
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="ips",
        validation_data={"expected": ["1.1.1.1", "2.2.2.2"], "order_matters": False},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state.get("termination_reason") == "success"
    assert (
        final_state.get("orchestrator_state", {}).get("attempted", [])
        == ["1.1.1.1, 2.2.2.2"]
    )


def test_orchestrator_critic_workers_list_match_submit_succeeds() -> None:
    class _OrchCriticListClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "submit_goal",
                            json.dumps({"answer": "1.1.1.1 2.2.2.2"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="submit",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="work",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchCriticListClient()
    goal = Goal(
        goal_type=GoalType.LIST_MATCH,
        description="ips",
        validation_data={"expected": ["1.1.1.1", "2.2.2.2"], "order_matters": False},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(3),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 150})

    assert final_state.get("termination_reason") == "success"
    assert (
        final_state.get("orchestrator_state", {}).get("attempted", [])
        == ["1.1.1.1 2.2.2.2"]
    )


def test_orchestrator_workers_state_update_attempted_ignored() -> None:
    class _OrchAttemptedClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "orch":
                payload = {
                    "state_update": {"attempted": ["not submitted"]},
                    "analysis": "no submit",
                    "tasks": [],
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchAttemptedClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(1),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("orchestrator_state", {}).get("attempted", []) == []


def test_orchestrator_critic_workers_state_update_attempted_ignored() -> None:
    class _OrchAttemptedClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "orch":
                payload = {
                    "state_update": {"attempted": ["not submitted"]},
                    "analysis": "no submit",
                    "proposed_plan": {"objective": "noop", "tasks": []},
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchAttemptedClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(1),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture(
        "orchestrator_critic_workers", "obj"
    )
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("orchestrator_state", {}).get("attempted", []) == []


def test_orchestrator_workers_tool_call_parse_error_no_task() -> None:
    class _OrchToolCallClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "orch":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "submit_goal",
                            json.dumps({"answer": "bad"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="not json",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="work",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchToolCallClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "good", "case_sensitive": True},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(1),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("current_tasks", []) == []


def test_orchestrator_critic_workers_tool_call_parse_error_no_task() -> None:
    class _OrchToolCallClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "orch":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "submit_goal",
                            json.dumps({"answer": "bad"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="not json",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                return ChatResponse(
                    content="work",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _OrchToolCallClient()
    goal = Goal(
        goal_type=GoalType.EXACT_MATCH,
        description="flag",
        validation_data={"expected": "good", "case_sensitive": True},
    )
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=DefaultGoalValidator(goal))
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(1),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture(
        "orchestrator_critic_workers", "obj"
    )
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    proposed_plan = final_state.get("proposed_plan", {})
    assert proposed_plan.get("tasks", []) == []


def test_single_react_llm_timeout_sets_reason() -> None:
    class _SlowClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            time.sleep(0.05)
            return ChatResponse(
                content="late",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _SlowClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    prompt = PromptConfig(system_prompt="{objective}")
    agent_cfg = BaseAgentConfig(model="agent", temperature=0.0, prompts_file="")

    limits = _limits(3)
    limits.worker_timeout_seconds = 0.01

    graph = build_single_agent_graph(
        prompt_config=prompt,
        agent_config=agent_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        client=client,
    )
    state = get_initial_state_for_architecture("single_react", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") == "llm_timeout"


def test_reasoner_executor_llm_timeout_sets_reason() -> None:
    class _SlowReasonerClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            time.sleep(0.05)
            return ChatResponse(
                content="late",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _SlowReasonerClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(system_prompt="{objective}", instruction_template="{worker_report}")
    reasoner_cfg = BaseAgentConfig(model="reasoner", temperature=0.0, prompts_file="")
    executor_cfg = BaseAgentConfig(model="executor", temperature=0.0, prompts_file="")

    limits = _limits(3)
    limits.worker_timeout_seconds = 0.01

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") == "llm_timeout"


def test_orchestrator_workers_llm_timeout_sets_reason() -> None:
    class _SlowOrchClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            time.sleep(0.05)
            return ChatResponse(
                content=json.dumps({"state_update": {}, "analysis": "late", "tasks": []}),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _SlowOrchClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    limits = _limits(3)
    limits.worker_timeout_seconds = 0.01

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state.get("termination_reason") == "llm_timeout"


def test_orchestrator_critic_workers_llm_timeout_sets_reason() -> None:
    class _SlowOrchCriticClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            time.sleep(0.05)
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            return ChatResponse(
                content=json.dumps({"state_update": {}, "analysis": "late", "proposed_plan": {}}),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    client = _SlowOrchCriticClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    limits = _limits(3)
    limits.worker_timeout_seconds = 0.01

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 150})

    assert final_state.get("termination_reason") == "llm_timeout"


def test_reasoner_executor_executor_timeout_nonfatal() -> None:
    class _SlowExecutorClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "reasoner":
                cmd = {"thought": "run", "command": {"type": "bash", "script": "ls"}}
                return ChatResponse(
                    content=json.dumps(cmd),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "executor":
                time.sleep(0.05)
                return ChatResponse(
                    content="late",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _SlowExecutorClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(system_prompt="{objective}", instruction_template="{worker_report}")
    reasoner_cfg = BaseAgentConfig(model="reasoner", temperature=0.0, prompts_file="")
    executor_cfg = BaseAgentConfig(model="executor", temperature=0.0, prompts_file="")

    limits = _limits(3)
    limits.worker_timeout_seconds = 0.01

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") in {"", "turn_limit"}
    # Executor timeout returns a synthetic execution result message.
    messages = final_state.get("messages", [])
    assert any(
        "EXECUTOR TIMED OUT" in (get_message_attr(m, "content", "") or "")
        for m in messages
    )


def test_orchestrator_workers_worker_timeout_reported(caplog) -> None:
    class _SlowWorkerClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                payload = {
                    "state_update": {},
                    "analysis": "plan",
                    "tasks": [
                        {"description": "inspect", "type": "explore", "role": "Worker"}
                    ],
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                time.sleep(0.05)
                return ChatResponse(
                    content="late",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _SlowWorkerClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    limits = _limits(3)
    limits.worker_timeout_seconds = 0.01

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert any(
        "timed out" in rec.message.lower()
        for rec in caplog.records
        if "orchestrator_workers" in rec.name
    )


def test_orchestrator_workers_worker_reports_reducer_resets() -> None:
    from MaLO.graphs.orchestrator_utils import reduce_worker_reports

    assert reduce_worker_reports(["a"], ["b"]) == ["a", "b"]
    assert reduce_worker_reports(["a"], []) == []
    assert reduce_worker_reports([], ["b"]) == ["b"]


def test_orchestrator_workers_tool_timeout_nonfatal() -> None:
    class _ToolTimeoutExecutor(ToolExecutor):
        def __init__(self) -> None:
            super().__init__(env_manager=None, truncation_chars=1000, goal_validator=None)

        def execute(
            self, tool_name: str, arguments: dict[str, Any], timeout_seconds: int | None = None
        ) -> ToolResult:  # type: ignore[override]
            return ToolResult(
                output="timed out",
                success=False,
                terminate=True,
                termination_reason="tool_timeout",
            )

    class _ToolTimeoutWorkerClient:
        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            if model == "orch":
                payload = {
                    "state_update": {},
                    "analysis": "plan",
                    "tasks": [
                        {"description": "inspect", "type": "explore", "role": "Worker"}
                    ],
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "execute_script",
                            json.dumps({"script": "echo hi", "language": "bash"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="run",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _ToolTimeoutWorkerClient()
    tool_executor = _ToolTimeoutExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    limits = _limits(3)

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 100})

    assert final_state.get("termination_reason") == "turn_limit"


def test_orchestrator_critic_workers_tool_timeout_nonfatal() -> None:
    class _ToolTimeoutExecutor(ToolExecutor):
        def __init__(self) -> None:
            super().__init__(env_manager=None, truncation_chars=1000, goal_validator=None)

        def execute(
            self, tool_name: str, arguments: dict[str, Any], timeout_seconds: int | None = None
        ) -> ToolResult:  # type: ignore[override]
            return ToolResult(
                output="timed out",
                success=False,
                terminate=True,
                termination_reason="tool_timeout",
            )

    class _ToolTimeoutWorkerClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orchestrator":
                payload = {
                    "state_update": {},
                    "analysis": "plan",
                    "proposed_plan": {
                        "tasks": [
                            {"description": "inspect", "type": "explore", "role": "Worker"}
                        ]
                    },
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "execute_script",
                            json.dumps({"script": "echo hi", "language": "bash"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="run",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _ToolTimeoutWorkerClient()
    tool_executor = _ToolTimeoutExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orchestrator", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    limits = _limits(3)

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 150})

    assert final_state.get("termination_reason") == "turn_limit"


def test_orchestrator_critic_workers_worker_reports_reducer_resets() -> None:
    from MaLO.graphs.orchestrator_utils import reduce_worker_reports

    assert reduce_worker_reports(["a"], ["b"]) == ["a", "b"]
    assert reduce_worker_reports(["a"], []) == []
    assert reduce_worker_reports([], ["b"]) == ["b"]


def test_orchestrator_critic_workers_worker_timeout_reported(caplog) -> None:
    class _SlowWorkerCriticClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                payload = {
                    "state_update": {},
                    "analysis": "plan",
                    "proposed_plan": {
                        "objective": "inspect",
                        "tasks": [
                            {
                                "description": "inspect",
                                "type": "explore",
                                "role": "Worker",
                            }
                        ],
                    },
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                time.sleep(0.05)
                return ChatResponse(
                    content="late",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _SlowWorkerCriticClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    limits = _limits(2)
    limits.worker_timeout_seconds = 0.01

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 150})

    assert any(
        "timed out" in rec.message.lower()
        for rec in caplog.records
        if "orchestrator_critic_workers" in rec.name
    )


def test_orchestrator_workers_worker_timeout_not_emitted_when_fast(
    caplog,
) -> None:
    class _FastWorkerClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                payload = {
                    "state_update": {},
                    "analysis": "plan",
                    "tasks": [
                        {"description": "inspect", "type": "explore", "role": "Worker"}
                    ],
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "execute_script",
                            json.dumps({"script": "ls", "language": "bash"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="work",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _FastWorkerClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    limits = _limits(2)
    limits.worker_timeout_seconds = 1

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    graph.invoke(state, config={"recursion_limit": 100})

    assert all(
        "timed out" not in rec.message.lower()
        for rec in caplog.records
        if "orchestrator_workers" in rec.name
    )


def test_orchestrator_critic_workers_worker_timeout_not_emitted_when_fast(
    caplog,
) -> None:
    class _FastWorkerCriticClient:
        def __init__(self) -> None:
            self._count = 0

        def generate(
            self,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float,
            tools: list[dict[str, Any]] | None = None,
            timeout: float | None = None,
            seed: int | None = None,
        ) -> ChatResponse:
            self._count += 1
            if model == "orch":
                payload = {
                    "state_update": {},
                    "analysis": "plan",
                    "proposed_plan": {
                        "objective": "inspect",
                        "tasks": [
                            {
                                "description": "inspect",
                                "type": "explore",
                                "role": "Worker",
                            }
                        ],
                    },
                }
                return ChatResponse(
                    content=json.dumps(payload),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "critic":
                verdict = {"verdict": "APPROVE", "feedback": "ok"}
                return ChatResponse(
                    content=json.dumps(verdict),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "worker":
                tool_calls = [
                    _ToolCall(
                        "1",
                        _ToolFunction(
                            "execute_script",
                            json.dumps({"script": "ls", "language": "bash"}),
                        ),
                    )
                ]
                return ChatResponse(
                    content="work",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _FastWorkerCriticClient()
    tool_executor = ToolExecutor(env_manager=None, truncation_chars=1000, goal_validator=None)
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch", temperature=0.0, prompts_file="")
    critic_cfg = BaseAgentConfig(model="critic", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker", temperature=0.0, prompts_file="")

    limits = _limits(2)
    limits.worker_timeout_seconds = 1

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=limits,
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    graph.invoke(state, config={"recursion_limit": 150})

    assert all(
        "timed out" not in rec.message.lower()
        for rec in caplog.records
        if "orchestrator_critic_workers" in rec.name
    )
