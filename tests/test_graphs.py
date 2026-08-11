from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from MaLO.config.models import BaseAgentConfig, CriticConfig, LimitsConfig, PromptConfig
from MaLO.graphs import (
    build_orchestrator_critic_workers_graph,
    build_orchestrator_workers_graph,
    build_reasoner_executor_graph,
    build_single_agent_graph,
)
from MaLO.graphs.utils import get_message_attr, message_to_dict, parse_tool_call_args
from MaLO.run_helpers import get_initial_state_for_architecture
from MaLO.tools.core_tools import ToolResult
from MaLO.tools.executor import ToolExecutor
from MaLO.graphs.utils import execute_tool_calls, normalize_tool_calls
from MaLO.utils.openrouter_client import ChatResponse


@dataclass
class _ToolFunction:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _ToolFunction


class _FakeToolExecutor(ToolExecutor):
    def __init__(self) -> None:
        super().__init__(
            env_manager=None,
            truncation_chars=5000,
            goal_validator=None,
        )

    def execute(self, tool_name: str, arguments: dict[str, Any], timeout_seconds: int | None = None) -> ToolResult:  # type: ignore[override]
        if tool_name == "execute_script":
            return ToolResult(output="ok", success=True)
        if tool_name == "submit_goal":
            return ToolResult(output="flag accepted", success=True, terminate=True, termination_reason="success")
        if tool_name == "give_up":
            return ToolResult(output="gave up", success=False, terminate=True, termination_reason="give_up")
        if tool_name == "task_complete":
            return ToolResult(output=arguments.get("findings", "done"), success=True, terminate=True, termination_reason="task_complete")
        return ToolResult(output=f"Unknown tool: {tool_name}", success=False)


class _FakeClient:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:  # noqa: D401
        count = self._counts.get(model, 0) + 1
        self._counts[model] = count

        if model == "test-agent":
            if count == 1:
                return ChatResponse(
                    content="run",
                    tool_calls=[_ToolCall("1", _ToolFunction("execute_script", json.dumps({"script": "ls", "language": "bash"})))],
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            return ChatResponse(
                content="submit",
                tool_calls=[_ToolCall("2", _ToolFunction("submit_goal", json.dumps({"answer": "flag"})))],
                finish_reason="tool_calls",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

        if model == "test-reasoner":
            if count == 1:
                command = {"thought": "check", "command": {"type": "bash", "script": "ls"}}
            else:
                command = {"thought": "submit", "command": {"type": "submit", "answer": "flag"}}
            return ChatResponse(
                content=json.dumps(command),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

        if model == "test-executor":
            # Emit tool calls for the reasoner command.
            if count == 1:
                return ChatResponse(
                    content="exec",
                    tool_calls=[_ToolCall("1", _ToolFunction("execute_script", json.dumps({"script": "ls", "language": "bash"})))],
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            return ChatResponse(
                content="submit",
                tool_calls=[_ToolCall("2", _ToolFunction("submit_goal", json.dumps({"answer": "flag"})))],
                finish_reason="tool_calls",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

        if model == "test-orchestrator":
            if count == 1:
                plan = {
                    "state_update": {"phase": "reconnaissance"},
                    "analysis": "plan",
                    "tasks": [
                        {"description": "inspect", "type": "explore", "role": "Security Specialist"}
                    ],
                }
            else:
                plan = {
                    "state_update": {"phase": "analysis"},
                    "analysis": "submit",
                    "tasks": [
                        {"description": "submit", "type": "submit", "answer": "flag"}
                    ],
                }
            return ChatResponse(
                content=json.dumps(plan),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

        if model == "test-critic":
            verdict = {"verdict": "APPROVE", "feedback": "ok"}
            return ChatResponse(
                content=json.dumps(verdict),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

        if model == "test-worker":
            if count % 2 == 1:
                return ChatResponse(
                    content="work",
                    tool_calls=[_ToolCall("1", _ToolFunction("execute_script", json.dumps({"script": "ls", "language": "bash"})))],
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            return ChatResponse(
                content="done",
                tool_calls=[_ToolCall("2", _ToolFunction("task_complete", json.dumps({"findings": "ok"})))],
                finish_reason="tool_calls",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

        raise AssertionError(f"Unexpected model: {model}")


def _limits() -> LimitsConfig:
    return LimitsConfig(
        max_turns=5,
        max_cost_usd=1.0,
        observation_truncation_chars=1000,
        recursion_limit=100,
        max_workers=1,
        worker_timeout_seconds=5,
        challenge_timeout_seconds=30,
    )


def test_single_react_graph() -> None:
    client = _FakeClient()
    tool_executor = _FakeToolExecutor()
    prompt = PromptConfig(system_prompt="{objective}")
    agent_cfg = BaseAgentConfig(model="test-agent", temperature=0.0, prompts_file="")

    graph = build_single_agent_graph(
        prompt_config=prompt,
        agent_config=agent_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        client=client,
    )
    state = get_initial_state_for_architecture("single_react", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") == "success"
    assert final_state.get("tool_call_count", 0) >= 2


def test_reasoner_executor_graph() -> None:
    client = _FakeClient()
    tool_executor = _FakeToolExecutor()
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(system_prompt="{objective}", instruction_template="{worker_report}")
    reasoner_cfg = BaseAgentConfig(model="test-reasoner", temperature=0.0, prompts_file="")
    executor_cfg = BaseAgentConfig(model="test-executor", temperature=0.0, prompts_file="")

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("termination_reason") == "success"
    assert final_state.get("tool_call_count", 0) >= 2


def test_reasoner_executor_token_totals() -> None:
    client = _FakeClient()
    tool_executor = _FakeToolExecutor()
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(
        system_prompt="{objective}", instruction_template="{worker_report}"
    )
    reasoner_cfg = BaseAgentConfig(
        model="test-reasoner", temperature=0.0, prompts_file=""
    )
    executor_cfg = BaseAgentConfig(
        model="test-executor", temperature=0.0, prompts_file=""
    )

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    assert final_state.get("prompt_tokens") == 4
    assert final_state.get("completion_tokens") == 4
    assert final_state.get("total_tokens") == 8


def test_orchestrator_workers_graph() -> None:
    client = _FakeClient()
    tool_executor = _FakeToolExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="test-orchestrator", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="test-worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 200})

    assert final_state.get("termination_reason") == "success"
    assert "worker_report" in final_state


def test_orchestrator_workers_token_totals_multi_turn_worker() -> None:
    client = _FakeClient()
    tool_executor = _FakeToolExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="test-orchestrator", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="test-worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        orchestrator_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 200})

    assert final_state.get("prompt_tokens") == 4
    assert final_state.get("completion_tokens") == 4
    assert final_state.get("total_tokens") == 8


def test_orchestrator_workers_graph_separate_clients() -> None:
    class _OrchestratorClient:
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
            assert model == "orch-model"
            self.calls += 1
            if self.calls == 1:
                plan = {
                    "state_update": {"phase": "recon"},
                    "analysis": "plan",
                    "tasks": [
                        {
                            "description": "inspect",
                            "type": "explore",
                            "role": "Security Specialist",
                        }
                    ],
                }
            else:
                plan = {
                    "state_update": {"phase": "analysis"},
                    "analysis": "submit",
                    "tasks": [
                        {"description": "submit", "type": "submit", "answer": "flag"}
                    ],
                }
            return ChatResponse(
                content=json.dumps(plan),
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    class _WorkerClient:
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
            assert model == "worker-model"
            self.calls += 1
            if self.calls % 2 == 1:
                return ChatResponse(
                    content="work",
                    tool_calls=[
                        _ToolCall(
                            "1",
                            _ToolFunction(
                                "execute_script",
                                json.dumps(
                                    {"script": "ls", "language": "bash"}
                                ),
                            ),
                        )
                    ],
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            return ChatResponse(
                content="done",
                tool_calls=[
                    _ToolCall(
                        "2",
                        _ToolFunction(
                            "task_complete",
                            json.dumps({"findings": "ok"}),
                        ),
                    )
                ],
                finish_reason="tool_calls",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    orchestrator_client = _OrchestratorClient()
    worker_client = _WorkerClient()
    tool_executor = _FakeToolExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="orch-model", temperature=0.0, prompts_file="")
    worker_cfg = BaseAgentConfig(model="worker-model", temperature=0.0, prompts_file="")

    graph = build_orchestrator_workers_graph(
        orchestrator_prompt=orch_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        orchestrator_client=orchestrator_client,
        worker_client=worker_client,
    )
    state = get_initial_state_for_architecture("orchestrator_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 200})

    assert orchestrator_client.calls >= 2
    assert worker_client.calls >= 2
    assert final_state.get("termination_reason") == "success"


def test_orchestrator_critic_workers_graph() -> None:
    client = _FakeClient()
    tool_executor = _FakeToolExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="test-orchestrator", temperature=0.0, prompts_file="")
    critic_cfg = CriticConfig(model="test-critic", temperature=0.0, prompts_file="", strictness="high")
    worker_cfg = BaseAgentConfig(model="test-worker", temperature=0.0, prompts_file="")

    graph = build_orchestrator_critic_workers_graph(
        orchestrator_prompt=orch_prompt,
        critic_prompt=critic_prompt,
        worker_prompt=worker_prompt,
        orchestrator_config=orch_cfg,
        critic_config=critic_cfg,
        worker_config=worker_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        orchestrator_client=client,
        critic_client=client,
        worker_client=client,
    )
    state = get_initial_state_for_architecture("orchestrator_critic_workers", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 300})

    assert final_state.get("termination_reason") in {"success", "", "turn_limit"}
    assert "worker_report" in final_state


def test_orchestrator_critic_workers_token_totals() -> None:
    client = _FakeClient()
    tool_executor = _FakeToolExecutor()
    orch_prompt = PromptConfig(system_prompt="{objective}")
    critic_prompt = PromptConfig(system_prompt="{objective}")
    worker_prompt = PromptConfig(system_prompt="{objective}")
    orch_cfg = BaseAgentConfig(model="test-orchestrator", temperature=0.0, prompts_file="")
    critic_cfg = CriticConfig(
        model="test-critic", temperature=0.0, prompts_file="", strictness="high"
    )
    worker_cfg = BaseAgentConfig(model="test-worker", temperature=0.0, prompts_file="")

    limits = _limits().model_copy(update={"worker_max_turns": 2})

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
    final_state = graph.invoke(state, config={"recursion_limit": 300})

    assert final_state.get("prompt_tokens") == 6
    assert final_state.get("completion_tokens") == 6
    assert final_state.get("total_tokens") == 12


def test_reasoner_executor_fallback_tool_messages_have_ids() -> None:
    class _NoToolExecutorClient:
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
            if model == "test-reasoner":
                command = {
                    "thought": "run",
                    "command": {"type": "bash", "script": "ls"},
                }
                return ChatResponse(
                    content=json.dumps(command),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "test-executor":
                return ChatResponse(
                    content="no tools",
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _NoToolExecutorClient()
    tool_executor = _FakeToolExecutor()
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(
        system_prompt="{objective}", instruction_template="{worker_report}"
    )
    reasoner_cfg = BaseAgentConfig(
        model="test-reasoner", temperature=0.0, prompts_file=""
    )
    executor_cfg = BaseAgentConfig(
        model="test-executor", temperature=0.0, prompts_file=""
    )

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    final_state = graph.invoke(state, config={"recursion_limit": 50})

    tool_messages = [
        message_to_dict(m)
        for m in final_state.get("messages", [])
        if message_to_dict(m).get("role") == "tool"
    ]
    assert tool_messages
    assert all(m.get("tool_call_id") for m in tool_messages)


def test_execute_tool_calls_adds_tool_call_id_when_missing() -> None:
    tool_executor = _FakeToolExecutor()
    tool_calls = [
        {
            "id": "",
            "type": "function",
            "function": {
                "name": "execute_script",
                "arguments": json.dumps({"script": "ls", "language": "bash"}),
            },
        }
    ]

    result = execute_tool_calls(
        tool_calls=tool_calls,
        tool_executor=tool_executor,
        truncation_chars=1000,
    )

    assert result.messages
    assert result.messages[0].get("tool_call_id")


def test_parse_tool_call_args_normalizes_prefixed_tool_name() -> None:
    tc = {
        "id": "toolcall_1",
        "function": {
            "name": "tool_execute_script_abc123",
            "arguments": json.dumps({"script": "ls", "language": "bash"}),
        },
    }
    _, tool_name, args = parse_tool_call_args(tc)

    assert tool_name == "execute_script"
    assert args["language"] == "bash"


def test_normalize_tool_calls_handles_langchain_dict() -> None:
    tool_calls = [
        {"id": "1", "name": "execute_script", "args": {"script": "ls", "language": "bash"}}
    ]
    normalized = normalize_tool_calls(tool_calls)
    assert normalized[0]["function"]["name"] == "execute_script"
    assert json.loads(normalized[0]["function"]["arguments"]) == {
        "script": "ls",
        "language": "bash",
    }


def test_extract_json_from_response_fenced_block() -> None:
    from MaLO.graphs.utils import extract_json_from_response

    content = "```\\n{'tasks': [{'type': 'submit'}]}\\n```"
    parsed = extract_json_from_response(content)
    assert parsed == {"tasks": [{"type": "submit"}]}


def test_reasoner_executor_does_not_emit_timeout_message_without_timeout(
    caplog,
) -> None:
    class _FastClient:
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
            if model == "test-reasoner":
                cmd = {
                    "thought": "run",
                    "command": {"type": "bash", "script": "ls"},
                }
                return ChatResponse(
                    content=json.dumps(cmd),
                    tool_calls=[],
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            if model == "test-executor":
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
                    content="exec",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                )
            raise AssertionError(f"Unexpected model: {model}")

    client = _FastClient()
    tool_executor = _FakeToolExecutor()
    reasoner_prompt = PromptConfig(system_prompt="{objective}")
    executor_prompt = PromptConfig(
        system_prompt="{objective}", instruction_template="{worker_report}"
    )
    reasoner_cfg = BaseAgentConfig(
        model="test-reasoner", temperature=0.0, prompts_file=""
    )
    executor_cfg = BaseAgentConfig(
        model="test-executor", temperature=0.0, prompts_file=""
    )

    graph = build_reasoner_executor_graph(
        reasoner_prompt=reasoner_prompt,
        executor_prompt=executor_prompt,
        reasoner_config=reasoner_cfg,
        executor_config=executor_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        reasoner_client=client,
        executor_client=client,
    )
    state = get_initial_state_for_architecture("reasoner_executor", "obj")
    graph.invoke(state, config={"recursion_limit": 50})

    assert all(
        "EXECUTOR TIMED OUT" not in rec.message
        for rec in caplog.records
        if "reasoner_executor" in rec.name
    )


def test_execute_tool_calls_stringifies_non_string_output() -> None:
    from MaLO.graphs.utils import execute_tool_calls
    from MaLO.observability.console_handler import ConsoleHandler
    from MaLO.tools.core_tools import ToolResult

    class _ListOutputExecutor:
        def execute(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            timeout_seconds: int | None = None,
        ) -> ToolResult:  # type: ignore[override]
            return ToolResult(output=["a", "b"], success=True)

    tool_calls = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "execute_script", "arguments": "{}"},
        }
    ]

    console = ConsoleHandler(verbose=True, quiet=False)
    from MaLO.observability.console_handler import AgentRole

    result = execute_tool_calls(
        tool_calls=tool_calls,
        tool_executor=_ListOutputExecutor(),
        truncation_chars=1000,
        console_handler=console,
        agent_role=AgentRole.AGENT,
        turn=1,
        max_turns=1,
        worker_id=None,
        timeout_seconds=None,
    )

    assert result.messages
    assert result.messages[0]["content"].startswith("[")
