from __future__ import annotations

from typing import Any

from MaLO.config.models import BaseAgentConfig, LimitsConfig, PromptConfig
from MaLO.graphs.state import WorkerState as WorkerStateBase
from MaLO.graphs.worker_subgraph import build_worker_subgraph
from MaLO.tools.executor import ToolExecutor
from MaLO.utils.openrouter_client import ChatResponse


class _FakeToolExecutor(ToolExecutor):
    def __init__(self) -> None:
        super().__init__(env_manager=None, truncation_chars=1000, goal_validator=None)


class _CapturingClient:
    def __init__(self) -> None:
        self.last_messages: list[dict[str, Any]] | None = None

    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        self.last_messages = list(messages)
        return ChatResponse(
            content="ok",
            tool_calls=[],
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


class WorkerState(WorkerStateBase):
    """State for worker subgraph tests."""


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


def _run_worker_with_task(task: dict[str, Any], *, task_desc_keys: tuple[str, ...], default_instruction_uses_task_content: bool) -> str:
    client = _CapturingClient()
    tool_executor = _FakeToolExecutor()
    prompt = PromptConfig(system_prompt="{objective}")
    worker_cfg = BaseAgentConfig(model="test-worker", temperature=0.0, prompts_file="")

    compiled_worker, _node = build_worker_subgraph(
        worker_state_type=WorkerState,
        worker_prompt=prompt,
        worker_config=worker_cfg,
        limits_config=_limits(),
        tool_executor=tool_executor,
        worker_client=client,
        console_handler=None,
        task_desc_keys=task_desc_keys,
        default_instruction_uses_task_content=default_instruction_uses_task_content,
        suppress_tool_termination=False,
    )

    state = {
        "objective": "obj",
        "task": task,
        "worker_id": 1,
        "worker_role": "Security Specialist",
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
    }

    compiled_worker.invoke(state, config={"recursion_limit": 20})
    assert client.last_messages is not None
    return client.last_messages[-1]["content"]


def test_worker_instruction_default_uses_desc_details() -> None:
    task = {"description": "scan", "details": "do x"}
    content = _run_worker_with_task(
        task,
        task_desc_keys=("description",),
        default_instruction_uses_task_content=False,
    )
    assert content == "## Assigned Task\n\nscan\n\ndo x"


def test_worker_instruction_default_uses_task_content_and_script_fallback() -> None:
    task = {"script": "echo hi", "details": "more"}
    content = _run_worker_with_task(
        task,
        task_desc_keys=("description", "script"),
        default_instruction_uses_task_content=True,
    )
    assert content == "## Assigned Task\n\n**Task:** echo hi\n\n**Details:** more"
