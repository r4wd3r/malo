from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from MaLO.graphs.engine import run_llm_call
from MaLO.observability.console_handler import AgentRole
from MaLO.utils.openrouter_client import ChatResponse


@dataclass
class _FakeClient:
    response: ChatResponse

    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        return self.response


def test_run_llm_call_cumulative_tokens() -> None:
    response = ChatResponse(
        content="ok",
        tool_calls=[],
        finish_reason="stop",
        prompt_tokens=4,
        completion_tokens=5,
        total_tokens=9,
    )
    client = _FakeClient(response=response)
    state = {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }

    result = run_llm_call(
        client=client,
        model="test",
        temperature=0.0,
        system_prompt="sys",
        messages=[],
        tools=None,
        timeout_seconds=None,
        seed=None,
        max_message_chars=None,
        console_handler=None,
        agent_role=AgentRole.AGENT,
        turn=1,
        max_turns=3,
        token_update_mode="cumulative",
        state=state,
    )

    assert result["assistant_message"]["content"] == "ok"
    assert result["token_updates"] == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }


def test_run_llm_call_delta_tokens() -> None:
    response = ChatResponse(
        content="ok",
        tool_calls=[],
        finish_reason="stop",
        prompt_tokens=4,
        completion_tokens=5,
        total_tokens=9,
    )
    client = _FakeClient(response=response)

    result = run_llm_call(
        client=client,
        model="test",
        temperature=0.0,
        system_prompt="sys",
        messages=[],
        tools=None,
        timeout_seconds=None,
        seed=None,
        max_message_chars=None,
        console_handler=None,
        agent_role=AgentRole.AGENT,
        turn=1,
        max_turns=3,
        token_update_mode="delta",
    )

    assert result["token_updates"] == {
        "prompt_tokens": 4,
        "completion_tokens": 5,
        "total_tokens": 9,
    }
