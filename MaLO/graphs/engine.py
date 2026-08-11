"""Shared graph execution helpers."""

from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from MaLO.graphs.utils import (
    build_assistant_message,
    execute_tool_calls,
    get_message_attr,
    message_to_dict,
    token_count_deltas,
    update_token_counts,
)
from MaLO.observability.console_handler import AgentRole, ConsoleHandler
from MaLO.tools.executor import ToolExecutor
from MaLO.utils.llm_client import LLMClient


def run_with_timeout(
    fn: Callable[[], dict[str, Any]],
    timeout_seconds: int | float | None,
    timeout_result: dict[str, Any] | Callable[[], dict[str, Any]],
    log_message: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Runs a function with an optional wall-clock timeout.

    Args:
        fn: Function returning a state update dict.
        timeout_seconds: Timeout in seconds. If None or <= 0, no timeout.
        timeout_result: State update to return on timeout.
        log_message: Warning message to log on timeout.
        logger: Logger for warnings.

    Returns:
        State update dict from fn or timeout_result when timed out.
    """
    if timeout_seconds is None or timeout_seconds <= 0:
        return fn()

    ctx = contextvars.copy_context()
    executor = ThreadPoolExecutor(max_workers=1)
    timed_out = False
    try:
        future = executor.submit(lambda: ctx.run(fn))
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        timed_out = True
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        if log_message:
            logger.warning(log_message)
        if callable(timeout_result):
            return timeout_result()
        return timeout_result
    finally:
        if not timed_out:
            executor.shutdown(wait=True)


def ensure_system_message(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Ensures a system message is present as the first message."""
    result = list(messages)
    if not result or get_message_attr(result[0], "role") != "system":
        result.insert(0, {"role": "system", "content": system_prompt})
    return result


def _sanitize_tool_message_order(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drops tool messages that are not adjacent to a tool-calling assistant."""
    sanitized: list[dict[str, Any]] = []
    in_tool_response = False
    expected_tool_ids: set[str] = set()

    for message in messages:
        role = get_message_attr(message, "role")
        if role == "assistant":
            tool_calls = get_message_attr(message, "tool_calls") or []
            if tool_calls:
                expected_tool_ids = {
                    get_message_attr(tc, "id", "") for tc in tool_calls
                }
                expected_tool_ids.discard("")
                in_tool_response = True
            else:
                in_tool_response = False
                expected_tool_ids = set()
            sanitized.append(message)
            continue

        if role == "tool":
            if not in_tool_response:
                continue
            tool_call_id = get_message_attr(message, "tool_call_id", "")
            if expected_tool_ids and tool_call_id not in expected_tool_ids:
                continue
            sanitized.append(message)
            continue

        in_tool_response = False
        expected_tool_ids = set()
        sanitized.append(message)

    return sanitized


def to_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converts messages to OpenAI-compatible dicts."""
    sanitized = _sanitize_tool_message_order(messages)
    return [message_to_dict(m) for m in sanitized]


def execute_tools_update(
    state: dict[str, Any],
    tool_calls: list[Any],
    tool_executor: ToolExecutor,
    truncation_chars: int,
    console_handler: ConsoleHandler | None,
    agent_role: AgentRole,
    turn: int,
    max_turns: int,
    worker_id: int | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Executes tool calls and returns a state update dict."""
    result = execute_tool_calls(
        tool_calls=tool_calls,
        tool_executor=tool_executor,
        truncation_chars=truncation_chars,
        console_handler=console_handler,
        agent_role=agent_role,
        turn=turn,
        max_turns=max_turns,
        worker_id=worker_id,
        timeout_seconds=timeout_seconds,
    )
    new_tool_call_count = (
        state.get("tool_call_count", 0) + result.calls_executed
    )
    return {
        "messages": result.messages,
        "tool_call_count": new_tool_call_count,
        "done": result.done,
        "termination_reason": result.termination_reason,
        "needs_human": result.needs_human,
    }


def run_llm_call(
    *,
    client: LLMClient,
    model: str,
    temperature: float,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_seconds: int | float | None,
    seed: int | None,
    max_message_chars: int | None,
    console_handler: ConsoleHandler | None,
    agent_role: AgentRole,
    turn: int,
    max_turns: int,
    worker_id: int | None = None,
    instruction: str | None = None,
    token_update_mode: str = "cumulative",
    state: dict[str, Any] | None = None,
    log_response: bool = True,
) -> dict[str, Any]:
    """Runs a single LLM call with shared prompt/message handling.

    Returns a dict with assistant_message, response, and token_updates.
    """
    if console_handler:
        console_handler.on_agent_thinking(
            turn=turn,
            max_turns=max_turns,
            agent=agent_role,
            worker_id=worker_id,
        )

    prepared_messages = ensure_system_message(list(messages), system_prompt)
    if instruction:
        prepared_messages.append({"role": "user", "content": instruction})

    api_messages = to_api_messages(prepared_messages)

    response = client.generate(
        model=model,
        messages=api_messages,
        temperature=temperature,
        tools=tools,
        timeout=timeout_seconds,
        seed=seed,
    )

    assistant_message = build_assistant_message(
        response,
        max_chars=max_message_chars,
    )

    if token_update_mode == "delta":
        token_updates = token_count_deltas(response)
    else:
        if state is None:
            raise ValueError("state is required for cumulative token updates")
        token_updates = update_token_counts(state, response)

    if console_handler and log_response:
        console_handler.on_agent_response(
            _turn=turn,
            _max_turns=max_turns,
            agent=agent_role,
            content=response.content,
            _has_tool_calls=bool(response.tool_calls),
            _worker_id=worker_id,
        )

    return {
        "assistant_message": assistant_message,
        "response": response,
        "token_updates": token_updates,
    }


def turn_limit_update() -> dict[str, Any]:
    """Returns a state update for turn-limit termination."""
    return {"termination_reason": "turn_limit"}
