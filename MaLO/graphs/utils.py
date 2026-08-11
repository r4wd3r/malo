"""Shared utilities for LangGraph implementations."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from MaLO.observability.console_handler import AgentRole, ConsoleHandler
from MaLO.tools.core_tools import ToolResult
from MaLO.tools.executor import ToolExecutor
from MaLO.utils.openrouter_client import ChatResponse
from MaLO.utils.text_utils import truncate_text


# -----------------------------------------------------------------------------
# Message Utilities
# -----------------------------------------------------------------------------


def get_message_attr(
    message: dict[str, Any] | BaseMessage,
    key: str,
    default: Any = None,
) -> Any:
    """Gets an attribute from a message (dict or LangChain object).

    Args:
        message: Message dict or LangChain message object.
        key: Attribute/key to retrieve.
        default: Default value if not found.

    Returns:
        The attribute value or default.
    """
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def get_tool_calls(message: dict[str, Any] | BaseMessage) -> list[Any]:
    """Extracts tool calls from a message (normalized when possible)."""
    if isinstance(message, dict):
        return normalize_tool_calls(message.get("tool_calls", []))
    if isinstance(message, AIMessage):
        return normalize_tool_calls(message.tool_calls or [])
    return []


def normalize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Normalizes tool calls into OpenAI-compatible dicts.

    Accepts OpenAI dicts, LangChain ToolCall dicts, or ToolCall objects.
    """
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc_id, tool_name, arguments = parse_tool_call_args(tc)
        normalized.append(
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
        )
    return normalized


def message_to_dict(message: dict[str, Any] | BaseMessage) -> dict[str, Any]:
    """Converts a LangChain message to OpenAI-compatible dict format.

    Args:
        message: Message dict or LangChain message object.

    Returns:
        Dict with role, content, and optionally tool_calls/tool_call_id.
    """
    # Role mapping from LangChain types to OpenAI roles
    role_map = {"human": "user", "ai": "assistant"}

    if isinstance(message, dict):
        # Normalize role if present
        role = message.get("role", "user")
        normalized = {**message, "role": role_map.get(role, role)}
        return normalized

    result: dict[str, Any] = {
        "role": "user",  # Default, will be overridden below
        "content": message.content or "",
    }

    # Map LangChain message types to OpenAI roles
    if isinstance(message, AIMessage):
        result["role"] = "assistant"
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.get("id", "") if isinstance(tc, dict) else tc.id,
                    "type": "function",
                    "function": {
                        "name": (
                            tc.get("name", "")
                            if isinstance(tc, dict)
                            else tc.name
                        ),
                        "arguments": json.dumps(
                            tc.get("args", {})
                            if isinstance(tc, dict)
                            else tc.args
                        ),
                    },
                }
                for tc in message.tool_calls
            ]
    elif isinstance(message, HumanMessage):
        result["role"] = "user"
    elif isinstance(message, ToolMessage):
        result["role"] = "tool"
        result["tool_call_id"] = message.tool_call_id
    elif hasattr(message, "type"):
        # Fallback for other message types
        msg_type = message.type
        result["role"] = role_map.get(msg_type, msg_type)

    return result


# -----------------------------------------------------------------------------
# Prompt Formatting
# -----------------------------------------------------------------------------


def format_system_prompt(
    template: str,
    objective: str,
    **kwargs: Any,
) -> str:
    """Formats a system prompt template with variables.

    Args:
        template: System prompt template with {placeholders}.
        objective: Challenge objective (always available as {objective}).
        **kwargs: Additional template variables.

    Returns:
        Formatted system prompt.
    """
    formatted = template.replace("{objective}", objective)
    for key, value in kwargs.items():
        formatted = formatted.replace(f"{{{key}}}", str(value))
    return formatted


# -----------------------------------------------------------------------------
# Response Building
# -----------------------------------------------------------------------------


def build_assistant_message(
    response: ChatResponse,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Builds an assistant message dict from a ChatResponse.

    Args:
        response: ChatResponse from the LLM client.

    Returns:
        OpenAI-compatible assistant message dict.
    """
    content = response.content or ""
    if max_chars is not None and max_chars > 0 and len(content) > max_chars:
        content = f"{content[:max_chars]}..."
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if response.tool_calls:
        message["tool_calls"] = normalize_tool_calls(response.tool_calls)
    return message


def token_count_deltas(response: ChatResponse) -> dict[str, int]:
    """Returns token count deltas for a single response.

    Use when state uses operator.add for token fields (e.g. orchestrator
    graphs with fan-in) so the reducer does not double-count.

    Args:
        response: ChatResponse from the LLM client.

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens (deltas).
    """
    return {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def update_token_counts(
    state: dict[str, Any],
    response: ChatResponse,
) -> dict[str, int]:
    """Updates token counts from a response (cumulative).

    Use for graphs that merge state by replacement. For state with
    Annotated[int, operator.add], use token_count_deltas() instead.

    Args:
        state: Current graph state.
        response: ChatResponse from the LLM client.

    Returns:
        Dict with updated cumulative token counts.
    """
    return {
        "prompt_tokens": (
            state.get("prompt_tokens", 0) + response.prompt_tokens
        ),
        "completion_tokens": (
            state.get("completion_tokens", 0) + response.completion_tokens
        ),
        "total_tokens": (
            state.get("total_tokens", 0) + response.total_tokens
        ),
    }


# -----------------------------------------------------------------------------
# Tool Execution
# -----------------------------------------------------------------------------


@dataclass
class ToolExecutionResult:
    """Result of executing tool calls."""

    messages: list[dict[str, Any]]
    calls_executed: int
    done: bool
    termination_reason: str
    needs_human: bool


def _normalize_tool_name(tool_name: str) -> str:
    if not tool_name:
        return tool_name
    if tool_name.startswith("tool_"):
        tool_name = tool_name[5:]
    for base in (
        "execute_script",
        "submit_goal",
        "give_up",
        "task_complete",
        "request_human_help",
    ):
        if tool_name == base or tool_name.startswith(f"{base}_"):
            return base
    return tool_name


def parse_tool_call_args(tc: Any) -> tuple[str, str, dict[str, Any]]:
    """Parses a tool call into its components.

    Args:
        tc: Tool call in OpenAI dict or LangChain format.

    Returns:
        Tuple of (tool_call_id, tool_name, arguments).
    """
    if isinstance(tc, dict):
        tc_id = tc.get("id", "")
        # Check for LangChain format first (name at top level)
        if "name" in tc:
            # LangChain dict format: {"id", "name", "args"}
            tool_name = tc.get("name", "")
            arguments = tc.get("args", {})
        else:
            # OpenAI format: {"id", "function": {"name", "arguments"}}
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}
    else:
        # LangChain ToolCall object with attributes
        tc_id = getattr(tc, "id", "")
        tool_name = getattr(tc, "name", "")
        arguments = getattr(tc, "args", {})
        if not tool_name and hasattr(tc, "function"):
            func = getattr(tc, "function", None)
            tool_name = getattr(func, "name", "") if func else ""
            args_str = getattr(func, "arguments", "") if func else ""
            if args_str:
                try:
                    arguments = json.loads(args_str)
                except json.JSONDecodeError:
                    arguments = {}

    tool_name = _normalize_tool_name(tool_name)

    if not tc_id:
        tc_id = "tool_call"
    return tc_id, tool_name, arguments


def execute_tool_calls(
    tool_calls: list[Any],
    tool_executor: ToolExecutor,
    truncation_chars: int,
    console_handler: ConsoleHandler | None = None,
    agent_role: AgentRole = AgentRole.AGENT,
    turn: int = 0,
    max_turns: int = 0,
    worker_id: int | None = None,
    timeout_seconds: int | None = None,
) -> ToolExecutionResult:
    """Executes a list of tool calls and returns results.

    Args:
        tool_calls: List of tool calls to execute.
        tool_executor: Tool executor instance.
        truncation_chars: Maximum characters for tool output truncation.
        console_handler: Optional console handler for logging.
        agent_role: Role of the agent executing tools.
        turn: Current turn number (for logging).
        max_turns: Maximum turns (for logging).
        worker_id: Optional worker ID (for logging).
        timeout_seconds: Optional timeout in seconds for tool execution.

    Returns:
        ToolExecutionResult with messages and status.
    """
    tool_messages: list[dict[str, Any]] = []
    done = False
    termination_reason = ""
    calls_executed = 0
    needs_human = False

    for index, tc in enumerate(tool_calls, start=1):
        tc_id, tool_name, arguments = parse_tool_call_args(tc)
        if not tc_id:
            tc_id = f"tool_call_{index}"

        # Log tool call start
        if console_handler:
            console_handler.on_tool_call(
                turn=turn,
                max_turns=max_turns,
                agent=agent_role,
                tool_name=tool_name,
                args=arguments,
                worker_id=worker_id,
            )

        result = tool_executor.execute(
            tool_name, arguments, timeout_seconds=timeout_seconds
        )
        calls_executed += 1

        # Determine success for submit_goal
        tool_success: bool | None = None
        if tool_name == "submit_goal":
            tool_success = result.termination_reason == "success"

        output_text = _stringify_tool_output(result.output)

        # Log tool result
        if console_handler:
            console_handler.on_tool_result(
                tool_name=tool_name,
                output=output_text,
                char_count=len(output_text),
                success=tool_success,
                agent=agent_role,
                _worker_id=worker_id,
            )

        # Apply truncation to prevent context exhaustion
        truncated_output = truncate_text(output_text, truncation_chars)
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": truncated_output,
        })

        if result.terminate:
            done = True
            termination_reason = result.termination_reason or "terminated"
        if result.needs_human:
            needs_human = True

    return ToolExecutionResult(
        messages=tool_messages,
        calls_executed=calls_executed,
        done=done,
        termination_reason=termination_reason,
        needs_human=needs_human,
    )


def _stringify_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        import json

        return json.dumps(output, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return str(output)


# -----------------------------------------------------------------------------
# JSON Parsing
# -----------------------------------------------------------------------------


def extract_json_from_response(content: str) -> dict[str, Any] | None:
    """Extracts JSON from an LLM response.

    Handles responses with markdown code blocks or raw JSON.

    Args:
        content: Response content from LLM.

    Returns:
        Parsed JSON dict, or None if parsing fails.
    """
    if not content:
        return None

    # Try to find JSON in code blocks (```json or generic ```).
    fenced_start = content.find("```")
    if fenced_start != -1:
        fence_line_end = content.find("\n", fenced_start)
        if fence_line_end != -1:
            fenced_body_start = fence_line_end + 1
            fenced_end = content.find("```", fenced_body_start)
            if fenced_end != -1:
                json_str = content[fenced_body_start:fenced_end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try Python-literal dicts (single quotes, trailing commas).
                    try:
                        parsed = ast.literal_eval(json_str)
                        if isinstance(parsed, dict):
                            return parsed
                    except (ValueError, SyntaxError):
                        pass

    # Try to find raw JSON (starts with {)
    brace_start = content.find("{")
    if brace_start != -1:
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(content[brace_start:], start=brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_str = content[brace_start : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        try:
                            parsed = ast.literal_eval(json_str)
                            if isinstance(parsed, dict):
                                return parsed
                        except (ValueError, SyntaxError):
                            break

    return None


def parse_typed_json(
    content: str,
    model: type[BaseModel],
) -> tuple[BaseModel | None, str | None]:
    """Extracts JSON and validates against a Pydantic model.

    Returns (model_instance, error_message). On failure, model_instance is None.
    """
    raw = extract_json_from_response(content)
    if raw is None:
        return None, "No JSON found"
    try:
        return model.model_validate(raw), None
    except ValidationError as err:
        return None, str(err)


# -----------------------------------------------------------------------------
# State Helpers
# -----------------------------------------------------------------------------


def check_turn_limit(state: dict[str, Any], max_turns: int) -> bool:
    """Checks if the turn limit has been reached.

    Args:
        state: Current graph state.
        max_turns: Maximum turns allowed.

    Returns:
        True if limit reached, False otherwise.
    """
    return state.get("turn_count", 0) >= max_turns


def should_terminate(state: dict[str, Any]) -> bool:
    """Checks if the graph should terminate.

    Args:
        state: Current graph state.

    Returns:
        True if done flag is set.
    """
    return state.get("done", False)
