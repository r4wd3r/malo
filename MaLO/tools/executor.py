"""Tool execution dispatcher for MaLO LangGraph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from MaLO.utils.weave_utils import get_weave

from MaLO.environment.environment_manager import EnvironmentManager
from MaLO.tools.core_tools import (
    ToolResult,
    execute_script_tool,
    give_up,
    request_human_help,
    submit_goal,
    task_complete,
)
from MaLO.validation.goal_validator import (
    GoalValidator,
)


@dataclass
class ToolExecutor:
    """Dispatches tool calls to their implementations."""

    env_manager: EnvironmentManager | None
    truncation_chars: int
    goal_validator: GoalValidator | None = None

    @get_weave().op()
    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: int | None = None,
    ) -> ToolResult:
        """Executes a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments as a dictionary.
            timeout_seconds: Optional timeout in seconds for tool execution.

        Returns:
            Result of the tool execution.
        """
        if tool_name == "execute_script":
            return self._execute_script(arguments, timeout_seconds=timeout_seconds)
        if tool_name == "submit_goal":
            return self._submit_goal(arguments)
        if tool_name == "give_up":
            return self._give_up(arguments)
        if tool_name == "task_complete":
            return self._task_complete(arguments)
        if tool_name == "request_human_help":
            return self._request_human_help(arguments)
        return ToolResult(
            output=f"Unknown tool: {tool_name}",
            success=False,
        )

    def _execute_script(
        self, arguments: dict[str, Any], timeout_seconds: int | None = None
    ) -> ToolResult:
        """Handles execute_script tool calls.

        Args:
            arguments: Tool arguments containing script and language.
            timeout_seconds: Optional timeout in seconds for script execution.

        Returns:
            Tool execution result.
        """
        if self.env_manager is None:
            return ToolResult(
                output="Environment not available (--skip-env mode).",
                success=False,
            )
        script = arguments.get("script", "")
        language = arguments.get("language", "bash")
        if language not in ("bash", "python"):
            language = "bash"
        return execute_script_tool(
            script=script,
            language=language,
            env_manager=self.env_manager,
            truncation_chars=self.truncation_chars,
            timeout_seconds=timeout_seconds,
        )

    def _submit_goal(self, arguments: dict[str, Any]) -> ToolResult:
        """Handles submit_goal tool calls."""
        answer = arguments.get("answer", "")
        # Try to parse as JSON list if it looks like one
        if isinstance(answer, str) and answer.startswith("["):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
        if self.goal_validator is None:
            return ToolResult(
                output="No goal validator configured for validation.",
                success=False,
                terminate=True,
                termination_reason="missing_goal",
            )

        is_valid, feedback = self.goal_validator.validate(answer)
        if is_valid:
            return ToolResult(
                output=feedback,
                success=True,
                terminate=True,
                termination_reason="success",
            )
        return ToolResult(
            output=f"{feedback} You may continue investigating.",
            success=False,
            terminate=False,
        )

    def _get_validator(self) -> GoalValidator | None:
        return self.goal_validator

    def _give_up(self, arguments: dict[str, Any]) -> ToolResult:
        """Handles give_up tool calls."""
        reason = arguments.get("reason", "No reason provided.")
        return give_up(reason)

    def _task_complete(self, arguments: dict[str, Any]) -> ToolResult:
        """Handles task_complete tool calls."""
        findings = arguments.get("findings", "Task completed.")
        return task_complete(findings)

    def _request_human_help(self, arguments: dict[str, Any]) -> ToolResult:
        """Handles request_human_help tool calls."""
        query = arguments.get("query", "")
        return request_human_help(query)


def parse_tool_call(
    tool_call: Any,
) -> tuple[str, str, dict[str, Any]]:
    """Parses a tool call from the OpenAI response format.

    Args:
        tool_call: Tool call object from the API response.

    Returns:
        Tuple of (tool_call_id, tool_name, arguments).
    """
    tool_call_id = tool_call.id
    tool_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        arguments = {}
    return tool_call_id, tool_name, arguments
