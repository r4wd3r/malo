"""Core tool implementations for MaLO LangGraph."""

from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass
from typing import Literal

from MaLO.challenges.models import Goal
from MaLO.environment.environment_manager import EnvironmentManager
from MaLO.utils.text_utils import truncate_text


@dataclass(frozen=True)
class ToolResult:
    """Result payload for tool execution."""

    output: str
    success: bool
    terminate: bool = False
    termination_reason: str | None = None
    needs_human: bool = False


def _build_script_command(script: str, language: str) -> str:
    """Builds a command to execute a script inside a container.

    Args:
        script: Script contents.
        language: Script language ("python" or "bash").

    Returns:
        A shell command string.
    """
    script_id = uuid.uuid4().hex
    if language == "python":
        file_path = f"/tmp/malo_exec_{script_id}.py"
        interpreter = "python"
    else:
        file_path = f"/tmp/malo_exec_{script_id}.sh"
        interpreter = "bash"

    script_block = (
        f"cat <<'EOF' > {file_path}\n{script}\nEOF\n"
        f"{interpreter} {file_path}"
    )
    return f"bash -lc {shlex.quote(script_block)}"


def execute_script_tool(
    script: str,
    language: Literal["python", "bash"],
    env_manager: EnvironmentManager,
    truncation_chars: int,
    timeout_seconds: int | None = None,
) -> ToolResult:
    """Executes a script inside the Docker environment.

    Args:
        script: Script contents.
        language: Script language.
        env_manager: Active environment manager.
        truncation_chars: Maximum output length.
        timeout_seconds: Optional timeout in seconds for script execution.

    Returns:
        Tool result with command output.
    """
    command = _build_script_command(script, language)
    result = env_manager.exec_run(command, timeout_seconds=timeout_seconds)

    output_parts = []
    if result.stdout:
        output_parts.append(result.stdout)
    if result.stderr:
        output_parts.append(result.stderr)
    combined_output = "\n".join(output_parts)

    truncated_output = truncate_text(combined_output, truncation_chars)
    success = result.exit_code == 0 and not result.timed_out
    if result.timed_out:
        truncated_output = "Command timed out."
        return ToolResult(
            output=truncated_output,
            success=False,
            terminate=True,
            termination_reason="tool_timeout",
        )
    return ToolResult(output=truncated_output, success=success)


def submit_goal(
    answer: str | list[str],
    goal: Goal | None,
) -> ToolResult:
    """Submits an answer for goal validation.

    Supports multiple goal types:
    - exact_match: String comparison
    - list_match: List comparison
    - regex: Pattern matching

    Args:
        answer: The answer to submit (string or list).
        goal: Goal object for validation, or None if not configured.

    Returns:
        Tool result indicating success or failure.
    """
    if goal is None:
        return ToolResult(
            output="No goal configured for validation.",
            success=False,
            terminate=True,
            termination_reason="missing_goal",
        )

    is_valid, feedback = goal.validate_answer(answer)

    if is_valid:
        return ToolResult(
            output=feedback,
            success=True,
            terminate=True,
            termination_reason="success",
        )
    # Continue execution when answer is incorrect - agent can retry
    return ToolResult(
        output=f"{feedback} You may continue investigating.",
        success=False,
        terminate=False,
    )


def give_up(reason: str) -> ToolResult:
    """Signals that the agent is giving up.

    Args:
        reason: Explanation for giving up.

    Returns:
        Tool result indicating termination.
    """
    return ToolResult(
        output=reason,
        success=False,
        terminate=True,
        termination_reason="give_up",
    )


def task_complete(findings: str) -> ToolResult:
    """Signals that the worker has completed the assigned task.

    Args:
        findings: Summary of findings and results from the task.

    Returns:
        Tool result indicating task completion.
    """
    return ToolResult(
        output=findings,
        success=True,
        terminate=True,
        termination_reason="task_complete",
    )


def request_human_help(query: str) -> ToolResult:
    """Requests human input to continue execution.

    Args:
        query: Question for the human operator.

    Returns:
        Tool result indicating a pause for human input.
    """
    return ToolResult(
        output=query,
        success=True,
        terminate=False,
        needs_human=True,
    )
