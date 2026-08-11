"""Tool schemas for LLM function calling."""

from __future__ import annotations

from typing import Any

# OpenAI-compatible tool schemas for function calling.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_script",
            "description": (
                "Execute a script inside the Docker environment. "
                "Use this to run commands, explore files, or interact with "
                "the challenge system."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The script contents to execute.",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["bash", "python"],
                        "description": "The scripting language to use.",
                    },
                },
                "required": ["script", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_goal",
            "description": (
                "Submit your answer to complete the challenge goal. "
                "The answer format depends on the goal type: "
                "for flags use a string, for list goals use comma-separated "
                "values or a JSON array. "
                "This will terminate the session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "Your answer to submit. For single values, "
                            "provide a string. For lists, use comma-separated "
                            "values (e.g., 'item1, item2, item3')."
                        ),
                    },
                },
                "required": ["answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": (
                "Give up on solving the challenge. "
                "Use this only when you have exhausted all options."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Explanation for why you are giving up.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": (
                "Signal that you have completed the assigned task. "
                "Use this when you have gathered all necessary information "
                "or accomplished the task objective. "
                "Include a summary of your findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "string",
                        "description": (
                            "Summary of findings, results, and any relevant "
                            "information discovered during the task."
                        ),
                    },
                },
                "required": ["findings"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_human_help",
            "description": (
                "Request help from a human operator. "
                "Use this when you need clarification or are stuck."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Your question for the human operator.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def get_tool_schemas() -> list[dict[str, Any]]:
    """Returns the list of tool schemas for function calling.

    Returns:
        List of OpenAI-compatible tool schemas.
    """
    return TOOL_SCHEMAS


def _filter_tools(names: set[str]) -> list[dict[str, Any]]:
    return [
        s for s in TOOL_SCHEMAS
        if s.get("function", {}).get("name") in names
    ]


def get_agent_tool_schemas() -> list[dict[str, Any]]:
    """Tool schemas for single-agent ReAct."""
    return _filter_tools(
        {"execute_script", "submit_goal", "give_up", "request_human_help"}
    )


def get_worker_tool_schemas() -> list[dict[str, Any]]:
    """Tool schemas for worker agents."""
    return _filter_tools(
        {"execute_script", "task_complete", "give_up", "request_human_help"}
    )


def get_executor_tool_schemas() -> list[dict[str, Any]]:
    """Tool schemas for the reasoner_executor executor."""
    return _filter_tools(
        {"execute_script", "submit_goal", "give_up", "request_human_help"}
    )


def get_orchestrator_tool_schemas() -> list[dict[str, Any]]:
    """Returns tool schemas for the orchestrator (submit_goal, give_up only).

    The orchestrator can call submit_goal or give_up directly instead of
    relying on JSON task output.

    Returns:
        List of OpenAI-compatible tool schemas for the orchestrator.
    """
    return _filter_tools({"submit_goal", "give_up"})
