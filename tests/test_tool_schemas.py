from __future__ import annotations

from MaLO.tools.schemas import (
    get_agent_tool_schemas,
    get_executor_tool_schemas,
    get_orchestrator_tool_schemas,
    get_worker_tool_schemas,
)


def _names(schemas: list[dict]) -> set[str]:
    return {s.get("function", {}).get("name") for s in schemas}


def test_agent_tools_exclude_task_complete() -> None:
    names = _names(get_agent_tool_schemas())
    assert "task_complete" not in names
    assert "execute_script" in names


def test_worker_tools_include_task_complete() -> None:
    names = _names(get_worker_tool_schemas())
    assert "task_complete" in names
    assert "submit_goal" not in names


def test_executor_tools_exclude_task_complete() -> None:
    names = _names(get_executor_tool_schemas())
    assert "task_complete" not in names
    assert "submit_goal" in names


def test_orchestrator_tools_only_submit_and_give_up() -> None:
    names = _names(get_orchestrator_tool_schemas())
    assert names == {"submit_goal", "give_up"}
