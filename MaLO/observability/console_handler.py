"""Console output handler for MaLO CLI using Rich."""

from __future__ import annotations

from enum import Enum
from typing import Any

from rich.console import Console
from rich.text import Text


class AgentRole(Enum):
    """Agent roles across all architectures."""

    AGENT = "Agent"  # single_react
    REASONER = "Reasoner"  # reasoner_executor
    EXECUTOR = "Executor"  # reasoner_executor
    ORCHESTRATOR = "Orchestrator"  # orchestrator_*
    CRITIC = "Critic"  # orchestrator_critic_workers
    WORKER = "Worker"  # orchestrator_*


class ConsoleHandler:
    """Handles console output for agent execution with Rich formatting.

    Supports multiple verbosity levels and multi-agent architectures.

    Attributes:
        verbose: Whether to show detailed output.
        quiet: Whether to show only final results.
    """

    # Color scheme
    COLOR_THINKING = "cyan"
    COLOR_TOOL = "yellow"
    COLOR_MESSAGE = "blue"
    COLOR_APPROVE = "green"
    COLOR_REJECT = "red"
    COLOR_SUCCESS = "bold green"
    COLOR_FAILURE = "bold red"
    COLOR_DIM = "dim"

    def __init__(
        self,
        verbose: bool = False,
        quiet: bool = False,
    ) -> None:
        """Initializes the console handler.

        Args:
            verbose: Show detailed output including full tool args/output.
            quiet: Show only final result line.
        """
        self._console = Console()
        self._verbose = verbose
        self._quiet = quiet
        self._worker_count = 0

    @property
    def verbose(self) -> bool:
        """Returns whether verbose mode is enabled."""
        return self._verbose

    @property
    def quiet(self) -> bool:
        """Returns whether quiet mode is enabled."""
        return self._quiet

    def _format_turn(self, turn: int, max_turns: int) -> str:
        """Formats the turn counter prefix.

        Args:
            turn: Current turn number.
            max_turns: Maximum turns allowed.

        Returns:
            Formatted turn string like "[1/30]".
        """
        return f"[{turn}/{max_turns}]"

    def _format_agent(
        self,
        agent: AgentRole,
        worker_id: int | None = None,
    ) -> str:
        """Formats the agent name with optional worker ID.

        Args:
            agent: Agent role.
            worker_id: Optional worker ID for worker agents.

        Returns:
            Formatted agent string like "Agent" or "Worker[1]".
        """
        if agent == AgentRole.WORKER and worker_id is not None:
            return f"Worker[{worker_id}]"
        return agent.value

    def _get_indent(self, agent: AgentRole) -> str:
        """Returns indentation for agent hierarchy.

        Args:
            agent: Agent role.

        Returns:
            Indentation string (empty or spaces).
        """
        if agent == AgentRole.WORKER:
            return "  "
        return ""

    def on_challenge_start(
        self,
        name: str,
        category: str,
        architecture: str,
        max_turns: int,
    ) -> None:
        """Called at the start of a challenge.

        Args:
            name: Challenge name.
            category: Challenge category.
            architecture: Architecture being used.
            max_turns: Maximum turns allowed.
        """
        if self._quiet:
            return

        header = (
            f"Challenge: {name} | Category: {category} | "
            f"{architecture} | Max turns: {max_turns}"
        )
        self._console.print(header)
        self._console.print("─" * len(header))

    def on_agent_thinking(
        self,
        turn: int,
        max_turns: int,
        agent: AgentRole,
        worker_id: int | None = None,
    ) -> None:
        """Called when an agent starts thinking.

        Args:
            turn: Current turn number.
            max_turns: Maximum turns allowed.
            agent: Agent role.
            worker_id: Optional worker ID.
        """
        if self._quiet:
            return

        indent = self._get_indent(agent)
        turn_str = self._format_turn(turn, max_turns)
        agent_str = self._format_agent(agent, worker_id)

        text = Text()
        text.append(f"{turn_str} {indent}{agent_str}: ", style=self.COLOR_DIM)
        text.append("thinking...", style=self.COLOR_THINKING)
        self._console.print(text)

    def on_agent_response(
        self,
        _turn: int,
        _max_turns: int,
        agent: AgentRole,
        content: str | None,
        _has_tool_calls: bool,
        _worker_id: int | None = None,
    ) -> None:
        """Called when an agent produces a response.

        Args:
            _turn: Current turn number (unused, for API consistency).
            _max_turns: Maximum turns allowed (unused, for API consistency).
            agent: Agent role.
            content: Response content (reasoning).
            _has_tool_calls: Whether response includes tool calls (unused).
            _worker_id: Optional worker ID (unused, for API consistency).
        """
        if self._quiet:
            return

        # In verbose mode, show the reasoning content
        if self._verbose and content:
            indent = self._get_indent(agent)
            # Truncate long content for display
            display_content = content[:200] + "..." if len(content) > 200 else content
            # Show as indented quote
            for line in display_content.split("\n")[:3]:
                self._console.print(
                    f"       {indent}\"{line}\"",
                    style=self.COLOR_DIM,
                )

    def on_agent_message(
        self,
        turn: int,
        max_turns: int,
        from_agent: AgentRole,
        to_agent: AgentRole | None,
        message: str,
        from_worker_id: int | None = None,
        to_worker_id: int | None = None,
    ) -> None:
        """Called for inter-agent communication.

        Args:
            turn: Current turn number.
            max_turns: Maximum turns allowed.
            from_agent: Source agent role.
            to_agent: Target agent role (None if broadcast).
            message: Message content.
            from_worker_id: Optional source worker ID.
            to_worker_id: Optional target worker ID.
        """
        if self._quiet:
            return

        indent = self._get_indent(from_agent)
        turn_str = self._format_turn(turn, max_turns)
        from_str = self._format_agent(from_agent, from_worker_id)

        text = Text()
        text.append(f"{turn_str} {indent}{from_str}", style=self.COLOR_DIM)

        if to_agent:
            to_str = self._format_agent(to_agent, to_worker_id)
            text.append(" -> ", style=self.COLOR_DIM)
            text.append(to_str, style=self.COLOR_MESSAGE)

        # Truncate message for compact display
        display_msg = message[:50] + "..." if len(message) > 50 else message
        text.append(f": \"{display_msg}\"", style=self.COLOR_MESSAGE)

        self._console.print(text)

    def on_tool_call(
        self,
        turn: int,
        max_turns: int,
        agent: AgentRole,
        tool_name: str,
        args: dict[str, Any],
        worker_id: int | None = None,
    ) -> None:
        """Called when a tool is invoked.

        Args:
            turn: Current turn number.
            max_turns: Maximum turns allowed.
            agent: Agent role.
            tool_name: Name of the tool being called.
            args: Tool arguments.
            worker_id: Optional worker ID.
        """
        if self._quiet:
            return

        indent = self._get_indent(agent)
        turn_str = self._format_turn(turn, max_turns)
        agent_str = self._format_agent(agent, worker_id)

        # Extract language for execute_script
        extra_info = ""
        if tool_name == "execute_script" and "language" in args:
            extra_info = f" ({args['language']})"

        text = Text()
        text.append(f"{turn_str} {indent}{agent_str}: ", style=self.COLOR_DIM)
        text.append(f"{tool_name}{extra_info}", style=self.COLOR_TOOL)

        self._console.print(text, end="")

        # In verbose mode, show full arguments
        if self._verbose:
            self._console.print()
            if "script" in args:
                script_preview = args["script"][:100]
                if len(args["script"]) > 100:
                    script_preview += "..."
                self._console.print(
                    f"       {indent}Script: {script_preview}",
                    style=self.COLOR_DIM,
                )

    def on_tool_result(
        self,
        tool_name: str,
        output: str,
        char_count: int,
        success: bool | None = None,
        agent: AgentRole | None = None,
        _worker_id: int | None = None,
    ) -> None:
        """Called when tool execution completes.

        Args:
            tool_name: Name of the tool.
            output: Tool output.
            char_count: Length of the output.
            success: Whether the tool succeeded (for submit_goal).
            agent: Agent role (for indentation).
            _worker_id: Optional worker ID (unused, for API consistency).
        """
        if self._quiet:
            return

        indent = ""
        if agent:
            indent = self._get_indent(agent)

        # Handle special tools
        if tool_name == "submit_goal":
            if success:
                # Extract flag preview
                flag_preview = output[:30] + "..." if len(output) > 30 else output
                self._console.print(
                    f" -> SUCCESS: {flag_preview}",
                    style=self.COLOR_SUCCESS,
                )
            else:
                self._console.print(" -> FAILED", style=self.COLOR_FAILURE)
        elif tool_name == "give_up":
            self._console.print(" -> GAVE UP", style=self.COLOR_FAILURE)
        else:
            # Regular tool - show char count
            self._console.print(f" -> {char_count} chars", style=self.COLOR_DIM)

        # In verbose mode, show output
        if self._verbose and output:
            output_preview = output[:200] + "..." if len(output) > 200 else output
            for line in output_preview.split("\n")[:5]:
                self._console.print(
                    f"       {indent}Output: {line}",
                    style=self.COLOR_DIM,
                )

    def on_critique(
        self,
        turn: int,
        max_turns: int,
        verdict: str,
        reason: str | None = None,
    ) -> None:
        """Called when critic evaluates a plan.

        Args:
            turn: Current turn number.
            max_turns: Maximum turns allowed.
            verdict: APPROVE or REJECT.
            reason: Optional rejection reason.
        """
        if self._quiet:
            return

        turn_str = self._format_turn(turn, max_turns)
        text = Text()
        text.append(f"{turn_str} Critic: ", style=self.COLOR_DIM)

        if verdict.upper() == "APPROVE":
            text.append("APPROVE", style=self.COLOR_APPROVE)
        else:
            text.append("REJECT", style=self.COLOR_REJECT)
            if reason:
                text.append(f" ({reason})", style=self.COLOR_DIM)

        self._console.print(text)

    def spawn_worker(self) -> int:
        """Spawns a new worker and returns its ID.

        Returns:
            New worker ID (1-indexed).
        """
        self._worker_count += 1
        return self._worker_count

    def on_run_end(
        self,
        success: bool,
        termination_reason: str,
        turn_count: int,
        total_tokens: int,
        cost_usd: float,
    ) -> None:
        """Called at the end of a run.

        Args:
            success: Whether the run succeeded.
            termination_reason: Reason for termination.
            turn_count: Total turns taken.
            total_tokens: Total tokens used.
            cost_usd: Estimated cost in USD.
        """
        if not self._quiet:
            self._console.print("─" * 60)

        text = Text()
        text.append("Result: ", style="bold")

        if success:
            text.append("SUCCESS", style=self.COLOR_SUCCESS)
        else:
            text.append(f"FAILED ({termination_reason})", style=self.COLOR_FAILURE)

        text.append(f" in {turn_count} turns", style=self.COLOR_DIM)
        text.append(f" | Tokens: {total_tokens:,}", style=self.COLOR_DIM)
        text.append(f" | Cost: ${cost_usd:.3f}", style=self.COLOR_DIM)

        self._console.print(text)

    def print_error(self, message: str) -> None:
        """Prints an error message.

        Args:
            message: Error message to display.
        """
        self._console.print(f"ERROR: {message}", style=self.COLOR_FAILURE)

    def print_info(self, message: str) -> None:
        """Prints an info message.

        Args:
            message: Info message to display.
        """
        if not self._quiet:
            self._console.print(message, style=self.COLOR_DIM)
