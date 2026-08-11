"""Metrics tracking for MaLO experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Approximate pricing per 1M tokens (USD) for common models via OpenRouter.
# These are rough estimates and may not reflect current pricing.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_cost_per_1m, output_cost_per_1m)
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4-turbo": (10.00, 30.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3-opus": (15.00, 75.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
}

DEFAULT_INPUT_COST = 1.00
DEFAULT_OUTPUT_COST = 3.00


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimates the cost of an API call.

    Args:
        model: Model identifier.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    input_cost, output_cost = MODEL_PRICING.get(
        model, (DEFAULT_INPUT_COST, DEFAULT_OUTPUT_COST)
    )
    cost = (prompt_tokens * input_cost + completion_tokens * output_cost) / 1e6
    return cost


@dataclass
class RunMetrics:
    """Accumulated metrics for a single experiment run."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    turn_count: int = 0
    tool_calls: int = 0
    success: bool = False
    termination_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Records metrics from an LLM call.

        Args:
            model: Model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += estimate_cost(
            model, prompt_tokens, completion_tokens
        )

    def record_tool_call(self) -> None:
        """Records a tool call."""
        self.tool_calls += 1

    def record_turn(self) -> None:
        """Records an agent turn."""
        self.turn_count += 1

    def finalize(self, success: bool, termination_reason: str) -> None:
        """Finalizes the run metrics.

        Args:
            success: Whether the run succeeded.
            termination_reason: Reason for termination.
        """
        self.success = success
        self.termination_reason = termination_reason

    def to_dict(self) -> dict[str, Any]:
        """Converts metrics to a dictionary.

        Returns:
            Metrics as a dictionary.
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": (
                self.total_prompt_tokens + self.total_completion_tokens
            ),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "turn_count": self.turn_count,
            "tool_calls": self.tool_calls,
            "success": self.success,
            "termination_reason": self.termination_reason,
            **self.extra,
        }
