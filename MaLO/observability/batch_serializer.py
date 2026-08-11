"""Serialization helpers for batch run results."""

from __future__ import annotations

from typing import Any, Protocol


class _ResultLike(Protocol):
    challenge_id: str
    challenge_name: str
    success: bool
    termination_reason: str
    turn_count: int
    total_tokens: int
    cost_usd: float
    error: str | None


class _BatchResultLike(Protocol):
    total_challenges: int
    successful: int
    failed: int
    errors: int
    results: list[_ResultLike]
    start_time: str
    end_time: str


def batch_result_to_dict(batch_result: _BatchResultLike) -> dict[str, Any]:
    """Converts a BatchResult into a JSON-serializable dictionary."""
    return {
        "total_challenges": batch_result.total_challenges,
        "successful": batch_result.successful,
        "failed": batch_result.failed,
        "errors": batch_result.errors,
        "success_rate": (
            batch_result.successful / batch_result.total_challenges
            if batch_result.total_challenges > 0
            else 0.0
        ),
        "start_time": batch_result.start_time,
        "end_time": batch_result.end_time,
        "total_cost_usd": round(
            sum(r.cost_usd for r in batch_result.results), 6
        ),
        "results": [
            {
                "challenge_id": r.challenge_id,
                "challenge_name": r.challenge_name,
                "success": r.success,
                "termination_reason": r.termination_reason,
                "turn_count": r.turn_count,
                "total_tokens": r.total_tokens,
                "cost_usd": round(r.cost_usd, 6),
                "error": r.error,
            }
            for r in batch_result.results
        ],
    }
