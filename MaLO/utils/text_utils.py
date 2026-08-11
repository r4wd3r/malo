"""Text utilities for truncation and formatting."""

from __future__ import annotations


def truncate_text(text: str, max_chars: int) -> str:
    """Truncates text to a maximum number of characters.

    Args:
        text: Original text.
        max_chars: Maximum allowed characters.

    Returns:
        Truncated text with an ellipsis if truncation occurred.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."
