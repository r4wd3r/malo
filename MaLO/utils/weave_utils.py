"""Helpers for optional weave integration."""

from __future__ import annotations


class _NoWeave:
    """Fallback weave shim when weave is not installed."""

    is_available = False

    def op(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def init(self, *args, **kwargs) -> bool:
        return False

    def finish(self, *args, **kwargs) -> None:
        return None


def get_weave():
    """Returns weave if installed, otherwise a no-op shim."""
    try:
        import weave  # type: ignore
    except Exception:
        return _NoWeave()
    weave.is_available = True  # type: ignore[attr-defined]
    return weave
