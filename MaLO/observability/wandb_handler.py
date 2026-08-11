"""WandB integration for MaLO experiment tracking."""

from __future__ import annotations

from typing import Any

from MaLO.observability.metrics import RunMetrics


class WandBHandler:
    """Handles logging to Weights & Biases."""

    def __init__(
        self,
        project: str,
        experiment_name: str,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        """Initializes the WandB handler.

        Args:
            project: WandB project name.
            experiment_name: Name for this run.
            config: Optional configuration to log.
            enabled: Whether WandB logging is enabled.
        """
        self._project = project
        self._experiment_name = experiment_name
        self._config = config or {}
        self._enabled = enabled
        self._run: Any = None

    def start(self) -> None:
        """Starts a WandB run."""
        if not self._enabled:
            return
        try:
            import wandb

            # Finish any existing run before starting a new one.
            if wandb.run is not None:
                wandb.finish()

            self._run = wandb.init(
                project=self._project,
                name=self._experiment_name,
                config=self._config,
            )
        except ImportError:
            self._enabled = False

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Logs metrics for a single step.

        Args:
            step: Step number.
            metrics: Metrics to log.
        """
        if not self._enabled or self._run is None:
            return
        try:
            import wandb

            wandb.log(metrics, step=step)
        except Exception:
            pass

    def log_summary(self, metrics: RunMetrics) -> None:
        """Logs final summary metrics.

        Args:
            metrics: Run metrics to log.
        """
        if not self._enabled or self._run is None:
            return
        try:
            import wandb

            for key, value in metrics.to_dict().items():
                wandb.run.summary[key] = value
        except Exception:
            pass

    def finish(self, metrics: RunMetrics | None = None) -> None:
        """Finishes the WandB run.

        Args:
            metrics: Optional final metrics to log before finishing.
        """
        if not self._enabled or self._run is None:
            return
        try:
            if metrics:
                self.log_summary(metrics)
            self._run.finish()
        except Exception:
            pass
        finally:
            self._run = None

    @property
    def enabled(self) -> bool:
        """Returns whether WandB logging is enabled."""
        return self._enabled
