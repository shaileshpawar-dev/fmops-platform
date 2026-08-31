"""Experiment tracking interface.

An :class:`ExperimentTracker` records the params, metrics, artifacts and tags of
one training run so it can be compared, reproduced and audited later.

Two backends ship:

* :class:`~app.tracking.mlflow_tracker.MLflowTracker` -- the default. Local runs
  use a SQLite tracking store; staging/production point at a tracking server.
* :class:`~app.tracking.local.LocalFileTracker` -- a JSON-on-disk fallback used
  in tests and in constrained environments where MLflow is not installed.

The interface is deliberately narrow so that swapping backends cannot change
pipeline behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunInfo:
    """Identity of an active or finished run."""

    run_id: str
    experiment_name: str
    run_name: str = ""
    status: str = "RUNNING"
    start_time: str = ""
    end_time: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifact_uri: str = ""


class ExperimentTracker(ABC):
    """Records one training run."""

    backend: str = "abstract"

    @abstractmethod
    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> RunInfo: ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    @abstractmethod
    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None: ...

    @abstractmethod
    def log_dict(self, payload: dict[str, Any], filename: str) -> None: ...

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None: ...

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> RunInfo | None: ...

    @abstractmethod
    def list_experiments(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def search_runs(
        self, experiment_name: str | None = None, max_results: int = 50
    ) -> list[RunInfo]: ...

    @abstractmethod
    def get_run(self, run_id: str) -> RunInfo | None: ...

    @property
    @abstractmethod
    def active_run(self) -> RunInfo | None: ...

    @contextmanager
    def run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Iterator[RunInfo]:
        """Context manager that always closes the run, failing it on exception."""
        info = self.start_run(experiment_name, run_name, tags, nested=nested)
        try:
            yield info
        except Exception:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")

    def log_model_metadata(self, metadata: dict[str, Any]) -> None:
        """Convenience: persist the reproducibility block with the run."""
        self.log_dict(metadata, "model_metadata.json")
