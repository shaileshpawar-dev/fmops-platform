"""MLflow-backed experiment tracker."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import DependencyMissingError, ProviderUnavailableError
from app.core.logging import get_logger
from app.core.utils import jsonable, utcnow_iso
from app.tracking.base import ExperimentTracker, RunInfo

logger = get_logger(__name__)


def _import_mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError as exc:  # pragma: no cover - mlflow is a core dependency
        raise DependencyMissingError(
            "mlflow is not installed; install it or set FMOPS_TRACKING__BACKEND=local",
            backend="mlflow",
        ) from exc


class MLflowTracker(ExperimentTracker):
    """Tracks runs in MLflow.

    Local development points at a SQLite tracking store under ``artifacts/``;
    staging and production set ``FMOPS_TRACKING__TRACKING_URI`` to a tracking
    server. Nothing else in the pipeline changes between the two.
    """

    backend = "mlflow"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._mlflow = _import_mlflow()
        self._active: RunInfo | None = None
        self._stack: list[RunInfo] = []
        try:
            self._mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            self._mlflow.set_registry_uri(self.settings.mlflow_registry_uri)
        except Exception as exc:
            raise ProviderUnavailableError(
                f"could not reach the MLflow tracking store: {exc}",
                tracking_uri=self.settings.mlflow_tracking_uri,
            ) from exc
        self.client = self._mlflow.tracking.MlflowClient()

    # -- run lifecycle ------------------------------------------------------- #
    def _ensure_experiment(self, name: str) -> str:
        experiment = self.client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        return self.client.create_experiment(
            name, artifact_location=self.settings.mlflow_artifact_root
        )

    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> RunInfo:
        experiment_id = self._ensure_experiment(experiment_name)
        merged_tags = {
            "fmops.environment": self.settings.environment,
            "fmops.service": self.settings.service_name,
            "fmops.git_commit": self.settings.git_commit,
            **(tags or {}),
        }
        try:
            run = self._mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags=merged_tags,
                nested=nested,
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"failed to start MLflow run: {exc}", experiment=experiment_name
            ) from exc

        info = RunInfo(
            run_id=run.info.run_id,
            experiment_name=experiment_name,
            run_name=run_name or run.info.run_name or "",
            status="RUNNING",
            start_time=utcnow_iso(),
            tags=merged_tags,
            artifact_uri=run.info.artifact_uri or "",
        )
        if nested and self._active is not None:
            self._stack.append(self._active)
        self._active = info
        logger.info(
            "tracking.run_started",
            extra={
                "run_id": info.run_id,
                "experiment": experiment_name,
                "backend": self.backend,
                "nested": nested,
            },
        )
        return info

    def log_params(self, params: dict[str, Any]) -> None:
        if not params:
            return
        # MLflow params are immutable strings capped at 500 chars.
        clean = {str(k): str(jsonable(v))[:500] for k, v in params.items()}
        self._mlflow.log_params(clean)
        if self._active:
            self._active.params.update(clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not clean:
            return
        self._mlflow.log_metrics(clean, step=step)
        if self._active:
            self._active.metrics.update(clean)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        source = Path(path)
        if not source.exists():
            logger.warning("tracking.artifact_missing", extra={"path": str(source)})
            return
        if source.is_dir():
            self._mlflow.log_artifacts(str(source), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(source), artifact_path=artifact_path)

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._mlflow.log_artifact(str(target))

    def set_tags(self, tags: dict[str, str]) -> None:
        if not tags:
            return
        self._mlflow.set_tags({str(k): str(v)[:500] for k, v in tags.items()})
        if self._active:
            self._active.tags.update({str(k): str(v) for k, v in tags.items()})

    def end_run(self, status: str = "FINISHED") -> RunInfo | None:
        info = self._active
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("tracking.end_run_failed", extra={"error": str(exc)})
        if info is not None:
            info.status = status
            info.end_time = utcnow_iso()
            logger.info(
                "tracking.run_finished",
                extra={"run_id": info.run_id, "status": status},
            )
        self._active = self._stack.pop() if self._stack else None
        return info

    # -- queries ------------------------------------------------------------- #
    def list_experiments(self) -> list[dict[str, Any]]:
        out = []
        for experiment in self.client.search_experiments():
            out.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "artifact_location": experiment.artifact_location,
                    "lifecycle_stage": experiment.lifecycle_stage,
                    "creation_time": experiment.creation_time,
                }
            )
        return out

    def search_runs(
        self, experiment_name: str | None = None, max_results: int = 50
    ) -> list[RunInfo]:
        name = experiment_name or self.settings.tracking.experiment_name
        experiment = self.client.get_experiment_by_name(name)
        if experiment is None:
            return []
        runs = self.client.search_runs(
            [experiment.experiment_id],
            max_results=max_results,
            order_by=["attribute.start_time DESC"],
        )
        return [self._to_info(run, name) for run in runs]

    def get_run(self, run_id: str) -> RunInfo | None:
        try:
            run = self.client.get_run(run_id)
        except Exception:
            return None
        experiment = self.client.get_experiment(run.info.experiment_id)
        return self._to_info(run, experiment.name if experiment else "")

    def _to_info(self, run, experiment_name: str) -> RunInfo:
        return RunInfo(
            run_id=run.info.run_id,
            experiment_name=experiment_name,
            run_name=run.info.run_name or "",
            status=run.info.status,
            start_time=str(run.info.start_time),
            end_time=str(run.info.end_time) if run.info.end_time else None,
            params=dict(run.data.params),
            metrics=dict(run.data.metrics),
            tags={k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
            artifact_uri=run.info.artifact_uri or "",
        )

    @property
    def active_run(self) -> RunInfo | None:
        return self._active
