"""Filesystem experiment tracker.

Used when MLflow is unavailable or unwanted (tests, air-gapped runs). Runs are
stored as ``artifacts/runs/<experiment>/<run_id>/run.json`` plus an ``artifacts/``
subdirectory, which is enough to compare runs and reproduce a model.

This is a real implementation, not a stub: the training pipeline works
identically on it. What it does not provide is MLflow's UI, model registry
integration or remote tracking -- which is exactly why MLflow is the default.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.utils import jsonable, new_id, utcnow_iso, write_json
from app.tracking.base import ExperimentTracker, RunInfo

logger = get_logger(__name__)


class LocalFileTracker(ExperimentTracker):
    """JSON-on-disk experiment tracker."""

    backend = "local"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.paths.artifacts_dir / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._active: RunInfo | None = None
        self._stack: list[RunInfo] = []

    def _run_dir(self, experiment_name: str, run_id: str) -> Path:
        return self.root / _safe(experiment_name) / run_id

    def _persist(self, info: RunInfo) -> None:
        path = self._run_dir(info.experiment_name, info.run_id) / "run.json"
        write_json(path, info.__dict__)

    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> RunInfo:
        run_id = new_id()
        info = RunInfo(
            run_id=run_id,
            experiment_name=experiment_name,
            run_name=run_name or run_id[:8],
            status="RUNNING",
            start_time=utcnow_iso(),
            tags={
                "fmops.environment": self.settings.environment,
                "fmops.git_commit": self.settings.git_commit,
                **(tags or {}),
            },
        )
        run_dir = self._run_dir(experiment_name, run_id)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        info.artifact_uri = (run_dir / "artifacts").resolve().as_uri()
        self._persist(info)
        if nested and self._active is not None:
            self._stack.append(self._active)
        self._active = info
        logger.info(
            "tracking.run_started",
            extra={"run_id": run_id, "experiment": experiment_name, "backend": self.backend},
        )
        return info

    def _require_active(self) -> RunInfo:
        if self._active is None:
            raise RuntimeError("no active run; call start_run() first")
        return self._active

    def log_params(self, params: dict[str, Any]) -> None:
        info = self._require_active()
        info.params.update({str(k): jsonable(v) for k, v in params.items()})
        self._persist(info)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        info = self._require_active()
        info.metrics.update(
            {
                str(k): float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        )
        self._persist(info)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        info = self._require_active()
        source = Path(path)
        if not source.exists():
            logger.warning("tracking.artifact_missing", extra={"path": str(source)})
            return
        target_dir = self._run_dir(info.experiment_name, info.run_id) / "artifacts"
        if artifact_path:
            target_dir = target_dir / artifact_path
        target_dir.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target_dir / source.name, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, target_dir / source.name)

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        info = self._require_active()
        target = self._run_dir(info.experiment_name, info.run_id) / "artifacts" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def set_tags(self, tags: dict[str, str]) -> None:
        info = self._require_active()
        info.tags.update({str(k): str(v) for k, v in tags.items()})
        self._persist(info)

    def end_run(self, status: str = "FINISHED") -> RunInfo | None:
        info = self._active
        if info is not None:
            info.status = status
            info.end_time = utcnow_iso()
            self._persist(info)
            logger.info(
                "tracking.run_finished", extra={"run_id": info.run_id, "status": status}
            )
        self._active = self._stack.pop() if self._stack else None
        return info

    def list_experiments(self) -> list[dict[str, Any]]:
        out = []
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            out.append(
                {
                    "experiment_id": directory.name,
                    "name": directory.name,
                    "artifact_location": str(directory),
                    "lifecycle_stage": "active",
                    "run_count": len([p for p in directory.iterdir() if p.is_dir()]),
                }
            )
        return out

    def search_runs(
        self, experiment_name: str | None = None, max_results: int = 50
    ) -> list[RunInfo]:
        name = experiment_name or self.settings.tracking.experiment_name
        directory = self.root / _safe(name)
        if not directory.is_dir():
            return []
        runs: list[RunInfo] = []
        for run_dir in directory.iterdir():
            payload = _read_run(run_dir / "run.json")
            if payload:
                runs.append(RunInfo(**payload))
        runs.sort(key=lambda r: r.start_time, reverse=True)
        return runs[:max_results]

    def get_run(self, run_id: str) -> RunInfo | None:
        for experiment_dir in self.root.iterdir():
            payload = _read_run(experiment_dir / run_id / "run.json")
            if payload:
                return RunInfo(**payload)
        return None

    @property
    def active_run(self) -> RunInfo | None:
        return self._active


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _read_run(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
