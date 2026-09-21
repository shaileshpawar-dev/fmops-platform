"""Dataset versioning.

Every training run must be traceable to the *exact bytes* it trained on. Two
mechanisms cooperate:

1. **Content addressing (always on).** Each registered dataset gets a version
   id derived from its SHA-256 content hash, e.g. ``v3-a1b2c3d4``. The manifest
   in ``data/versions.json`` records the hash, row/column counts, schema, git
   commit and parent version. This works with no external tooling and is what
   the model registry stores against each model version.

2. **DVC (optional, recommended).** When DVC is installed and initialised, the
   same file is also tracked with ``dvc add``, producing a ``.dvc`` pointer that
   is committed to git while the payload goes to the DVC remote (S3 in this
   project). :class:`DVCTracker` shells out to the DVC CLI and degrades to a
   clear, logged no-op when DVC is unavailable -- it never pretends to have
   versioned something it did not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import DatasetNotFoundError
from app.core.logging import get_logger
from app.core.utils import git_commit, hash_file, read_json, utcnow_iso, write_json
from app.schemas.data import DatasetProfile, DatasetVersion

logger = get_logger(__name__)

MANIFEST_NAME = "versions.json"


class DVCTracker:
    """Thin wrapper around the DVC CLI.

    Every method reports whether it actually did anything, so callers can record
    ``dvc_tracked`` honestly in the manifest.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or get_settings().paths.data_dir.parent

    @property
    def executable(self) -> str | None:
        return shutil.which("dvc")

    @property
    def available(self) -> bool:
        return self.executable is not None

    @property
    def initialised(self) -> bool:
        return (self.repo_root / ".dvc").is_dir()

    def _run(self, *args: str, timeout: int = 120) -> tuple[bool, str]:
        executable = self.executable
        if executable is None:
            return False, "dvc executable not found on PATH"
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [executable, *args],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output.strip()

    def init(self, no_scm: bool = False) -> tuple[bool, str]:
        if self.initialised:
            return True, "dvc already initialised"
        args = ["init"]
        if no_scm or not (self.repo_root / ".git").exists():
            args.append("--no-scm")
        return self._run(*args)

    def add(self, path: Path | str) -> tuple[bool, str]:
        """Track a file with DVC. Returns (tracked, message)."""
        if not self.available:
            return False, "dvc not installed; dataset tracked by content hash only"
        if not self.initialised:
            return False, "dvc not initialised; run 'make dvc-init'"
        try:
            relative = Path(path).resolve().relative_to(self.repo_root.resolve())
        except ValueError:
            relative = Path(path)
        ok, output = self._run("add", str(relative).replace("\\", "/"))
        if ok:
            logger.info("dvc.added", extra={"path": str(relative)})
        else:
            logger.warning("dvc.add_failed", extra={"path": str(relative), "output": output})
        return ok, output

    def push(self, remote: str | None = None) -> tuple[bool, str]:
        if not (self.available and self.initialised):
            return False, "dvc unavailable"
        args = ["push"]
        if remote:
            args += ["-r", remote]
        return self._run(*args, timeout=600)

    def status(self) -> dict[str, Any]:
        return {
            "installed": self.available,
            "initialised": self.initialised,
            "executable": self.executable,
            "repo_root": str(self.repo_root),
        }


class DatasetRegistry:
    """Content-addressed dataset manifest, optionally backed by DVC."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.paths.ensure()
        self.manifest_path = self.settings.paths.data_dir / MANIFEST_NAME
        self.dvc = DVCTracker(repo_root=self.settings.paths.data_dir.parent)

    # -- manifest I/O -------------------------------------------------------- #
    def _load(self) -> dict[str, Any]:
        return read_json(self.manifest_path, default={"versions": []}) or {"versions": []}

    def _save(self, manifest: dict[str, Any]) -> None:
        write_json(self.manifest_path, manifest)

    # -- registration -------------------------------------------------------- #
    def register(
        self,
        path: Path | str,
        dataset_name: str | None = None,
        description: str = "",
        use_dvc: bool = True,
        parent_version: str | None = None,
    ) -> DatasetVersion:
        """Register a dataset file and return its immutable version record.

        ``parent_version`` records what this version was derived from (a
        retraining set names the dataset it extended). Without it the parent is
        the previous version of the same dataset name.

        Registering identical bytes twice returns the existing version rather
        than creating a duplicate -- versions are content, not events.
        """
        source = Path(path)
        if not source.is_file():
            raise DatasetNotFoundError(f"dataset file not found: {source}", path=str(source))

        dataset_name = dataset_name or self.settings.data.dataset_name
        content_hash = hash_file(source)

        existing = self._find(content_hash, dataset_name)
        if existing is not None:
            return existing

        # Everything slow -- reading the file, DVC -- happens before the lock.
        frame = pd.read_csv(source, nrows=200_000)
        tracked = False
        if use_dvc:
            tracked, message = self.dvc.add(source)
            if not tracked:
                logger.info("dataset.dvc_skipped", extra={"reason": message})
        details = {
            "dataset_name": dataset_name,
            "path": str(source),
            "content_hash": content_hash,
            "n_rows": len(frame),
            "n_columns": int(frame.shape[1]),
            "columns": [str(c) for c in frame.columns],
            "size_bytes": source.stat().st_size,
            "git_commit": git_commit(self.settings.paths.data_dir.parent),
            "dvc_tracked": tracked,
            "description": description,
            "stats": _frame_stats(frame, self.settings),
        }

        # The manifest itself is read-modified-written; with more than one API
        # process two uploads could each drop the other's entry. The database
        # write lock is shared by every process, so it serialises this step --
        # which is only the check-and-append, not the work above.
        from app.core.db import get_database

        with get_database().transaction():
            manifest = self._load()
            duplicate = self._find(content_hash, dataset_name, manifest)
            if duplicate is not None:
                return duplicate
            siblings = [v for v in manifest["versions"] if v["dataset_name"] == dataset_name]
            version = f"v{len(siblings) + 1}-{content_hash[:8]}"
            record = DatasetVersion(
                version=version,
                parent_version=parent_version
                or (siblings[-1]["version"] if siblings else None),
                created_at=utcnow_iso(),
                **details,
            )
            manifest["versions"].append(record.model_dump(mode="json"))
            manifest["updated_at"] = utcnow_iso()
            self._save(manifest)
        logger.info(
            "dataset.registered",
            extra={
                "dataset": dataset_name,
                "version": version,
                "rows": record.n_rows,
                "dvc_tracked": tracked,
            },
        )
        return record

    def _find(
        self, content_hash: str, dataset_name: str, manifest: dict[str, Any] | None = None
    ) -> DatasetVersion | None:
        for existing in (manifest or self._load())["versions"]:
            if (
                existing["content_hash"] == content_hash
                and existing["dataset_name"] == dataset_name
            ):
                logger.info(
                    "dataset.version_exists",
                    extra={"version": existing["version"], "hash": content_hash[:12]},
                )
                return DatasetVersion(**existing)
        return None

    def register_frame(
        self,
        frame: pd.DataFrame,
        dataset_name: str | None = None,
        filename: str | None = None,
        description: str = "",
        use_dvc: bool = True,
        parent_version: str | None = None,
    ) -> DatasetVersion:
        """Persist an in-memory frame to ``data/processed`` and register it.

        The file name carries the content hash. A version is an immutable
        snapshot: writing to a fixed name would let the next upload of
        "churn.csv" -- or the next retraining run -- overwrite the bytes an
        earlier version and every model trained on it still point at.
        """
        dataset_name = dataset_name or self.settings.data.dataset_name
        stem = Path(filename or f"{dataset_name}.csv").stem
        directory = self.settings.paths.processed_dir
        directory.mkdir(parents=True, exist_ok=True)
        staging = directory / f".{stem}-{uuid.uuid4().hex}.tmp"
        frame.to_csv(staging, index=False)
        target = directory / f"{stem}-{hash_file(staging)[:12]}.csv"
        if target.exists():
            staging.unlink()  # identical bytes are already on disk
        else:
            os.replace(staging, target)
        return self.register(target, dataset_name, description, use_dvc, parent_version)

    # -- lookup -------------------------------------------------------------- #
    def list_versions(self, dataset_name: str | None = None) -> list[DatasetVersion]:
        manifest = self._load()
        records = manifest["versions"]
        if dataset_name:
            records = [r for r in records if r["dataset_name"] == dataset_name]
        return [DatasetVersion(**r) for r in records]

    def get(self, version: str) -> DatasetVersion:
        for record in self._load()["versions"]:
            if record["version"] == version:
                return DatasetVersion(**record)
        raise DatasetNotFoundError(f"unknown dataset version: {version}", version=version)

    def latest(self, dataset_name: str | None = None) -> DatasetVersion | None:
        versions = self.list_versions(dataset_name)
        return versions[-1] if versions else None

    def load(self, version: str) -> pd.DataFrame:
        """Load a registered version, verifying the content hash first."""
        record = self.get(version)
        path = Path(record.path)
        if not path.is_file():
            raise DatasetNotFoundError(
                f"dataset file for {version} is missing at {path}; "
                "run 'dvc pull' or regenerate with 'make data'",
                version=version,
                path=str(path),
            )
        actual = hash_file(path)
        if actual != record.content_hash:
            raise DatasetNotFoundError(
                f"content hash mismatch for {version}: the file on disk has "
                "changed since it was registered",
                version=version,
                expected=record.content_hash[:12],
                actual=actual[:12],
            )
        return pd.read_csv(path)

    def resolve(
        self, version: str | None = None, path: Path | str | None = None
    ) -> tuple[pd.DataFrame, DatasetVersion | None]:
        """Resolve a training input from a version id, an explicit path, or the latest.

        Returns the frame plus the version record when one exists. An explicit
        path that is not yet registered is registered on the fly so the run
        stays traceable.
        """
        if version:
            return self.load(version), self.get(version)
        if path:
            record = self.register(path)
            return self.load(record.version), record
        latest = self.latest()
        if latest is not None:
            return self.load(latest.version), latest
        raise DatasetNotFoundError(
            "no dataset registered; run 'make data' to generate the sample datasets"
        )


def _frame_stats(frame: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    target = settings.data.target_column
    stats: dict[str, Any] = {}
    if target in frame.columns:
        series = pd.to_numeric(frame[target], errors="coerce").dropna()
        if not series.empty:
            stats["positive_rate"] = round(float((series == 1).mean()), 5)
            stats["class_counts"] = {str(k): int(v) for k, v in series.value_counts().items()}
    stats["missing_cells"] = int(frame.isna().sum().sum())
    stats["duplicate_rows"] = int(frame.duplicated().sum())
    return stats


def build_profile(
    frame: pd.DataFrame,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    settings: Settings | None = None,
) -> DatasetProfile:
    """Column-level profile used as the drift reference window."""
    settings = settings or get_settings()
    cfg = settings.data
    numeric: dict[str, dict[str, float]] = {}
    for column in cfg.numeric_features:
        if column not in frame:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        numeric[column] = {
            "count": float(len(series)),
            "mean": float(series.mean()),
            "std": float(series.std() or 0.0),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
        }

    categorical: dict[str, dict[str, float]] = {}
    for column in cfg.categorical_features:
        if column not in frame:
            continue
        counts = frame[column].dropna().astype(str).value_counts(normalize=True)
        categorical[column] = {str(k): float(v) for k, v in counts.items()}

    target: dict[str, float] = {}
    if cfg.target_column in frame:
        raw = frame[cfg.target_column].dropna()
        if cfg.class_labels:
            positive = str(cfg.class_labels[1])
            series = (raw.astype(str) == positive).astype(int)
        else:
            series = pd.to_numeric(raw, errors="coerce").dropna()
        if not series.empty:
            target = {
                "positive_rate": float((series == 1).mean()),
                "count": float(len(series)),
            }

    return DatasetProfile(
        dataset_name=dataset_name or cfg.dataset_name,
        dataset_version=dataset_version,
        n_rows=len(frame),
        numeric=numeric,
        categorical=categorical,
        target=target,
    )


_REGISTRY: DatasetRegistry | None = None


def get_dataset_registry() -> DatasetRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DatasetRegistry()
    return _REGISTRY
