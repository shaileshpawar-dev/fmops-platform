"""Small shared utilities: ids, time, hashing, git metadata, safe JSON."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Identifiers and time
# --------------------------------------------------------------------------- #
def new_id(prefix: str = "") -> str:
    """A short, sortable-enough unique identifier."""
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}-{raw}" if prefix else raw


def utcnow() -> datetime:
    return datetime.now(UTC)


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with a trailing Z, used for every stored time."""
    return utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_iso(value: str) -> datetime:
    """Parse a timestamp produced by :func:`utcnow_iso` (or any ISO string)."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def iso_days_ago(days: int) -> str:
    from datetime import timedelta

    return (utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def iso_minutes_ago(minutes: int) -> str:
    from datetime import timedelta

    return (utcnow() - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --------------------------------------------------------------------------- #
# Hashing / fingerprints
# --------------------------------------------------------------------------- #
def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(text.encode("utf-8"))


def hash_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file -- used for dataset versioning."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_mapping(mapping: dict[str, Any]) -> str:
    """Stable hash of a JSON-serialisable mapping (key order independent)."""
    return hash_text(json.dumps(mapping, sort_keys=True, default=str))


def fingerprint(*parts: Any) -> str:
    """Short fingerprint used for alert de-duplication."""
    return hash_text("|".join(str(p) for p in parts))[:16]


# --------------------------------------------------------------------------- #
# Git metadata (best effort -- the platform must run outside a repo too)
# --------------------------------------------------------------------------- #
def git_commit(root: Path | str | None = None) -> str:
    """Current git SHA, or the ``FMOPS_GIT_COMMIT``/CI value, or ``unknown``."""
    for env_var in ("FMOPS_GIT_COMMIT", "GIT_COMMIT", "GITHUB_SHA"):
        value = os.environ.get(env_var)
        if value:
            return value.strip()
    try:
        # Fixed argv, shell=False, short timeout. git is resolved from PATH
        # deliberately: pinning an absolute path would break containers.
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - PATH lookup is intended
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def git_branch(root: Path | str | None = None) -> str:
    for env_var in ("GITHUB_REF_NAME", "FMOPS_GIT_BRANCH"):
        value = os.environ.get(env_var)
        if value:
            return value.strip()
    try:
        # Fixed argv, shell=False, short timeout. git is resolved from PATH
        # deliberately: pinning an absolute path would break containers.
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


# --------------------------------------------------------------------------- #
# Numbers / JSON safety
# --------------------------------------------------------------------------- #
def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to a finite float; NaN and inf collapse to ``default``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars into JSON-safe Python types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return safe_float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    # numpy / pandas scalars expose .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except (ValueError, TypeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return jsonable(tolist())
        except (ValueError, TypeError):
            pass
    return str(value)


def write_json(path: Path | str, payload: Any, indent: int = 2) -> Path:
    """Atomically write JSON (temp file + replace) so readers never see a partial file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(jsonable(payload), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def read_json(path: Path | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("json.read_failed", extra={"path": str(p), "error": str(exc)})
        return default


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile without pulling in numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return float(ordered[int(k)])
    return float(ordered[lower] * (upper - k) + ordered[upper] * (k - lower))


def truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def chunked(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]
