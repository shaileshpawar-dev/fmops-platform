"""Schemas for data validation and dataset versioning."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.utils import utcnow_iso
from app.schemas.common import ValidationSeverity


class ExpectationResult(BaseModel):
    """Outcome of a single expectation against a dataset."""

    expectation: str
    column: str | None = None
    success: bool
    severity: ValidationSeverity = ValidationSeverity.ERROR
    observed: Any = None
    expected: Any = None
    message: str = ""

    @property
    def blocking(self) -> bool:
        return not self.success and self.severity == ValidationSeverity.ERROR


class ValidationReport(BaseModel):
    """Aggregate result of a validation suite run.

    ``passed`` is the gate the training pipeline honours: when it is False the
    pipeline stops and no model is trained on the data.
    """

    dataset_name: str
    dataset_version: str | None = None
    engine: str = "native"
    n_rows: int = 0
    n_columns: int = 0
    passed: bool = True
    results: list[ExpectationResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)

    @property
    def errors(self) -> list[ExpectationResult]:
        return [r for r in self.results if r.blocking]

    @property
    def warnings(self) -> list[ExpectationResult]:
        return [
            r
            for r in self.results
            if not r.success and r.severity == ValidationSeverity.WARNING
        ]

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 1.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def render_text(self) -> str:
        """Human-readable report, printed by the CLI and attached to alerts."""
        lines = [
            f"Data validation report -- {self.dataset_name}"
            + (f" @ {self.dataset_version}" if self.dataset_version else ""),
            f"  engine       : {self.engine}",
            f"  rows/columns : {self.n_rows} x {self.n_columns}",
            f"  expectations : {len(self.results)} "
            f"({self.n_failed} failed, {self.success_rate:.0%} passed)",
            f"  status       : {'PASSED' if self.passed else 'FAILED'}",
        ]
        if self.errors:
            lines.append("  errors:")
            lines.extend(f"    [x] {r.message}" for r in self.errors)
        if self.warnings:
            lines.append("  warnings:")
            lines.extend(f"    [!] {r.message}" for r in self.warnings)
        return "\n".join(lines)


class DatasetVersion(BaseModel):
    """An immutable, content-addressed snapshot of a dataset."""

    version: str
    dataset_name: str
    path: str
    content_hash: str
    n_rows: int
    n_columns: int
    columns: list[str] = Field(default_factory=list)
    size_bytes: int = 0
    git_commit: str = "unknown"
    dvc_tracked: bool = False
    parent_version: str | None = None
    description: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)


class DatasetProfile(BaseModel):
    """Column-level statistics used as the drift reference."""

    dataset_name: str
    dataset_version: str | None = None
    n_rows: int
    numeric: dict[str, dict[str, float]] = Field(default_factory=dict)
    categorical: dict[str, dict[str, float]] = Field(default_factory=dict)
    target: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)
