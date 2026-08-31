"""Automated data validation -- the first gate in the pipeline.

Two engines are available behind one interface:

``native``
    The default. A self-contained expectation engine (schema, nullity,
    duplicates, dtypes, ranges, outliers, class balance, cardinality) with no
    dependencies beyond pandas/numpy. Always available, fully unit-tested.

``great_expectations``
    Runs the equivalent suite through Great Expectations when the ``[quality]``
    extra is installed, for teams that already standardise on GE Data Docs.

Both produce the same :class:`~app.schemas.data.ValidationReport`, so the
pipeline gate does not care which ran.

Gate behaviour: a report with ``passed=False`` stops the pipeline. The caller
uses :func:`validate_or_raise` to turn that into a
:class:`~app.core.exceptions.DataValidationError` carrying the full report.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import DataConfig, Settings, ValidationConfig, get_settings
from app.core.exceptions import DataValidationError, DependencyMissingError
from app.core.logging import get_logger
from app.core.utils import safe_float
from app.schemas.common import ValidationSeverity
from app.schemas.data import ExpectationResult, ValidationReport

logger = get_logger(__name__)


@dataclass(frozen=True)
class ColumnSpec:
    """Declared contract for one column."""

    name: str
    dtype: str  # "numeric" | "integer" | "categorical" | "string" | "date"
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[str, ...] | None = None
    max_missing_fraction: float | None = None
    unique: bool = False


def build_schema(
    data_config: DataConfig | None = None,
) -> dict[str, ColumnSpec]:
    """The declared schema for the reference loan-default dataset.

    Ranges mirror the pydantic constraints on the inference request schema, so
    an input rejected at training time is also rejected at serving time.
    """
    cfg = data_config or get_settings().data
    specs = [
        ColumnSpec(cfg.id_column, "string", unique=True),
        ColumnSpec(cfg.timestamp_column, "date"),
        ColumnSpec("age", "numeric", minimum=18, maximum=100),
        ColumnSpec("annual_income", "numeric", minimum=0, maximum=5_000_000),
        ColumnSpec("loan_amount", "numeric", minimum=100, maximum=2_000_000),
        ColumnSpec("loan_term_months", "integer", minimum=6, maximum=480),
        ColumnSpec("credit_score", "numeric", minimum=300, maximum=850),
        ColumnSpec("debt_to_income", "numeric", minimum=0, maximum=3.0),
        ColumnSpec("employment_years", "numeric", minimum=0, maximum=60),
        ColumnSpec("num_credit_lines", "integer", minimum=0, maximum=60),
        ColumnSpec("num_late_payments_12m", "integer", minimum=0, maximum=40),
        ColumnSpec("credit_utilization", "numeric", minimum=0, maximum=2.0),
        ColumnSpec(
            "employment_type",
            "categorical",
            allowed_values=(
                "salaried",
                "self_employed",
                "contract",
                "retired",
                "unemployed",
            ),
        ),
        ColumnSpec(
            "housing_status",
            "categorical",
            allowed_values=("own", "mortgage", "rent", "other"),
        ),
        ColumnSpec(
            "loan_purpose",
            "categorical",
            allowed_values=(
                "debt_consolidation",
                "home_improvement",
                "auto",
                "medical",
                "business",
            ),
        ),
        ColumnSpec(
            "region",
            "categorical",
            allowed_values=("north", "south", "east", "west", "central"),
        ),
        ColumnSpec(cfg.target_column, "integer", minimum=0, maximum=1),
    ]
    return {spec.name: spec for spec in specs}


class ValidationEngine(ABC):
    """Runs an expectation suite over a dataframe."""

    name: str = "abstract"

    @abstractmethod
    def validate(
        self,
        frame: pd.DataFrame,
        schema: dict[str, ColumnSpec],
        dataset_name: str,
        dataset_version: str | None = None,
    ) -> ValidationReport: ...


class NativeValidationEngine(ValidationEngine):
    """Dependency-free expectation engine. The platform default."""

    name = "native"

    def __init__(
        self,
        config: ValidationConfig | None = None,
        data_config: DataConfig | None = None,
    ) -> None:
        settings = get_settings()
        self.config = config or settings.validation
        self.data_config = data_config or settings.data

    # -- individual expectation groups -------------------------------------- #
    def _check_shape(self, frame: pd.DataFrame) -> list[ExpectationResult]:
        return [
            ExpectationResult(
                expectation="table_row_count_above_minimum",
                success=len(frame) >= self.config.min_rows,
                observed=len(frame),
                expected=f">= {self.config.min_rows}",
                message=(f"dataset has {len(frame)} rows, minimum is {self.config.min_rows}"),
            )
        ]

    def _check_schema(
        self, frame: pd.DataFrame, schema: dict[str, ColumnSpec]
    ) -> list[ExpectationResult]:
        results: list[ExpectationResult] = []
        missing = [
            name for name, spec in schema.items() if spec.required and name not in frame
        ]
        results.append(
            ExpectationResult(
                expectation="expected_columns_present",
                success=not missing,
                observed=sorted(missing),
                expected="all declared columns",
                message=(
                    f"missing required columns: {', '.join(missing)}"
                    if missing
                    else "all required columns present"
                ),
            )
        )
        unexpected = [c for c in frame.columns if c not in schema]
        results.append(
            ExpectationResult(
                expectation="no_unexpected_columns",
                success=not unexpected,
                severity=ValidationSeverity.WARNING,
                observed=sorted(unexpected),
                expected="none",
                message=(
                    f"unexpected columns present: {', '.join(unexpected)}"
                    if unexpected
                    else "no unexpected columns"
                ),
            )
        )
        return results

    def _check_dtypes(
        self, frame: pd.DataFrame, schema: dict[str, ColumnSpec]
    ) -> list[ExpectationResult]:
        results: list[ExpectationResult] = []
        for name, spec in schema.items():
            if name not in frame:
                continue
            series = frame[name]
            if spec.dtype in ("numeric", "integer"):
                ok = pd.api.types.is_numeric_dtype(series)
                # A column that is object-typed but fully coercible is a warning,
                # not an error -- CSV round-trips do this constantly.
                severity = ValidationSeverity.ERROR
                if not ok:
                    coerced = pd.to_numeric(series, errors="coerce")
                    if coerced.notna().sum() >= series.notna().sum():
                        severity = ValidationSeverity.WARNING
                results.append(
                    ExpectationResult(
                        expectation="column_type_matches",
                        column=name,
                        success=ok,
                        severity=severity,
                        observed=str(series.dtype),
                        expected=spec.dtype,
                        message=f"{name} should be {spec.dtype}, found {series.dtype}",
                    )
                )
            elif spec.dtype == "date":
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                bad = int(parsed.isna().sum() - series.isna().sum())
                results.append(
                    ExpectationResult(
                        expectation="column_parses_as_date",
                        column=name,
                        success=bad == 0,
                        observed=bad,
                        expected=0,
                        message=f"{name} has {bad} unparseable date values",
                    )
                )
        return results

    def _check_missing(
        self, frame: pd.DataFrame, schema: dict[str, ColumnSpec]
    ) -> list[ExpectationResult]:
        results: list[ExpectationResult] = []
        for name, spec in schema.items():
            if name not in frame:
                continue
            fraction = float(frame[name].isna().mean())
            limit = (
                spec.max_missing_fraction
                if spec.max_missing_fraction is not None
                else self.config.max_missing_fraction
            )
            results.append(
                ExpectationResult(
                    expectation="column_missing_fraction_below_limit",
                    column=name,
                    success=fraction <= limit,
                    observed=round(fraction, 5),
                    expected=f"<= {limit}",
                    message=(f"{name} is {fraction:.2%} null, limit is {limit:.2%}"),
                )
            )
        return results

    def _check_duplicates(
        self, frame: pd.DataFrame, schema: dict[str, ColumnSpec]
    ) -> list[ExpectationResult]:
        results: list[ExpectationResult] = []
        dup_fraction = float(frame.duplicated().mean()) if len(frame) else 0.0
        results.append(
            ExpectationResult(
                expectation="duplicate_row_fraction_below_limit",
                success=dup_fraction <= self.config.max_duplicate_fraction,
                observed=round(dup_fraction, 5),
                expected=f"<= {self.config.max_duplicate_fraction}",
                message=(
                    f"{dup_fraction:.2%} of rows are exact duplicates, "
                    f"limit is {self.config.max_duplicate_fraction:.2%}"
                ),
            )
        )
        for name, spec in schema.items():
            if spec.unique and name in frame:
                dupes = int(frame[name].duplicated().sum())
                results.append(
                    ExpectationResult(
                        expectation="column_values_unique",
                        column=name,
                        success=dupes == 0,
                        observed=dupes,
                        expected=0,
                        message=f"{name} has {dupes} duplicated identifiers",
                    )
                )
        return results

    def _check_ranges(
        self, frame: pd.DataFrame, schema: dict[str, ColumnSpec]
    ) -> list[ExpectationResult]:
        results: list[ExpectationResult] = []
        for name, spec in schema.items():
            if name not in frame:
                continue
            if spec.minimum is None and spec.maximum is None:
                continue
            series = pd.to_numeric(frame[name], errors="coerce").dropna()
            if series.empty:
                continue
            violations = 0
            if spec.minimum is not None:
                violations += int((series < spec.minimum).sum())
            if spec.maximum is not None:
                violations += int((series > spec.maximum).sum())
            results.append(
                ExpectationResult(
                    expectation="column_values_between",
                    column=name,
                    success=violations == 0,
                    observed=violations,
                    expected=f"[{spec.minimum}, {spec.maximum}]",
                    message=(
                        f"{name} has {violations} values outside "
                        f"[{spec.minimum}, {spec.maximum}]"
                    ),
                )
            )
        return results

    def _check_allowed_values(
        self, frame: pd.DataFrame, schema: dict[str, ColumnSpec]
    ) -> list[ExpectationResult]:
        results: list[ExpectationResult] = []
        for name, spec in schema.items():
            if name not in frame or not spec.allowed_values:
                continue
            observed = set(frame[name].dropna().astype(str).unique())
            unexpected = sorted(observed - set(spec.allowed_values))
            results.append(
                ExpectationResult(
                    expectation="column_values_in_set",
                    column=name,
                    success=not unexpected,
                    observed=unexpected[:10],
                    expected=list(spec.allowed_values),
                    message=(
                        f"{name} contains unexpected categories: "
                        f"{', '.join(unexpected[:10])}"
                        if unexpected
                        else f"{name} categories all recognised"
                    ),
                )
            )
        return results

    def _check_outliers(
        self, frame: pd.DataFrame, schema: dict[str, ColumnSpec]
    ) -> list[ExpectationResult]:
        """Robust (median/MAD) z-score outlier share per numeric column."""
        results: list[ExpectationResult] = []
        for name, spec in schema.items():
            if name not in frame or spec.dtype not in ("numeric", "integer"):
                continue
            if name in (self.data_config.target_column,):
                continue
            series = pd.to_numeric(frame[name], errors="coerce").dropna()
            if len(series) < 20:
                continue
            # Skip low-cardinality discrete columns (loan_term_months, counts).
            # A robust z-score on a handful of distinct levels flags the rare
            # levels as outliers, which is noise rather than a data-quality
            # signal; range and value-set checks already cover those columns.
            if series.nunique() <= 15:
                continue
            median = float(series.median())
            mad = float((series - median).abs().median())
            if mad <= 0:
                std = float(series.std() or 0.0)
                if std <= 0:
                    continue
                z = (series - float(series.mean())).abs() / std
            else:
                z = 0.6745 * (series - median).abs() / mad
            fraction = float((z > self.config.outlier_z_threshold).mean())
            results.append(
                ExpectationResult(
                    expectation="column_outlier_fraction_below_limit",
                    column=name,
                    success=fraction <= self.config.max_outlier_fraction,
                    severity=ValidationSeverity.WARNING,
                    observed=round(fraction, 5),
                    expected=f"<= {self.config.max_outlier_fraction}",
                    message=(
                        f"{name} has {fraction:.2%} robust-z outliers beyond "
                        f"{self.config.outlier_z_threshold}"
                    ),
                )
            )
        return results

    def _check_class_balance(self, frame: pd.DataFrame) -> list[ExpectationResult]:
        target = self.data_config.target_column
        if target not in frame:
            return []
        series = pd.to_numeric(frame[target], errors="coerce").dropna()
        if series.empty:
            return [
                ExpectationResult(
                    expectation="target_present",
                    column=target,
                    success=False,
                    observed=0,
                    expected="> 0 labelled rows",
                    message="target column is entirely null",
                )
            ]
        classes = sorted(series.unique().tolist())
        positive_rate = float((series == 1).mean())
        results = [
            ExpectationResult(
                expectation="target_is_binary",
                column=target,
                success=set(classes).issubset({0, 1}) and len(classes) == 2,
                observed=classes,
                expected=[0, 1],
                message=f"target classes observed: {classes}",
            ),
            ExpectationResult(
                expectation="class_distribution_within_bounds",
                column=target,
                success=(
                    self.config.min_class_fraction
                    <= positive_rate
                    <= self.config.max_class_fraction
                ),
                observed=round(positive_rate, 5),
                expected=(
                    f"[{self.config.min_class_fraction}, " f"{self.config.max_class_fraction}]"
                ),
                message=(
                    f"positive class rate is {positive_rate:.2%}, expected within "
                    f"[{self.config.min_class_fraction:.1%}, "
                    f"{self.config.max_class_fraction:.1%}]"
                ),
            ),
        ]
        return results

    # -- entry point --------------------------------------------------------- #
    def validate(
        self,
        frame: pd.DataFrame,
        schema: dict[str, ColumnSpec],
        dataset_name: str,
        dataset_version: str | None = None,
    ) -> ValidationReport:
        results: list[ExpectationResult] = []
        results += self._check_shape(frame)
        results += self._check_schema(frame, schema)
        results += self._check_dtypes(frame, schema)
        results += self._check_missing(frame, schema)
        results += self._check_duplicates(frame, schema)
        results += self._check_ranges(frame, schema)
        results += self._check_allowed_values(frame, schema)
        results += self._check_outliers(frame, schema)
        results += self._check_class_balance(frame)

        blocking = [r for r in results if r.blocking]
        warnings = [
            r for r in results if not r.success and r.severity == ValidationSeverity.WARNING
        ]
        passed = not blocking and not (self.config.fail_on_warning and warnings)

        report = ValidationReport(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            engine=self.name,
            n_rows=len(frame),
            n_columns=frame.shape[1],
            passed=passed,
            results=results,
            summary=_summarise(frame, self.data_config),
        )
        logger.info(
            "validation.completed",
            extra={
                "dataset": dataset_name,
                "engine": self.name,
                "passed": passed,
                "expectations": len(results),
                "failed": report.n_failed,
                "errors": len(blocking),
                "warnings": len(warnings),
            },
        )
        return report


class GreatExpectationsEngine(ValidationEngine):
    """Great Expectations adapter.

    Requires the ``[quality]`` extra. It runs the same contract as the native
    engine through GE so the resulting suite can feed GE Data Docs. If GE is not
    installed this raises :class:`DependencyMissingError` rather than silently
    falling back -- an operator who asked for GE should be told it is absent.
    """

    name = "great_expectations"

    def validate(
        self,
        frame: pd.DataFrame,
        schema: dict[str, ColumnSpec],
        dataset_name: str,
        dataset_version: str | None = None,
    ) -> ValidationReport:
        try:
            import great_expectations as gx  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError(
                "great_expectations is not installed; install the [quality] extra "
                "or set FMOPS_VALIDATION__ENGINE=native",
                engine="great_expectations",
            ) from exc

        context = gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas(name=f"{dataset_name}-source")
        asset = data_source.add_dataframe_asset(name=dataset_name)
        batch_definition = asset.add_batch_definition_whole_dataframe("batch")
        suite = context.suites.add(gx.ExpectationSuite(name=f"{dataset_name}-suite"))

        for expectation in _ge_expectations(gx, schema, self_config=get_settings()):
            suite.add_expectation(expectation)

        validation_definition = context.validation_definitions.add(
            gx.ValidationDefinition(
                name=f"{dataset_name}-validation",
                data=batch_definition,
                suite=suite,
            )
        )
        outcome = validation_definition.run(batch_parameters={"dataframe": frame})

        results: list[ExpectationResult] = []
        for item in outcome.results:
            config = item.expectation_config
            kwargs = dict(config.kwargs)
            results.append(
                ExpectationResult(
                    expectation=config.type,
                    column=kwargs.get("column"),
                    success=bool(item.success),
                    observed=item.result.get("observed_value")
                    or item.result.get("unexpected_percent"),
                    expected={
                        k: v for k, v in kwargs.items() if k not in ("column", "batch_id")
                    },
                    message=(
                        f"{config.type} on {kwargs.get('column', 'table')}: "
                        f"{'passed' if item.success else 'failed'}"
                    ),
                )
            )

        passed = bool(outcome.success)
        report = ValidationReport(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            engine=self.name,
            n_rows=len(frame),
            n_columns=frame.shape[1],
            passed=passed,
            results=results,
            summary=_summarise(frame, get_settings().data),
        )
        logger.info(
            "validation.completed",
            extra={
                "dataset": dataset_name,
                "engine": self.name,
                "passed": passed,
                "expectations": len(results),
                "failed": report.n_failed,
            },
        )
        return report


def _ge_expectations(
    gx, schema: dict[str, ColumnSpec], self_config: Settings
) -> list:  # noqa: ANN001
    """Translate the declared schema into Great Expectations expectations."""
    expectations = [
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=self_config.validation.min_rows
        )
    ]
    for name, spec in schema.items():
        expectations.append(gx.expectations.ExpectColumnToExist(column=name))
        limit = (
            spec.max_missing_fraction
            if spec.max_missing_fraction is not None
            else self_config.validation.max_missing_fraction
        )
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=name, mostly=max(0.0, 1.0 - limit)
            )
        )
        if spec.minimum is not None or spec.maximum is not None:
            expectations.append(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=name, min_value=spec.minimum, max_value=spec.maximum
                )
            )
        if spec.allowed_values:
            expectations.append(
                gx.expectations.ExpectColumnValuesToBeInSet(
                    column=name, value_set=list(spec.allowed_values)
                )
            )
        if spec.unique:
            expectations.append(gx.expectations.ExpectColumnValuesToBeUnique(column=name))
    return expectations


def _summarise(frame: pd.DataFrame, data_config: DataConfig) -> dict[str, Any]:
    """Compact profile attached to every report (also useful in alerts)."""
    summary: dict[str, Any] = {
        "n_rows": len(frame),
        "n_columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()) if len(frame) else 0,
        "total_missing_cells": int(frame.isna().sum().sum()),
        "columns": {},
    }
    for column in frame.columns:
        series = frame[column]
        info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_fraction": round(float(series.isna().mean()), 5),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if not clean.empty:
                info.update(
                    mean=safe_float(clean.mean()),
                    std=safe_float(clean.std()),
                    min=safe_float(clean.min()),
                    p25=safe_float(np.percentile(clean, 25)),
                    median=safe_float(clean.median()),
                    p75=safe_float(np.percentile(clean, 75)),
                    max=safe_float(clean.max()),
                )
        summary["columns"][column] = info
    target = data_config.target_column
    if target in frame:
        counts = frame[target].value_counts(dropna=True).to_dict()
        summary["class_distribution"] = {str(k): int(v) for k, v in counts.items()}
    return summary


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_engine(settings: Settings | None = None) -> ValidationEngine:
    settings = settings or get_settings()
    if settings.validation.engine == "great_expectations":
        return GreatExpectationsEngine()
    return NativeValidationEngine(settings.validation, settings.data)


def validate_dataframe(
    frame: pd.DataFrame,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    settings: Settings | None = None,
) -> ValidationReport:
    """Run the configured engine and return the report (never raises on failure)."""
    settings = settings or get_settings()
    engine = build_engine(settings)
    schema = build_schema(settings.data)
    return engine.validate(
        frame,
        schema,
        dataset_name or settings.data.dataset_name,
        dataset_version,
    )


def validate_or_raise(
    frame: pd.DataFrame,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    settings: Settings | None = None,
) -> ValidationReport:
    """Validate and stop the pipeline if the data is not fit to train on."""
    report = validate_dataframe(frame, dataset_name, dataset_version, settings)
    if not report.passed:
        logger.error(
            "validation.gate_failed",
            extra={
                "dataset": report.dataset_name,
                "errors": [r.message for r in report.errors],
                "n_failed": report.n_failed,
            },
        )
        raise DataValidationError(
            f"data validation failed for {report.dataset_name}: "
            f"{len(report.errors)} blocking expectation(s) failed",
            dataset=report.dataset_name,
            dataset_version=report.dataset_version,
            failed=[r.message for r in report.errors],
            report=report.model_dump(mode="json"),
        )
    return report
