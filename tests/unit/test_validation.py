"""Data validation tests -- the pipeline's first gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DataValidationError
from app.data.validation import (
    NativeValidationEngine,
    build_schema,
    validate_dataframe,
    validate_or_raise,
)
from app.schemas.common import ValidationSeverity

pytestmark = pytest.mark.unit


def test_valid_dataset_passes(valid_frame, settings):
    report = validate_dataframe(valid_frame, "loan_default", "v-test", settings)
    assert report.passed
    assert report.errors == []
    assert report.n_rows == len(valid_frame)
    assert len(report.results) > 30, "expected a substantive suite, not a token check"


def test_invalid_dataset_fails_and_names_every_problem(invalid_frame, settings):
    report = validate_dataframe(invalid_frame, "loan_default", "v-bad", settings)
    assert not report.passed

    failed = " ".join(r.message for r in report.errors)
    # Each corruption the generator injected must be caught by name.
    assert "null" in failed
    assert "duplicate" in failed
    assert "outside" in failed


def test_validate_or_raise_carries_the_report(invalid_frame, settings):
    with pytest.raises(DataValidationError) as excinfo:
        validate_or_raise(invalid_frame, "loan_default", "v-bad", settings)

    error = excinfo.value
    assert error.code == "data_validation_failed"
    assert error.details["failed"], "the exception must list the failed expectations"
    assert error.details["report"]["passed"] is False


def test_missing_column_is_a_blocking_error(valid_frame, settings):
    frame = valid_frame.drop(columns=["credit_score"])
    report = validate_dataframe(frame, "loan_default", None, settings)
    assert not report.passed
    assert any("missing required columns" in r.message for r in report.errors)


def test_unexpected_column_is_a_warning_not_an_error(valid_frame, settings):
    frame = valid_frame.copy()
    frame["experimental_feature"] = 1.0
    report = validate_dataframe(frame, "loan_default", None, settings)
    # Extra columns are dropped by the preprocessor, so they must not block.
    assert report.passed
    assert any(
        r.expectation == "no_unexpected_columns" and not r.success for r in report.results
    )


def test_out_of_range_values_are_caught(valid_frame, settings):
    frame = valid_frame.copy()
    frame.loc[frame.index[:20], "age"] = -3
    report = validate_dataframe(frame, "loan_default", None, settings)
    assert not report.passed
    assert any(r.column == "age" and not r.success for r in report.errors)


def test_unknown_category_is_caught(valid_frame, settings):
    frame = valid_frame.copy()
    frame.loc[frame.index[:15], "employment_type"] = "freelance_martian"
    report = validate_dataframe(frame, "loan_default", None, settings)
    assert not report.passed
    assert any(
        r.expectation == "column_values_in_set" and r.column == "employment_type"
        for r in report.errors
    )


def test_single_class_target_is_caught(valid_frame, settings):
    frame = valid_frame.copy()
    frame["default"] = 0
    report = validate_dataframe(frame, "loan_default", None, settings)
    assert not report.passed
    assert any("class" in r.message.lower() for r in report.errors)


def test_too_few_rows_is_caught(valid_frame, settings):
    report = validate_dataframe(valid_frame.head(10), "loan_default", None, settings)
    assert not report.passed
    assert any(r.expectation == "table_row_count_above_minimum" for r in report.errors)


def test_low_cardinality_columns_skip_the_outlier_check(valid_frame, settings):
    """loan_term_months has 7 levels; a robust z-score there is noise, not signal."""
    report = validate_dataframe(valid_frame, "loan_default", None, settings)
    outlier_columns = {
        r.column
        for r in report.results
        if r.expectation == "column_outlier_fraction_below_limit"
    }
    assert "loan_term_months" not in outlier_columns
    assert "annual_income" in outlier_columns


def test_outliers_are_warnings_not_blocking(valid_frame, settings):
    frame = valid_frame.copy()
    idx = frame.index[:80]
    frame.loc[idx, "annual_income"] = frame.loc[idx, "annual_income"] * 50
    report = validate_dataframe(frame, "loan_default", None, settings)
    outlier_results = [
        r
        for r in report.results
        if r.expectation == "column_outlier_fraction_below_limit" and not r.success
    ]
    assert outlier_results
    assert all(r.severity == ValidationSeverity.WARNING for r in outlier_results)


def test_fail_on_warning_promotes_warnings_to_blocking(valid_frame, settings):
    frame = valid_frame.copy()
    frame["surprise_column"] = 1
    config = settings.validation.model_copy(update={"fail_on_warning": True})
    engine = NativeValidationEngine(config, settings.data)
    report = engine.validate(frame, build_schema(settings.data), "loan_default")
    assert not report.passed


def test_report_summary_profiles_every_column(valid_frame, settings):
    report = validate_dataframe(valid_frame, "loan_default", None, settings)
    summary = report.summary
    assert summary["n_rows"] == len(valid_frame)
    assert "class_distribution" in summary
    assert summary["columns"]["credit_score"]["mean"] > 0


def test_render_text_is_human_readable(invalid_frame, settings):
    report = validate_dataframe(invalid_frame, "loan_default", "v-bad", settings)
    text = report.render_text()
    assert "FAILED" in text
    assert "errors:" in text


def test_nan_heavy_column_fails_the_missing_check(valid_frame, settings):
    frame = valid_frame.copy()
    frame.loc[frame.index[: int(len(frame) * 0.4)], "credit_score"] = np.nan
    report = validate_dataframe(frame, "loan_default", None, settings)
    assert not report.passed
    assert any(
        r.expectation == "column_missing_fraction_below_limit" and r.column == "credit_score"
        for r in report.errors
    )


def test_empty_frame_does_not_crash(settings):
    report = validate_dataframe(pd.DataFrame(), "loan_default", None, settings)
    assert not report.passed
