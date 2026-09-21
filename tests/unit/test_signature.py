"""Model signatures: the input contract recorded at training time.

These pin the rules a prediction request is held to, and how a two-class target
is resolved -- both of which used to be hard-coded to the loan dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.config import DataConfig
from app.core.signature import (
    ModelSignature,
    SignatureError,
    build_signature,
    encode_target,
    resolve_class_labels,
    validate_model_name,
)


def _config(**update) -> DataConfig:
    base = DataConfig().model_copy(
        update={
            "contract": "inferred",
            "dataset_name": "uploaded",
            "target_column": "churned",
            "id_column": "customer_id",
            "timestamp_column": "",
            "numeric_features": ["tenure", "spend"],
            "categorical_features": ["plan"],
        }
    )
    return base.model_copy(update=update)


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "tenure": rng.integers(1, 60, n),
            "spend": rng.normal(50, 10, n).round(2),
            "plan": rng.choice(["basic", "pro"], n),
            "churned": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        }
    )


# --------------------------------------------------------------------------- #
# Class labels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 1, 1, 0], ["0", "1"]),
        ([0.0, 1.0, 1.0], ["0", "1"]),
        (["yes", "no", "no"], ["no", "yes"]),
        (["True", "False", "False"], ["False", "True"]),
        ([True, False, False], ["false", "true"]),
    ],
)
def test_affirmative_pairs_resolve_without_guessing(values, expected):
    labels, rule = resolve_class_labels(pd.Series(values))
    assert labels == expected
    assert rule


def test_an_unrecognised_pair_takes_the_rarer_class_as_positive_and_says_so():
    series = pd.Series(["stayed"] * 90 + ["left"] * 10)
    labels, rule = resolve_class_labels(series)
    assert labels == ["stayed", "left"]
    assert "rarer" in rule


def test_an_explicit_positive_class_wins():
    series = pd.Series(["stayed"] * 90 + ["left"] * 10)
    labels, rule = resolve_class_labels(series, ["left", "stayed"])
    assert labels == ["left", "stayed"]
    assert rule == "chosen explicitly"


@pytest.mark.parametrize("values", [[1, 1, 1], [1, 2, 3], ["a", "b", "c", "d"]])
def test_a_target_without_exactly_two_classes_is_refused(values):
    with pytest.raises(SignatureError, match="exactly two classes"):
        resolve_class_labels(pd.Series(values))


def test_encoding_uses_the_recorded_labels_and_refuses_strays():
    assert encode_target(pd.Series(["no", "yes", "yes"]), ["no", "yes"]).tolist() == [0, 1, 1]
    with pytest.raises(SignatureError, match="outside"):
        encode_target(pd.Series(["no", "maybe"]), ["no", "yes"])


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_signature_records_kinds_ranges_and_categories():
    sig = build_signature(_frame(), _config())
    assert sig.target == "churned"
    assert sig.class_labels == ["no", "yes"]
    assert sig.numeric_features == ["tenure", "spend"]
    assert sig.categorical_features == ["plan"]
    tenure = sig.feature("tenure")
    assert tenure.minimum is not None and tenure.maximum is not None
    assert set(sig.feature("plan").categories) == {"basic", "pro"}
    assert sig.id_column == "customer_id"


def test_signature_round_trips_to_the_data_config_it_came_from():
    cfg = _config()
    rebuilt = build_signature(_frame(), cfg).to_data_config(DataConfig())
    assert rebuilt.contract == "inferred"
    assert rebuilt.target_column == "churned"
    assert rebuilt.class_labels == ["no", "yes"]
    assert rebuilt.feature_columns == cfg.feature_columns


def test_reference_contract_keeps_encoding_labels_and_adds_display_names(settings):
    from app.data.generator import GenerationSpec, generate_dataset

    frame = generate_dataset(GenerationSpec(n_rows=300, seed=5))
    sig = build_signature(frame, settings.data)
    assert sig.class_labels == ["0", "1"], "encoding labels must match the column"
    assert sig.labels == ["no_default", "default"]
    # Rebuilding the contract must still encode 0/1 numerically.
    assert sig.to_data_config(settings.data).class_labels is None


# --------------------------------------------------------------------------- #
# The serving contract
# --------------------------------------------------------------------------- #
@pytest.fixture
def signature() -> ModelSignature:
    return build_signature(_frame(), _config())


def test_a_valid_record_passes_clean(signature):
    clean, notes = signature.check_record(signature.example())
    assert set(clean) == set(signature.feature_columns)
    assert notes == []


def test_unknown_and_missing_fields_are_reported_together(signature):
    record = signature.example()
    record.pop("tenure")
    record["tenur"] = 12
    with pytest.raises(SignatureError) as info:
        signature.check_record(record)
    message = str(info.value)
    assert "unknown field(s): tenur" in message
    assert "missing field(s): tenure" in message


def test_types_are_enforced(signature):
    record = signature.example() | {"tenure": "twelve"}
    with pytest.raises(SignatureError, match="tenure: expected a number"):
        signature.check_record(record)
    record = signature.example() | {"spend": True}
    with pytest.raises(SignatureError, match="spend: expected a number"):
        signature.check_record(record)
    record = signature.example() | {"spend": float("nan")}
    with pytest.raises(SignatureError, match="finite"):
        signature.check_record(record)


def test_out_of_range_unseen_and_null_are_accepted_but_reported(signature):
    record = signature.example() | {"tenure": 10_000, "plan": "enterprise", "spend": None}
    clean, notes = signature.check_record(record)
    assert clean["spend"] is None
    joined = " ".join(notes)
    assert "outside the training range" in joined
    assert "was not seen in training" in joined
    assert "imputed" in joined


def test_the_id_column_may_be_sent_but_is_not_a_feature(signature):
    clean, _ = signature.check_record(signature.example() | {"customer_id": "c1"})
    assert "customer_id" not in clean


def test_labels_decode_predictions(signature):
    assert signature.label_for(1) == "yes"
    assert signature.label_for(0) == "no"


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["churn_model", "fraud-v2", "abc"])
def test_valid_model_names(name):
    assert validate_model_name(name) == name


@pytest.mark.parametrize("name", ["", "ab", "Churn", "1model", "a b", "../etc", "x" * 65])
def test_invalid_model_names(name):
    with pytest.raises(SignatureError):
        validate_model_name(name)
