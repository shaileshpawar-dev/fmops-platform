"""Feature engineering and the preprocessing pipeline.

Design rule: **every transformation lives inside the fitted sklearn pipeline**
that gets serialised with the model. Nothing is applied at training time that is
not also applied automatically at inference time, which removes the single most
common source of training/serving skew.

The derived features here are deliberately simple ratios and flags -- the point
of the project is the platform around the model, not feature cleverness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from app.core.config import DataConfig, get_settings
from app.core.logging import get_logger
from app.core.signature import encode_target, stringify_label

logger = get_logger(__name__)

# Features derived inside the pipeline from the raw inputs.
DERIVED_FEATURES: tuple[str, ...] = (
    "loan_to_income",
    "monthly_payment",
    "payment_to_income",
    "credit_score_band",
    "late_payment_flag",
    "high_utilization_flag",
    "income_per_credit_line",
    "tenure_ratio",
)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Pure, stateless, and safe on partial frames.

    Called both by the pipeline (via :class:`FunctionTransformer`) and directly
    by tests. Never mutates its argument.
    """
    out = frame.copy()

    income = pd.to_numeric(out.get("annual_income"), errors="coerce")
    loan = pd.to_numeric(out.get("loan_amount"), errors="coerce")
    term = pd.to_numeric(out.get("loan_term_months"), errors="coerce")
    score = pd.to_numeric(out.get("credit_score"), errors="coerce")
    late = pd.to_numeric(out.get("num_late_payments_12m"), errors="coerce")
    util = pd.to_numeric(out.get("credit_utilization"), errors="coerce")
    lines = pd.to_numeric(out.get("num_credit_lines"), errors="coerce")
    tenure = pd.to_numeric(out.get("employment_years"), errors="coerce")
    age = pd.to_numeric(out.get("age"), errors="coerce")

    safe_income = income.replace(0, np.nan)
    safe_term = term.replace(0, np.nan)

    out["loan_to_income"] = loan / safe_income
    out["monthly_payment"] = loan / safe_term
    out["payment_to_income"] = (out["monthly_payment"] * 12.0) / safe_income
    # Ordinal band rather than one-hot: monotone in risk, cheap to explain.
    out["credit_score_band"] = pd.cut(
        score,
        bins=[-np.inf, 580, 670, 740, 800, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype("float")
    out["late_payment_flag"] = (late > 0).astype(float)
    out["high_utilization_flag"] = (util > 0.6).astype(float)
    out["income_per_credit_line"] = income / lines.replace(0, np.nan)
    out["tenure_ratio"] = tenure / age.replace(0, np.nan)

    # Ratios on degenerate inputs produce inf; the imputer only handles NaN.
    for column in DERIVED_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return out


def categories_as_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Categorical inputs in the text form the signature records. Missing stays missing."""
    out = pd.DataFrame(frame).copy()
    for column in out.columns:
        out[column] = (
            out[column]
            .map(lambda v: np.nan if pd.isna(v) else stringify_label(v).strip())
            .astype(object)
        )
    return out


def build_preprocessor(
    data_config: DataConfig | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build the ColumnTransformer plus the numeric/categorical column lists.

    The loan-specific derived ratios exist only for the reference contract. A
    user dataset is preprocessed from its own columns and nothing else -- the
    engineered loan features would be all-NaN there at best.
    """
    cfg = data_config or get_settings().data
    derived = DERIVED_FEATURES if uses_reference_engineering(cfg) else ()
    numeric = [*cfg.numeric_features, *derived]
    categorical = list(cfg.categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            # Categories are compared as text, exactly as the serving contract
            # sends them. Without this a column of integer codes or booleans is
            # fitted on 3/True and served "3"/"true", every served value encodes
            # as unknown, and predictions are silently wrong.
            (
                "as_text",
                FunctionTransformer(
                    categories_as_text, validate=False, feature_names_out="one-to-one"
                ),
            ),
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Unknown categories at serving time must not explode -- they encode
            # to all-zeros. This is why a *new* category shows up as drift
            # rather than as a 500.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer, numeric, categorical


def build_feature_pipeline(data_config: DataConfig | None = None) -> Pipeline:
    """Feature engineering + preprocessing, as a single fittable step.

    The estimator is appended by :mod:`app.training.train`; keeping this
    function separate means the same feature stack is reused by every algorithm
    and by the drift reference profile.
    """
    cfg = data_config or get_settings().data
    transformer, _, _ = build_preprocessor(cfg)
    if not uses_reference_engineering(cfg):
        return Pipeline(steps=[("preprocess", transformer)])
    return Pipeline(
        steps=[
            (
                "engineer",
                FunctionTransformer(engineer_features, validate=False, feature_names_out=None),
            ),
            ("preprocess", transformer),
        ]
    )


def uses_reference_engineering(data_config: DataConfig) -> bool:
    """Whether the loan domain features apply to this contract."""
    return data_config.contract == "loan_reference"


def split_features_target(
    frame: pd.DataFrame, data_config: DataConfig | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Split into the model input frame and the label series.

    Identifier and timestamp columns are dropped: they carry no signal and
    leaking them would make the model non-reproducible across time windows.
    """
    cfg = data_config or get_settings().data
    if cfg.target_column not in frame.columns:
        raise KeyError(f"target column {cfg.target_column!r} not present")
    unlabelled = frame[cfg.target_column].isna()
    if unlabelled.any():
        # A row without a label cannot be learned from. Validation reports the
        # fraction; here the rows are dropped and the count is logged.
        logger.warning(
            "preprocessing.unlabelled_rows_dropped",
            extra={"rows": int(unlabelled.sum()), "target": cfg.target_column},
        )
        frame = frame.loc[~unlabelled]
    drop = [cfg.target_column, cfg.id_column, cfg.timestamp_column]
    features = frame.drop(columns=[c for c in drop if c in frame.columns])
    # Encoded through the recorded class labels, so a "yes"/"no" target trains
    # rather than failing a numeric cast. Unknown values raise instead of
    # silently becoming a class.
    target = encode_target(frame[cfg.target_column], cfg.class_labels)
    return features, target


def prepare_inference_frame(
    records: list[dict] | pd.DataFrame, data_config: DataConfig | None = None
) -> pd.DataFrame:
    """Coerce API payloads into the exact column set the pipeline expects.

    Missing raw columns are inserted as NaN rather than raising, so the
    pipeline's imputer handles them; unexpected columns are dropped.
    """
    cfg = data_config or get_settings().data
    frame = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
    for column in cfg.feature_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[list(cfg.feature_columns)]


def feature_names(pipeline: Pipeline) -> list[str]:
    """Output feature names of a fitted preprocessing pipeline."""
    try:
        transformer = pipeline.named_steps["preprocess"]
        return [str(n) for n in transformer.get_feature_names_out()]
    except (KeyError, AttributeError, ValueError) as exc:
        logger.warning("preprocessing.feature_names_unavailable", extra={"error": str(exc)})
        return []
