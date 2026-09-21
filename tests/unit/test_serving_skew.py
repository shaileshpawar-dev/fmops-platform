"""Training/serving skew for categorical codes.

A categorical column stored as integers (region codes) or booleans is fitted on
3 / True but arrives at serving as JSON, stringified by the input contract to
"3" / "true". Before the pipeline normalised categories to text itself, every
such served value one-hot encoded as *unknown* -- all zeros -- and the model
returned a confident, silently wrong answer.

The invariant: scoring a row through the serving contract gives exactly the
probability the fitted pipeline gives on the raw training row.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.config import DataConfig
from app.core.signature import build_signature
from app.data.preprocessing import (
    build_feature_pipeline,
    prepare_inference_frame,
    split_features_target,
)


def _frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    region = rng.integers(1, 6, n)  # integer-coded categorical
    promo = rng.random(n) < 0.4  # boolean categorical
    spend = rng.normal(40, 12, n)
    logit = -1.2 + 0.9 * (region == 3) - 0.7 * promo + 0.02 * (spend - 40)
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"region": region, "promo": promo, "spend": spend, "bought": target})


def test_codes_and_booleans_score_identically_through_the_serving_contract():
    frame = _frame()
    cfg = DataConfig().model_copy(
        update={
            "contract": "inferred",
            "dataset_name": "uploaded",
            "target_column": "bought",
            "id_column": "",
            "timestamp_column": "",
            "numeric_features": ["spend"],
            "categorical_features": ["region", "promo"],
        }
    )
    features, target = split_features_target(frame, cfg)
    pipeline = Pipeline(
        [*build_feature_pipeline(cfg).steps, ("estimator", LogisticRegression(max_iter=500))]
    )
    pipeline.fit(features, target)
    signature = build_signature(frame, cfg)

    raw = pipeline.predict_proba(prepare_inference_frame(features.head(40), cfg))[:, 1]
    served = []
    for record in features.head(40).to_dict(orient="records"):
        # What arrives over HTTP: JSON-native numbers and booleans.
        payload = json.loads(json.dumps(record, default=lambda v: v.item()))
        clean, notes = signature.check_record(payload)
        assert not any("not seen in training" in n for n in notes), notes
        served.append(pipeline.predict_proba(prepare_inference_frame([clean], cfg))[0, 1])
    np.testing.assert_allclose(served, raw, rtol=0, atol=1e-12)
