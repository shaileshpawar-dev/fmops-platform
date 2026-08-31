"""Send synthetic production traffic through the serving path.

This exists so the demo is honest: drift detection, canary evaluation and live
performance all read the *real* inference log, so they need real requests to
have happened. This script generates applications from the same generator the
training data came from -- optionally with drift injected -- and scores them
through :class:`~app.api.serving.PredictionService`, exactly as an HTTP request
would.

``--label-fraction`` submits ground-truth feedback for a share of the requests.
That is what unlocks live-performance metrics and measured concept drift; with
no labels, the platform correctly reports both as unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/demo.py` as well as `python -m scripts.demo`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from typing import Any

from app.api.serving import get_prediction_service
from app.core.config import get_settings
from app.core.logging import configure_from_settings, get_logger, new_request_id
from app.core.utils import percentile
from app.data.generator import DriftMode, build_production_dataset
from app.monitoring.inference_log import get_inference_log

logger = get_logger("fmops.simulate")


def simulate(
    rows: int = 500,
    drift: DriftMode = "none",
    drift_strength: float = 1.0,
    label_fraction: float = 0.0,
    seed: int | None = None,
    model_version: int | None = None,
) -> dict[str, Any]:
    """Score ``rows`` synthetic applications and return summary statistics."""
    settings = get_settings()
    service = get_prediction_service()
    inference_log = get_inference_log()

    frame = build_production_dataset(
        n_rows=rows,
        drift=drift,
        drift_strength=drift_strength,
        seed=seed if seed is not None else 202,
    )
    feature_columns = settings.data.feature_columns
    target_column = settings.data.target_column

    latencies: list[float] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    errors = 0
    labelled = 0
    versions: dict[int, int] = {}

    for index, row in frame.iterrows():
        features = {c: row[c] for c in feature_columns}
        request_id = new_request_id()
        try:
            response = service.predict_one(
                features=_coerce(features),
                request_id=request_id,
                version=model_version,
            )
        except Exception as exc:
            errors += 1
            logger.warning(
                "simulate.request_failed", extra={"index": int(index), "error": str(exc)}
            )
            continue

        latencies.append(response.inference_latency_ms)
        predictions.append(response.prediction)
        probabilities.append(response.probability)
        versions[response.model_version] = versions.get(response.model_version, 0) + 1

        # Ground truth arrives for a share of requests, mimicking a real
        # feedback loop where outcomes are observed late and incompletely.
        if (
            label_fraction > 0
            and (index % max(1, int(1 / label_fraction))) == 0
            and target_column in row
        ):
            inference_log.record_feedback(
                request_id, int(row[target_column]), source="simulation"
            )
            labelled += 1

    total = len(latencies)
    stats: dict[str, Any] = {
        "requests_sent": len(frame),
        "requests_scored": total,
        "errors": errors,
        "error_rate": round(errors / len(frame), 5) if len(frame) else 0.0,
        "labels_submitted": labelled,
        "drift_mode": drift,
        "drift_strength": drift_strength,
        "positive_rate": round(sum(predictions) / total, 4) if total else 0.0,
        "mean_probability": round(sum(probabilities) / total, 4) if total else 0.0,
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "latency_p99_ms": round(percentile(latencies, 99), 2),
        "served_by_version": {f"v{k}": v for k, v in sorted(versions.items())},
    }
    if target_column in frame.columns:
        stats["actual_positive_rate"] = round(float(frame[target_column].mean()), 4)

    logger.info("simulate.completed", extra=stats)
    return stats


def _coerce(features: dict[str, Any]) -> dict[str, Any]:
    """numpy scalars are not JSON/pydantic friendly."""
    out: dict[str, Any] = {}
    for key, value in features.items():
        item = getattr(value, "item", None)
        out[key] = item() if callable(item) else value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send synthetic traffic to the model")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument(
        "--drift",
        default="none",
        choices=["none", "covariate", "categorical", "concept", "severe"],
    )
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--label-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-version", type=int, default=None)
    args = parser.parse_args(argv)

    configure_from_settings(force=True)
    stats = simulate(
        rows=args.rows,
        drift=args.drift,
        drift_strength=args.strength,
        label_fraction=args.label_fraction,
        seed=args.seed,
        model_version=args.model_version,
    )
    for key, value in stats.items():
        print(f"  {key:22s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
