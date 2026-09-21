"""Inference log and ground-truth feedback.

Every served prediction is recorded with its features, output, latency, serving
version and variant. That log is the substrate for three things the platform
could not otherwise do:

* **Drift detection** -- the production window compared against the training
  reference comes from here.
* **Live performance** -- joined against :class:`feedback` rows, this is the only
  honest source of production accuracy. Without labels, no accuracy number is
  reported at all.
* **Canary evidence** -- per-version error rate and latency over a recent window.

Sampling: ``monitoring.prediction_log_sample_rate`` controls what fraction of
requests are persisted. Sampling reduces write load but weakens drift
statistics, so the effective sample rate is recorded alongside every drift
report rather than being invisible.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.db import Database, dumps, get_database, loads
from app.core.exceptions import PredictionError, PredictionNotFoundError
from app.core.logging import get_logger
from app.core.utils import iso_minutes_ago, percentile, utcnow_iso

logger = get_logger(__name__)


class InferenceLog:
    """Repository over the ``inference_log`` and ``feedback`` tables."""

    def __init__(self, db: Database | None = None, settings: Settings | None = None) -> None:
        self._db = db
        self.settings = settings or get_settings()
        self._rng = random.Random()

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    # -- writes -------------------------------------------------------------- #
    def record(
        self,
        request_id: str,
        model_name: str,
        model_version: int | None,
        features: dict[str, Any],
        prediction: int | None,
        probability: float | None,
        latency_ms: float,
        deployment_id: str | None = None,
        variant: str = "primary",
        shadow: bool = False,
        status: str = "ok",
        error_code: str | None = None,
        force: bool = False,
    ) -> bool:
        """Persist one prediction. Returns whether it was actually written."""
        monitoring = self.settings.monitoring
        if not monitoring.log_predictions and not force:
            return False
        # Errors are always logged: they are rare and they are what you need.
        sampled = (
            force
            or status != "ok"
            or monitoring.prediction_log_sample_rate >= 1.0
            or self._rng.random() < monitoring.prediction_log_sample_rate
        )
        if not sampled:
            return False

        try:
            self.db.execute(
                "INSERT INTO inference_log (request_id, model_name, model_version, "
                "deployment_id, variant, shadow, features, prediction, probability, "
                "latency_ms, status, error_code, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    model_name,
                    model_version,
                    deployment_id,
                    variant,
                    1 if shadow else 0,
                    dumps(features),
                    prediction,
                    probability,
                    float(latency_ms),
                    status,
                    error_code,
                    utcnow_iso(),
                ),
            )
            return True
        except Exception as exc:
            # Never fail a prediction because logging failed.
            logger.error(
                "inference_log.write_failed",
                extra={"request_id": request_id, "error": str(exc)},
            )
            return False

    def served(self, request_id: str) -> dict[str, Any] | None:
        """The primary (non-shadow) prediction recorded for a request id."""
        row = self.db.query_one(
            "SELECT model_name, model_version, prediction, probability, created_at "
            "FROM inference_log WHERE request_id = ? AND shadow = 0 AND status = 'ok' "
            "ORDER BY id DESC LIMIT 1",
            (request_id,),
        )
        return dict(row) if row else None

    def record_feedback(
        self, request_id: str, actual_label: int | str, source: str = "manual"
    ) -> dict[str, Any]:
        """Attach ground truth to a prediction this platform actually served.

        * Feedback for an unknown request is refused: a label with nothing to
          join to would silently inflate nothing and hide a client bug.
        * A named label ("yes", "churned") is resolved through the signature of
          the model version that served the request.
        * A second label for the same request replaces the first. Corrections
          happen; counting a request twice would skew live quality and weight
          that row twice in retraining.
        """
        served = self.served(request_id)
        if served is None:
            raise PredictionNotFoundError(
                f"no served prediction has request id {request_id!r}",
                request_id=request_id,
            )
        encoded = self._encode_label(
            served["model_name"], served["model_version"], actual_label
        )
        with self.db.transaction() as conn:
            replaced = conn.execute(
                "DELETE FROM feedback WHERE request_id = ?", (request_id,)
            ).rowcount
            conn.execute(
                "INSERT INTO feedback (request_id, actual_label, source, created_at) "
                "VALUES (?,?,?,?)",
                (request_id, encoded, source, utcnow_iso()),
            )
        total = self.labelled_count(served["model_name"])
        logger.info(
            "feedback.recorded",
            extra={
                "request_id": request_id,
                "model": served["model_name"],
                "model_version": served["model_version"],
                "label": encoded,
                "replaced": bool(replaced and replaced > 0),
                "labelled_total": total,
            },
        )
        return {
            "request_id": request_id,
            "model_name": served["model_name"],
            "model_version": served["model_version"],
            "actual_label": encoded,
            "replaced_previous": bool(replaced and replaced > 0),
            "labelled_total": total,
        }

    @staticmethod
    def _encode_label(model_name: str, version: int | None, label: int | str) -> int:
        if isinstance(label, int):
            return int(label)
        text = label.strip()
        if text in ("0", "1"):
            return int(text)
        from app.registry.context import signature_of
        from app.registry.factory import get_registry

        signature = None
        if version is not None:
            try:
                signature = signature_of(get_registry().get(model_name, int(version)))
            except Exception:
                signature = None
        if signature is not None:
            for names in (signature.class_labels, signature.display_labels or []):
                if names and text in names:
                    return 1 if text == names[1] else 0
            raise PredictionError(
                f"label {text!r} is not a class of {model_name}; expected one of "
                f"{signature.labels} (or 0/1)",
                model=model_name,
            )
        raise PredictionError(
            f"{model_name} v{version} has no recorded classes; send 0 or 1",
            model=model_name,
        )

    # -- reads --------------------------------------------------------------- #
    def recent(
        self,
        model_name: str | None = None,
        model_version: int | None = None,
        minutes: int | None = None,
        limit: int = 5000,
        include_shadow: bool = False,
        only_ok: bool = True,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if model_version is not None:
            clauses.append("model_version = ?")
            params.append(int(model_version))
        if minutes is not None:
            clauses.append("created_at >= ?")
            params.append(iso_minutes_ago(minutes))
        if not include_shadow:
            clauses.append("shadow = 0")
        if only_ok:
            clauses.append("status = 'ok'")
        sql = "SELECT * FROM inference_log"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = self.db.query(sql, params)
        out = []
        for row in rows:
            item = dict(row)
            item["features"] = loads(item.get("features"), {})
            out.append(item)
        return out

    def feature_frame(
        self,
        model_name: str | None = None,
        model_version: int | None = None,
        limit: int = 5000,
        minutes: int | None = None,
    ) -> pd.DataFrame:
        """Recent production features as a frame, for drift detection."""
        rows = self.recent(
            model_name, model_version, minutes=minutes, limit=limit, only_ok=True
        )
        if not rows:
            return pd.DataFrame()
        records = []
        for row in rows:
            record = dict(row["features"])
            record["_prediction"] = row["prediction"]
            record["_probability"] = row["probability"]
            record["_request_id"] = row["request_id"]
            record["_model_version"] = row["model_version"]
            records.append(record)
        return pd.DataFrame(records)

    def labelled_frame(
        self,
        model_name: str | None = None,
        model_version: int | None = None,
        limit: int = 5000,
        with_features: bool = True,
    ) -> pd.DataFrame:
        """Predictions joined to ground truth. Empty when no labels exist.

        ``with_features=False`` omits the stored feature payload. Drift
        detection and retraining need those columns; scoring live quality only
        reads prediction, probability and actual_label, and deserialising a
        JSON blob per row to discard it costs 7.3ms per call on a 452-row
        window -- around a third of the dashboard request. The rows selected
        and their order are identical either way.
        """
        columns = (
            "i.request_id, i.model_name, i.model_version, i.features, "
            "i.prediction, i.probability, i.created_at, f.actual_label"
            if with_features
            else "i.request_id, i.model_name, i.model_version, "
            "i.prediction, i.probability, i.created_at, f.actual_label"
        )
        sql = (
            f"SELECT {columns} "
            "FROM inference_log i JOIN feedback f ON f.request_id = i.request_id "
            "WHERE i.shadow = 0 AND i.status = 'ok'"
        )
        params: list[Any] = []
        if model_name:
            sql += " AND i.model_name = ?"
            params.append(model_name)
        if model_version is not None:
            sql += " AND i.model_version = ?"
            params.append(int(model_version))
        sql += " ORDER BY i.id DESC LIMIT ?"
        params.append(limit)

        rows = self.db.query(sql, params)
        if not rows:
            return pd.DataFrame()
        if not with_features:
            return pd.DataFrame([dict(row) for row in rows])
        records = []
        for row in rows:
            record = dict(row)
            features = loads(record.pop("features"), {})
            records.append({**features, **record})
        return pd.DataFrame(records)

    def window_stats(
        self,
        model_name: str | None = None,
        model_version: int | None = None,
        minutes: int = 60,
        variant: str | None = None,
    ) -> dict[str, Any]:
        """Request count, error rate and latency percentiles over a window."""
        clauses = ["created_at >= ?", "shadow = 0"]
        params: list[Any] = [iso_minutes_ago(minutes)]
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if model_version is not None:
            clauses.append("model_version = ?")
            params.append(int(model_version))
        if variant:
            clauses.append("variant = ?")
            params.append(variant)
        where = " WHERE " + " AND ".join(clauses)

        total = int(self.db.scalar(f"SELECT COUNT(*) FROM inference_log{where}", params, 0))
        errors = int(
            self.db.scalar(
                f"SELECT COUNT(*) FROM inference_log{where} AND status != 'ok'",
                params,
                0,
            )
        )
        latencies = [
            float(row[0])
            for row in self.db.query(
                f"SELECT latency_ms FROM inference_log{where} AND status = 'ok'", params
            )
        ]
        positives = int(
            self.db.scalar(
                f"SELECT COUNT(*) FROM inference_log{where} AND prediction = 1", params, 0
            )
        )
        scored = total - errors

        return {
            "window_minutes": minutes,
            "request_count": total,
            "error_count": errors,
            "error_rate": (errors / total) if total else 0.0,
            "throughput_rpm": (total / minutes) if minutes else 0.0,
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "latency_p99_ms": percentile(latencies, 99),
            "latency_max_ms": max(latencies) if latencies else 0.0,
            "latency_mean_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "positive_rate": (positives / scored) if scored else None,
        }

    def count(self, model_name: str | None = None, since_minutes: int | None = None) -> int:
        clauses: list[str] = ["shadow = 0"]
        params: list[Any] = []
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if since_minutes is not None:
            clauses.append("created_at >= ?")
            params.append(iso_minutes_ago(since_minutes))
        where = " WHERE " + " AND ".join(clauses)
        return int(self.db.scalar(f"SELECT COUNT(*) FROM inference_log{where}", params, 0))

    def labelled_count(
        self,
        model_name: str | None = None,
        since: str | None = None,
        feedback_since: str | None = None,
    ) -> int:
        """Labelled primary predictions, optionally only those served after ``since``.

        ``since`` is how retraining asks "is there anything the serving version
        has not already learned from": traffic served after a version was
        trained cannot have been in its training data.
        """
        sql = (
            "SELECT COUNT(*) FROM inference_log i "
            "JOIN feedback f ON f.request_id = i.request_id "
            "WHERE i.shadow = 0 AND i.status = 'ok'"
        )
        params: list[Any] = []
        if model_name:
            sql += " AND i.model_name = ?"
            params.append(model_name)
        if since:
            sql += " AND i.created_at >= ?"
            params.append(since)
        if feedback_since:
            # Labels that arrived after a point -- e.g. after the last
            # retraining attempt, which already learned from everything before.
            sql += " AND f.created_at > ?"
            params.append(feedback_since)
        return int(self.db.scalar(sql, params, 0))

    def purge_older_than(self, days: int) -> int:
        from app.core.utils import iso_days_ago

        cursor = self.db.execute(
            "DELETE FROM inference_log WHERE created_at < ?", (iso_days_ago(days),)
        )
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0


_LOG: InferenceLog | None = None


def get_inference_log() -> InferenceLog:
    global _LOG
    if _LOG is None:
        _LOG = InferenceLog()
    return _LOG


def set_inference_log(log: InferenceLog | None) -> None:
    global _LOG
    _LOG = log
