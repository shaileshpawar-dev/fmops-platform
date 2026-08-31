"""Drift detection.

Four things are commonly lumped together as "drift". They are not the same, and
this module keeps them apart deliberately:

**Data drift / feature drift** -- P(x) changed. Measurable from unlabelled
production traffic by comparing each feature's production distribution against
the training reference. This is what the platform monitors continuously.

**Prediction drift** -- P(y_hat) changed. Also measurable without labels: the
model's output distribution shifts. It is a *symptom*: it can be caused by input
drift, or by a genuine change in the population, or by nothing at all.

**Concept drift** -- P(y | x) changed: the same inputs now imply a different
outcome. **This cannot be measured from unlabelled data.** No amount of
input-distribution monitoring detects it, because the inputs may be completely
unchanged (see the ``concept`` mode in :mod:`app.data.generator`, which leaves
every input distribution identical while reversing a feature's effect).
The platform therefore reports concept drift as ``unavailable`` unless
ground-truth labels have arrived via the feedback endpoint, in which case it is
computed as the decay in live model performance. Prediction drift is offered as
a *proxy signal* and is explicitly labelled as such -- never as a measurement.

Choice of statistic
-------------------
The headline ``score`` for every feature is the **Jensen-Shannon distance**
between the reference and current distributions: bounded in [0, 1], symmetric,
and defined for both numeric (binned) and categorical features, so scores are
comparable across feature types and can be averaged.

The ``drifted`` flag uses **PSI** (an effect size) as the primary criterion,
with the KS / chi-square p-value as supporting evidence. This ordering is
deliberate: with production windows of a few thousand rows, KS p-values are
driven below 0.05 by sample size alone, so a p-value-only rule flags everything
as drifted. PSI measures how *large* the shift is, which is what actually
matters operationally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import DriftConfig, Settings, get_settings
from app.core.exceptions import DependencyMissingError, InsufficientDataError
from app.core.logging import get_logger
from app.core.utils import safe_float, write_json
from app.schemas.evaluation import DriftReport, FeatureDrift

logger = get_logger(__name__)

# PSI convention: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant.
PSI_MODERATE = 0.10
_EPSILON = 1e-6


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two numeric samples using reference quantile bins.

    Quantile bins (rather than equal-width) keep the reference buckets balanced,
    which is what makes PSI comparable across features on different scales.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), _EPSILON, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), _EPSILON, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI over category shares, including categories new in production."""
    ref_share = reference.astype(str).value_counts(normalize=True)
    cur_share = current.astype(str).value_counts(normalize=True)
    categories = sorted(set(ref_share.index) | set(cur_share.index))
    ref = np.clip(np.array([ref_share.get(c, 0.0) for c in categories]), _EPSILON, None)
    cur = np.clip(np.array([cur_share.get(c, 0.0) for c in categories]), _EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """JS distance (sqrt of JS divergence, base 2). Bounded in [0, 1]."""
    p = np.clip(np.asarray(p, dtype=float), _EPSILON, None)
    q = np.clip(np.asarray(q, dtype=float), _EPSILON, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log2(a / b)))

    divergence = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(divergence, 0.0)))


def numeric_js_distance(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """JS distance between two numeric samples, binned on reference quantiles."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0
    edges = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    return jensen_shannon_distance(ref_counts, cur_counts)


def categorical_js_distance(reference: pd.Series, current: pd.Series) -> float:
    ref_share = reference.astype(str).value_counts(normalize=True)
    cur_share = current.astype(str).value_counts(normalize=True)
    categories = sorted(set(ref_share.index) | set(cur_share.index))
    ref = np.array([ref_share.get(c, 0.0) for c in categories])
    cur = np.array([cur_share.get(c, 0.0) for c in categories])
    return jensen_shannon_distance(ref, cur)


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    try:
        from scipy import stats

        result = stats.ks_2samp(reference, current)
        return float(result.statistic), float(result.pvalue)
    except Exception as exc:
        logger.debug("drift.ks_unavailable", extra={"error": str(exc)})
        return 0.0, 1.0


def chi_square_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Chi-square test of independence over category counts."""
    try:
        from scipy import stats

        ref_counts = reference.astype(str).value_counts()
        cur_counts = current.astype(str).value_counts()
        categories = sorted(set(ref_counts.index) | set(cur_counts.index))
        table = np.array(
            [
                [ref_counts.get(c, 0) for c in categories],
                [cur_counts.get(c, 0) for c in categories],
            ],
            dtype=float,
        )
        # Drop all-zero columns; chi2 is undefined on them.
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            return 0.0, 1.0
        statistic, p_value, _, _ = stats.chi2_contingency(table)
        return float(statistic), float(p_value)
    except Exception as exc:
        logger.debug("drift.chi2_unavailable", extra={"error": str(exc)})
        return 0.0, 1.0


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
class DriftEngine(ABC):
    """Compares a production window against a reference window."""

    name: str = "abstract"

    @abstractmethod
    def analyse(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
        config: DriftConfig,
    ) -> list[FeatureDrift]: ...


class NativeDriftEngine(DriftEngine):
    """PSI + JS + KS/chi-square. The platform default; no extra dependencies."""

    name = "native"

    def analyse(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
        config: DriftConfig,
    ) -> list[FeatureDrift]:
        results: list[FeatureDrift] = []

        for feature in numeric_features:
            if feature not in reference or feature not in current:
                continue
            ref = pd.to_numeric(reference[feature], errors="coerce").dropna().to_numpy()
            cur = pd.to_numeric(current[feature], errors="coerce").dropna().to_numpy()
            if ref.size < 10 or cur.size < 10:
                continue

            psi = population_stability_index(ref, cur, config.numeric_bins)
            score = numeric_js_distance(ref, cur, config.numeric_bins)
            statistic, p_value = ks_test(ref, cur)
            # Effect size leads; the p-value only confirms.
            drifted = psi > config.psi_threshold or (
                p_value < config.ks_p_value and psi > PSI_MODERATE
            )
            results.append(
                FeatureDrift(
                    feature=feature,
                    feature_type="numeric",
                    test="psi+ks",
                    statistic=round(psi, 6),
                    p_value=round(p_value, 6),
                    score=round(score, 6),
                    threshold=config.psi_threshold,
                    drifted=bool(drifted),
                    reference_summary={
                        "mean": safe_float(ref.mean()),
                        "std": safe_float(ref.std()),
                        "median": safe_float(np.median(ref)),
                        "count": float(ref.size),
                    },
                    current_summary={
                        "mean": safe_float(cur.mean()),
                        "std": safe_float(cur.std()),
                        "median": safe_float(np.median(cur)),
                        "count": float(cur.size),
                        "ks_statistic": round(statistic, 6),
                    },
                )
            )

        for feature in categorical_features:
            if feature not in reference or feature not in current:
                continue
            ref = reference[feature].dropna()
            cur = current[feature].dropna()
            if len(ref) < 10 or len(cur) < 10:
                continue

            psi = categorical_psi(ref, cur)
            score = categorical_js_distance(ref, cur)
            statistic, p_value = chi_square_test(ref, cur)
            new_categories = sorted(
                set(cur.astype(str).unique()) - set(ref.astype(str).unique())
            )
            drifted = (
                psi > config.psi_threshold
                or (p_value < config.chi2_p_value and psi > PSI_MODERATE)
                or bool(new_categories)
            )
            results.append(
                FeatureDrift(
                    feature=feature,
                    feature_type="categorical",
                    test="psi+chi2",
                    statistic=round(psi, 6),
                    p_value=round(p_value, 6),
                    score=round(score, 6),
                    threshold=config.psi_threshold,
                    drifted=bool(drifted),
                    reference_summary={
                        "n_categories": float(ref.nunique()),
                        "count": float(len(ref)),
                    },
                    current_summary={
                        "n_categories": float(cur.nunique()),
                        "count": float(len(cur)),
                        "chi2_statistic": round(statistic, 6),
                        "new_categories": float(len(new_categories)),
                    },
                )
            )

        return results


class EvidentlyDriftEngine(DriftEngine):
    """Evidently AI adapter.

    Requires the ``[quality]`` extra. Produces the same
    :class:`~app.schemas.evaluation.FeatureDrift` records as the native engine
    so the rest of the platform is unaffected by the choice, and additionally
    enables Evidently's HTML reports for human review.
    """

    name = "evidently"

    def analyse(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
        config: DriftConfig,
    ) -> list[FeatureDrift]:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
        except ImportError as exc:
            raise DependencyMissingError(
                "evidently is not installed; install the [quality] extra or set "
                "FMOPS_DRIFT__ENGINE=native",
                engine="evidently",
            ) from exc

        columns = [
            c
            for c in [*numeric_features, *categorical_features]
            if c in reference.columns and c in current.columns
        ]
        report = Report(metrics=[DataDriftPreset()])
        evaluation = report.run(
            reference_data=reference[columns], current_data=current[columns]
        )
        payload = evaluation.dict()

        results: list[FeatureDrift] = []
        for metric in payload.get("metrics", []):
            metric_id = str(metric.get("metric_id", ""))
            if "ValueDrift" not in metric_id:
                continue
            feature = _extract_column(metric_id)
            if feature is None:
                continue
            value = safe_float(metric.get("value"), 1.0)
            is_numeric = feature in numeric_features
            results.append(
                FeatureDrift(
                    feature=feature,
                    feature_type="numeric" if is_numeric else "categorical",
                    test="evidently",
                    statistic=value,
                    p_value=value,
                    # Evidently reports a p-value/distance depending on the test;
                    # we recompute the comparable JS score ourselves so scores
                    # stay on the same scale as the native engine.
                    score=(
                        numeric_js_distance(
                            pd.to_numeric(reference[feature], errors="coerce")
                            .dropna()
                            .to_numpy(),
                            pd.to_numeric(current[feature], errors="coerce")
                            .dropna()
                            .to_numpy(),
                            config.numeric_bins,
                        )
                        if is_numeric
                        else categorical_js_distance(
                            reference[feature].dropna(), current[feature].dropna()
                        )
                    ),
                    threshold=config.ks_p_value,
                    drifted=value < config.ks_p_value,
                    reference_summary={"count": float(len(reference))},
                    current_summary={"count": float(len(current))},
                )
            )
        return results

    def html_report(
        self, reference: pd.DataFrame, current: pd.DataFrame, path: str
    ) -> str | None:
        """Write Evidently's HTML report. Returns the path, or None on failure."""
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            evaluation = report.run(reference_data=reference, current_data=current)
            evaluation.save_html(path)
            return path
        except Exception as exc:
            logger.warning("drift.html_report_failed", extra={"error": str(exc)})
            return None


def _extract_column(metric_id: str) -> str | None:
    if "column=" in metric_id:
        return metric_id.split("column=")[1].split(",")[0].strip(") '\"")
    if "(" in metric_id and ")" in metric_id:
        inner = metric_id[metric_id.index("(") + 1 : metric_id.rindex(")")]
        return inner.split(",")[0].strip(" '\"") or None
    return None


def build_engine(settings: Settings | None = None) -> DriftEngine:
    settings = settings or get_settings()
    if settings.drift.engine == "evidently":
        return EvidentlyDriftEngine()
    return NativeDriftEngine()


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class DriftDetector:
    """Runs a drift scan and turns it into a persisted, alertable report."""

    def __init__(
        self, settings: Settings | None = None, engine: DriftEngine | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or build_engine(self.settings)

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        model_name: str,
        model_version: int | None = None,
        prediction_column: str | None = None,
        reference_prediction_column: str | None = None,
        labelled: pd.DataFrame | None = None,
        baseline_metric: float | None = None,
        persist: bool = True,
    ) -> DriftReport:
        config = self.settings.drift
        if len(current) < config.min_samples:
            raise InsufficientDataError(
                f"drift scan needs at least {config.min_samples} production rows, "
                f"got {len(current)}; either wait for more traffic or lower "
                "FMOPS_DRIFT__MIN_SAMPLES",
                required=config.min_samples,
                available=len(current),
            )
        if reference.empty:
            raise InsufficientDataError(
                "no reference window available; train a model first so a "
                "training-distribution reference profile exists",
                model=model_name,
            )

        data = self.settings.data
        feature_drift = self.engine.analyse(
            reference,
            current,
            data.numeric_features,
            data.categorical_features,
            config,
        )

        drifted = [f.feature for f in feature_drift if f.drifted]
        dataset_score = (
            float(np.mean([f.score for f in feature_drift])) if feature_drift else 0.0
        )
        share = len(drifted) / len(feature_drift) if feature_drift else 0.0

        # A dataset is "drifted" if the average shift is large OR enough
        # individual features moved. Either alone misses real cases: one
        # catastrophically shifted feature, or many mildly shifted ones.
        detected = dataset_score > config.threshold or share >= config.dataset_drift_share

        prediction_score, prediction_drifted = self._prediction_drift(
            reference, current, prediction_column, reference_prediction_column, config
        )
        concept_score, concept_status, concept_detail = self._concept_drift(
            labelled, baseline_metric
        )
        if concept_status == "measured" and concept_score is not None:
            detected = detected or concept_score > config.threshold

        report = DriftReport(
            model_name=model_name,
            model_version=model_version,
            engine=self.engine.name,
            drift_detected=bool(detected),
            dataset_drift_score=round(dataset_score, 6),
            dataset_drift_share=round(share, 6),
            threshold=config.threshold,
            prediction_drift_score=prediction_score,
            prediction_drift_detected=prediction_drifted,
            concept_drift_score=concept_score,
            concept_drift_status=concept_status,
            concept_drift_detail=concept_detail,
            feature_drift=feature_drift,
            drifted_features=drifted,
            n_reference=len(reference),
            n_current=len(current),
        )

        self._export_metrics(report)
        if persist:
            self._persist(report)
            self._maybe_alert(report)

        logger.info(
            "drift.scan_completed",
            extra={
                "model": model_name,
                "engine": self.engine.name,
                "dataset_drift_score": report.dataset_drift_score,
                "drifted_features": len(drifted),
                "total_features": len(feature_drift),
                "prediction_drift": prediction_score,
                "concept_drift_status": concept_status,
                "detected": report.drift_detected,
            },
        )
        return report

    # -- components ---------------------------------------------------------- #
    def _prediction_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        prediction_column: str | None,
        reference_prediction_column: str | None,
        config: DriftConfig,
    ) -> tuple[float | None, bool]:
        cur_col = prediction_column or "_probability"
        ref_col = reference_prediction_column or prediction_column or "_probability"
        if cur_col not in current or ref_col not in reference:
            return None, False
        ref = pd.to_numeric(reference[ref_col], errors="coerce").dropna().to_numpy()
        cur = pd.to_numeric(current[cur_col], errors="coerce").dropna().to_numpy()
        if ref.size < 10 or cur.size < 10:
            return None, False
        score = numeric_js_distance(ref, cur, config.numeric_bins)
        psi = population_stability_index(ref, cur, config.numeric_bins)
        return round(score, 6), bool(psi > config.psi_threshold)

    def _concept_drift(
        self, labelled: pd.DataFrame | None, baseline_metric: float | None
    ) -> tuple[float | None, str, str]:
        """Measure concept drift, or state honestly that it cannot be measured.

        With labels, concept drift is quantified as the *relative degradation*
        in live ROC-AUC versus the model's offline baseline. Without labels it
        is simply not computable, and we say so.
        """
        unavailable = (
            "Concept drift (a change in P(y|x)) cannot be computed from "
            "unlabelled production data. No ground-truth labels were available "
            "for this window, so only data, feature and prediction drift were "
            "measured. Prediction drift is a proxy signal, not a measurement of "
            "concept drift. Submit labels via POST /api/v1/feedback to enable it."
        )
        if labelled is None or labelled.empty or baseline_metric is None:
            return None, "unavailable", unavailable

        target = "actual_label"
        if target not in labelled or "probability" not in labelled:
            return None, "unavailable", unavailable

        y_true = pd.to_numeric(labelled[target], errors="coerce").dropna()
        probabilities = pd.to_numeric(labelled["probability"], errors="coerce")
        aligned = labelled.assign(_y=y_true, _p=probabilities).dropna(subset=["_y", "_p"])
        if len(aligned) < 50 or aligned["_y"].nunique() < 2:
            return (
                None,
                "insufficient_labels",
                f"Only {len(aligned)} labelled rows with both classes were available; "
                "at least 50 are needed before live performance is meaningful.",
            )

        try:
            from sklearn.metrics import roc_auc_score

            live_auc = float(roc_auc_score(aligned["_y"], aligned["_p"]))
        except Exception as exc:
            return None, "unavailable", f"live ROC-AUC could not be computed: {exc}"

        # Relative degradation, clipped to [0, 1] so it is comparable to the
        # other drift scores.
        degradation = max(0.0, (baseline_metric - live_auc) / max(baseline_metric, _EPSILON))
        return (
            round(min(degradation, 1.0), 6),
            "measured",
            (
                f"Measured from {len(aligned)} labelled production rows: live "
                f"ROC-AUC {live_auc:.4f} vs offline baseline {baseline_metric:.4f} "
                f"({degradation:.1%} relative degradation)."
            ),
        )

    # -- side effects --------------------------------------------------------- #
    def _export_metrics(self, report: DriftReport) -> None:
        from app.monitoring.metrics import set_drift_metrics

        set_drift_metrics(
            report.model_name,
            report.dataset_drift_score,
            report.prediction_drift_score,
            report.concept_drift_score,
            report.drift_detected,
            {f.feature: f.score for f in report.feature_drift},
        )

    def _persist(self, report: DriftReport) -> None:
        from app.core.db import dumps, get_database

        path = self.settings.paths.reports_dir / "drift" / f"{report.id}.json"
        try:
            write_json(path, report.model_dump(mode="json"))
            report.report_uri = path.resolve().as_uri()
        except Exception as exc:
            logger.warning("drift.report_write_failed", extra={"error": str(exc)})

        try:
            get_database().execute(
                "INSERT INTO drift_reports (id, model_name, model_version, engine, "
                "drift_detected, dataset_drift_score, prediction_drift_score, "
                "concept_drift_score, concept_drift_status, drifted_features, "
                "n_reference, n_current, report, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    report.id,
                    report.model_name,
                    report.model_version,
                    report.engine,
                    1 if report.drift_detected else 0,
                    report.dataset_drift_score,
                    report.prediction_drift_score,
                    report.concept_drift_score,
                    report.concept_drift_status,
                    dumps(report.drifted_features),
                    report.n_reference,
                    report.n_current,
                    dumps(report.model_dump(mode="json")),
                    report.created_at,
                ),
            )
        except Exception as exc:
            logger.error("drift.persist_failed", extra={"error": str(exc)})

    def _maybe_alert(self, report: DriftReport) -> None:
        if not report.drift_detected:
            return
        from app.monitoring.alerts import get_alert_manager
        from app.schemas.common import AlertCategory, Severity

        severity = (
            Severity.CRITICAL
            if report.dataset_drift_score > report.threshold * 2
            else Severity.WARNING
        )
        # Say which rule actually fired. A drift report that claims the score
        # breached a threshold it did not breach destroys trust in the alert.
        share_threshold = self.settings.drift.dataset_drift_share
        reasons: list[str] = []
        if report.dataset_drift_score > report.threshold:
            reasons.append(
                f"mean drift score {report.dataset_drift_score:.4f} exceeds the "
                f"threshold {report.threshold:.2f}"
            )
        if report.dataset_drift_share >= share_threshold:
            reasons.append(
                f"{report.dataset_drift_share:.0%} of features drifted "
                f"(limit {share_threshold:.0%})"
            )
        if (
            report.concept_drift_status == "measured"
            and (report.concept_drift_score or 0.0) > report.threshold
        ):
            reasons.append(
                f"measured concept drift {report.concept_drift_score:.4f} exceeds "
                f"the threshold {report.threshold:.2f}"
            )

        get_alert_manager().raise_alert(
            severity,
            AlertCategory.DRIFT,
            f"Data drift detected on {report.model_name}",
            (
                f"Drift detected because {' and '.join(reasons) or 'the drift rule fired'}. "
                f"{len(report.drifted_features)} of {len(report.feature_drift)} "
                f"features drifted: {', '.join(report.drifted_features[:8])}. "
                f"Mean drift score {report.dataset_drift_score:.4f}."
            ),
            context={
                "model_name": report.model_name,
                "model_version": report.model_version,
                "drift_report_id": report.id,
                "dataset_drift_score": report.dataset_drift_score,
                "drifted_features": report.drifted_features,
                "concept_drift_status": report.concept_drift_status,
            },
            dedupe_keys=("model_name", "model_version"),
        )


# --------------------------------------------------------------------------- #
# Report store
# --------------------------------------------------------------------------- #
def recent_drift_reports(
    model_name: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    from app.core.db import get_database, loads

    sql = "SELECT * FROM drift_reports"
    params: list[Any] = []
    if model_name:
        sql += " WHERE model_name = ?"
        params.append(model_name)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    out = []
    for row in get_database().query(sql, params):
        item = dict(row)
        item["drifted_features"] = loads(item.get("drifted_features"), [])
        item["report"] = loads(item.get("report"), {})
        item["drift_detected"] = bool(item["drift_detected"])
        out.append(item)
    return out


def latest_drift_report(model_name: str | None = None) -> DriftReport | None:
    reports = recent_drift_reports(model_name, limit=1)
    if not reports:
        return None
    payload = reports[0].get("report")
    if not payload:
        return None
    try:
        return DriftReport(**payload)
    except Exception:
        return None
