"""Deterministic synthetic dataset for the reference use case.

The platform is the point of this project, not the model -- but a *realistic*
dataset matters because it decides whether drift detection, approval gates and
retraining comparisons behave the way they would in production.

This generator produces a loan-default table with:

* correlated features (credit score depends on late payments and utilisation,
  income depends on age and tenure, loan size depends on income),
* a target driven by a logistic function of those features plus noise, giving a
  learnable-but-not-trivial signal (ROC-AUC lands around 0.85),
* reproducible output for a given seed, so every run of the pipeline is
  comparable,
* explicit *corruption* and *drift* modes used by the failure-path tests and the
  drift demo.

Nothing here reads the network. ``make data`` regenerates every artifact.
"""

from __future__ import annotations
from collections.abc import Callable

from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from app.core.config import DataConfig, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DriftMode = Literal["none", "covariate", "categorical", "concept", "severe"]

EMPLOYMENT_TYPES = ("salaried", "self_employed", "contract", "retired", "unemployed")
HOUSING_STATUS = ("own", "mortgage", "rent", "other")
LOAN_PURPOSES = ("debt_consolidation", "home_improvement", "auto", "medical", "business")
REGIONS = ("north", "south", "east", "west", "central")

# Base category mixes. The drift modes perturb these.
_EMPLOYMENT_P = np.array([0.55, 0.18, 0.14, 0.08, 0.05])
_HOUSING_P = np.array([0.28, 0.34, 0.33, 0.05])
_PURPOSE_P = np.array([0.34, 0.21, 0.20, 0.13, 0.12])
_REGION_P = np.array([0.24, 0.26, 0.20, 0.18, 0.12])

# Risk contributions per category level, in log-odds.
_EMPLOYMENT_RISK = {
    "salaried": -0.35,
    "self_employed": 0.15,
    "contract": 0.25,
    "retired": -0.10,
    "unemployed": 0.95,
}
_HOUSING_RISK = {"own": -0.30, "mortgage": -0.05, "rent": 0.30, "other": 0.25}
_PURPOSE_RISK = {
    "debt_consolidation": 0.30,
    "home_improvement": -0.15,
    "auto": -0.05,
    "medical": 0.20,
    "business": 0.35,
}
_REGION_RISK = {"north": 0.0, "south": 0.08, "east": -0.05, "west": -0.08, "central": 0.05}


# Controls separability of the two classes. 2.35 puts held-out ROC-AUC around
# 0.85 -- learnable, but far from perfect, like a real credit-risk problem.
_SIGNAL_STRENGTH = 2.35
# Baseline default rate. Realistic for an unsecured consumer loan book.
_BASE_POSITIVE_RATE = 0.18
# Fixed sample used to derive standardisation constants (see _reference_moments).
_MOMENT_SEED = 12_345
_MOMENT_ROWS = 20_000


@dataclass
class GenerationSpec:
    """Knobs for one dataset generation call."""

    n_rows: int = 8000
    seed: int = 42
    drift: DriftMode = "none"
    drift_strength: float = 1.0
    start_date: datetime = field(
        default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)
    )
    days_span: int = 365
    # Corruption knobs, used to exercise the validation failure path.
    missing_fraction: float = 0.0
    duplicate_fraction: float = 0.0
    outlier_fraction: float = 0.0
    invalid_value_fraction: float = 0.0
    class_imbalance: float | None = None


def _sample_categorical(
    rng: np.random.Generator, levels: tuple[str, ...], probs: np.ndarray, n: int
) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    return rng.choice(np.array(levels), size=n, p=p)


def _shift_probs(probs: np.ndarray, toward: int, strength: float) -> np.ndarray:
    """Move probability mass toward one level; used for categorical drift."""
    shifted = probs.copy().astype(float)
    move = shifted * min(0.85, 0.5 * strength)
    shifted -= move
    shifted[toward] += move.sum()
    return shifted / shifted.sum()


def _concept_weights(concept_strength: float) -> tuple[float, ...]:
    """Feature weights in log-odds space.

    ``concept_strength`` of 0 gives the baseline relationship. Raising it
    weakens the credit-score signal, strengthens debt-to-income and late
    payments, and reverses the sign on utilisation -- a genuine change in
    P(y | x) with P(x) untouched.
    """
    s = float(concept_strength)
    return (
        -0.0135 * (1.0 - 1.30 * s),  # credit_score
        1.85 * (1.0 + 1.40 * s),  # debt_to_income
        1.35 * (1.0 - 2.20 * s),  # credit_utilization (flips sign at s=0.45)
        0.42 * (1.0 + 1.60 * s),  # num_late_payments_12m
        -0.021,  # employment_years
        -0.0000032,  # annual_income
        0.0000019,  # loan_amount
        -0.0055,  # age
    )


def _risk_score(
    weights: tuple[float, ...],
    *,
    credit_score: np.ndarray,
    debt_to_income: np.ndarray,
    credit_utilization: np.ndarray,
    num_late_payments_12m: np.ndarray,
    employment_years: np.ndarray,
    annual_income: np.ndarray,
    loan_amount: np.ndarray,
    age: np.ndarray,
    employment_type: np.ndarray,
    housing_status: np.ndarray,
    loan_purpose: np.ndarray,
    region: np.ndarray,
) -> np.ndarray:
    """Deterministic (noise-free) log-odds contribution of every feature."""
    w_credit, w_dti, w_util, w_late, w_emp, w_inc, w_loan, w_age = weights
    score = (
        w_credit * (credit_score - 640.0)
        + w_dti * (debt_to_income - 0.32)
        + w_util * (credit_utilization - 0.34)
        + w_late * num_late_payments_12m
        + w_emp * employment_years
        + w_inc * annual_income
        + w_loan * loan_amount
        + w_age * (age - 42.0)
    )
    score = score + np.array([_EMPLOYMENT_RISK[v] for v in employment_type])
    score = score + np.array([_HOUSING_RISK[v] for v in housing_status])
    score = score + np.array([_PURPOSE_RISK[v] for v in loan_purpose])
    score = score + np.array([_REGION_RISK[v] for v in region])
    return score


@lru_cache(maxsize=16)
def _reference_score_sample(weights: tuple[float, ...]) -> np.ndarray:
    """Risk scores over a fixed, drift-free reference population.

    Cached per weight vector. Everything downstream (standardisation, intercept
    solving) is derived from this, which is what keeps the positive rate and
    the class separability stable across seeds while still letting covariate
    drift move the observed rate.
    """
    frame = _base_features(_MOMENT_ROWS, _MOMENT_SEED)
    return _risk_score(weights, **frame)


@lru_cache(maxsize=16)
def _reference_moments(weights: tuple[float, ...]) -> tuple[float, float]:
    sample = _reference_score_sample(weights)
    return float(sample.mean()), float(sample.std() or 1.0)


@lru_cache(maxsize=64)
def _solve_intercept(weights: tuple[float, ...], target_rate: float) -> float:
    """Bisect for the intercept that yields ``target_rate`` on the reference set."""
    sample = _reference_score_sample(weights)
    mean, std = _reference_moments(weights)
    z = (sample - mean) / std
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        rate = float((1.0 / (1.0 + np.exp(-(_SIGNAL_STRENGTH * z + mid)))).mean())
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _base_features(n: int, seed: int) -> dict[str, np.ndarray]:
    """Drift-free feature draw, used only to calibrate the target."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(42.0, 12.0, n), 18, 95)
    employment_years = np.clip(rng.gamma(shape=2.0, scale=3.2, size=n) * (age / 42.0), 0, 45)
    annual_income = np.clip(
        (32_000 + 1_450 * employment_years + 380 * age) * rng.lognormal(0.0, 0.42, n),
        12_000,
        900_000,
    )
    num_late_payments_12m = np.clip(rng.poisson(0.55, n), 0, 30)
    credit_utilization = np.clip(rng.beta(2.0, 4.2, n), 0.0, 1.95)
    credit_score = np.clip(
        720.0
        - 26.0 * num_late_payments_12m
        - 95.0 * credit_utilization
        + 1.15 * employment_years
        + 0.00028 * annual_income
        + rng.normal(0, 33, n),
        300,
        850,
    )
    loan_amount = np.clip(
        annual_income * rng.uniform(0.08, 0.62, n) + rng.normal(0, 3_500, n),
        1_000,
        1_500_000,
    )
    loan_term_months = rng.choice(
        np.array([12, 24, 36, 48, 60, 84, 120]),
        size=n,
        p=np.array([0.07, 0.14, 0.31, 0.19, 0.16, 0.08, 0.05]),
    )
    debt_to_income = np.clip(
        (loan_amount / np.maximum(loan_term_months, 1) * 12.0) / np.maximum(annual_income, 1.0)
        + rng.normal(0.06, 0.05, n),
        0.0,
        2.9,
    )
    return {
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "credit_utilization": credit_utilization,
        "num_late_payments_12m": num_late_payments_12m,
        "employment_years": employment_years,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "age": age,
        "employment_type": _sample_categorical(rng, EMPLOYMENT_TYPES, _EMPLOYMENT_P, n),
        "housing_status": _sample_categorical(rng, HOUSING_STATUS, _HOUSING_P, n),
        "loan_purpose": _sample_categorical(rng, LOAN_PURPOSES, _PURPOSE_P, n),
        "region": _sample_categorical(rng, REGIONS, _REGION_P, n),
    }


def generate_dataset(spec: GenerationSpec, config: DataConfig | None = None) -> pd.DataFrame:
    """Generate a loan-application table according to ``spec``."""
    config = config or get_settings().data
    rng = np.random.default_rng(spec.seed)
    n = spec.n_rows
    s = spec.drift_strength

    # ---------------- covariate drift shifts the numeric distributions ----- #
    covariate = spec.drift in ("covariate", "severe")
    categorical_drift = spec.drift in ("categorical", "severe")
    concept = spec.drift in ("concept", "severe")

    age_mu = 42.0 + (11.0 * s if covariate else 0.0)
    age_sigma = 12.0 + (4.0 * s if covariate else 0.0)
    age = np.clip(rng.normal(age_mu, age_sigma, n), 18, 95)

    employment_years = np.clip(rng.gamma(shape=2.0, scale=3.2, size=n) * (age / 42.0), 0, 45)

    income_base = 32_000 + 1_450 * employment_years + 380 * age
    income_noise = rng.lognormal(mean=0.0, sigma=0.42, size=n)
    annual_income = np.clip(income_base * income_noise, 12_000, 900_000)
    if covariate:
        # Incomes fall and spread out -- the classic post-shock production shift.
        annual_income = np.clip(annual_income * (1.0 - 0.28 * s), 8_000, 900_000)

    num_credit_lines = np.clip(
        rng.poisson(lam=np.clip(2.0 + age / 14.0, 1, 12), size=n), 0, 45
    )
    num_late_payments_12m = np.clip(
        rng.poisson(lam=0.55 + (0.9 * s if covariate else 0.0), size=n), 0, 30
    )
    credit_utilization = np.clip(
        rng.beta(2.0, 4.2, size=n) * (1.0 + (0.55 * s if covariate else 0.0)), 0.0, 1.95
    )

    credit_score = (
        720.0
        - 26.0 * num_late_payments_12m
        - 95.0 * credit_utilization
        + 1.15 * employment_years
        + 0.00028 * annual_income
        + rng.normal(0, 33, n)
    )
    credit_score = np.clip(credit_score, 300, 850)

    loan_amount = np.clip(
        annual_income * rng.uniform(0.08, 0.62, n) + rng.normal(0, 3_500, n),
        1_000,
        1_500_000,
    )
    loan_term_months = rng.choice(
        np.array([12, 24, 36, 48, 60, 84, 120]),
        size=n,
        p=np.array([0.07, 0.14, 0.31, 0.19, 0.16, 0.08, 0.05]),
    )

    monthly_payment = loan_amount / np.maximum(loan_term_months, 1)
    debt_to_income = np.clip(
        (monthly_payment * 12.0) / np.maximum(annual_income, 1.0) + rng.normal(0.06, 0.05, n),
        0.0,
        2.9,
    )

    # ---------------- categorical features -------------------------------- #
    emp_p = _shift_probs(_EMPLOYMENT_P, 4, s) if categorical_drift else _EMPLOYMENT_P
    reg_p = _shift_probs(_REGION_P, 2, s) if categorical_drift else _REGION_P
    pur_p = _shift_probs(_PURPOSE_P, 4, s) if categorical_drift else _PURPOSE_P

    employment_type = _sample_categorical(rng, EMPLOYMENT_TYPES, emp_p, n)
    housing_status = _sample_categorical(rng, HOUSING_STATUS, _HOUSING_P, n)
    loan_purpose = _sample_categorical(rng, LOAN_PURPOSES, pur_p, n)
    region = _sample_categorical(rng, REGIONS, reg_p, n)

    # ---------------- target ----------------------------------------------- #
    # Concept drift changes how strongly (and for utilisation, in which
    # direction) the features map to the outcome, while leaving the input
    # distribution alone. That is precisely the case unlabelled monitoring
    # cannot see -- see docs/monitoring.md.
    weights = _concept_weights(s if concept else 0.0)

    raw = _risk_score(
        weights,
        credit_score=credit_score,
        debt_to_income=debt_to_income,
        credit_utilization=credit_utilization,
        num_late_payments_12m=num_late_payments_12m,
        employment_years=employment_years,
        annual_income=annual_income,
        loan_amount=loan_amount,
        age=age,
        employment_type=employment_type,
        housing_status=housing_status,
        loan_purpose=loan_purpose,
        region=region,
    )

    # Standardise against *fixed* reference moments, not this sample's own
    # moments: otherwise covariate drift would be normalised away and the
    # drifted datasets would look identical to the baseline.
    mean, std = _reference_moments(weights)
    z = (raw - mean) / std

    target_rate = (
        float(np.clip(spec.class_imbalance, 0.001, 0.999))
        if spec.class_imbalance is not None
        else _BASE_POSITIVE_RATE
    )
    intercept = _solve_intercept(weights, target_rate)
    logits = _SIGNAL_STRENGTH * z + intercept

    probability = 1.0 / (1.0 + np.exp(-logits))
    default = (rng.uniform(0, 1, n) < probability).astype(int)

    # ---------------- assembly --------------------------------------------- #
    day_offsets = rng.integers(0, max(spec.days_span, 1), n)
    dates = [spec.start_date + timedelta(days=int(d)) for d in day_offsets]

    frame = pd.DataFrame(
        {
            config.id_column: [f"APP-{spec.seed:04d}-{i:07d}" for i in range(n)],
            config.timestamp_column: [d.strftime("%Y-%m-%d") for d in dates],
            "age": np.round(age, 1),
            "annual_income": np.round(annual_income, 2),
            "loan_amount": np.round(loan_amount, 2),
            "loan_term_months": loan_term_months.astype(int),
            "credit_score": np.round(credit_score, 1),
            "debt_to_income": np.round(debt_to_income, 4),
            "employment_years": np.round(employment_years, 2),
            "num_credit_lines": num_credit_lines.astype(int),
            "num_late_payments_12m": num_late_payments_12m.astype(int),
            "credit_utilization": np.round(credit_utilization, 4),
            "employment_type": employment_type,
            "housing_status": housing_status,
            "loan_purpose": loan_purpose,
            "region": region,
            config.target_column: default,
        }
    )
    frame = frame.sort_values(config.timestamp_column).reset_index(drop=True)

    frame = _apply_corruption(frame, spec, config, rng)

    logger.info(
        "dataset.generated",
        extra={
            "rows": len(frame),
            "seed": spec.seed,
            "drift": spec.drift,
            "positive_rate": round(float(frame[config.target_column].mean()), 4),
        },
    )
    return frame


def _apply_corruption(
    frame: pd.DataFrame,
    spec: GenerationSpec,
    config: DataConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject the specific defects the validation suite is meant to catch."""
    n = len(frame)

    if spec.missing_fraction > 0:
        targets = ["annual_income", "credit_score", "debt_to_income", "employment_type"]
        for column in targets:
            idx = rng.choice(n, size=int(n * spec.missing_fraction), replace=False)
            frame.loc[idx, column] = None

    if spec.outlier_fraction > 0:
        idx = rng.choice(n, size=max(1, int(n * spec.outlier_fraction)), replace=False)
        frame.loc[idx, "annual_income"] = frame.loc[idx, "annual_income"] * 85.0
        frame.loc[idx, "loan_amount"] = frame.loc[idx, "loan_amount"] * 60.0

    if spec.invalid_value_fraction > 0:
        k = max(1, int(n * spec.invalid_value_fraction))
        idx = rng.choice(n, size=k, replace=False)
        frame.loc[idx[: k // 2 or 1], "age"] = -5.0
        frame.loc[idx[k // 2 :], "credit_score"] = 1_400.0

    if spec.duplicate_fraction > 0:
        k = max(1, int(n * spec.duplicate_fraction))
        idx = rng.choice(n, size=k, replace=False)
        frame = pd.concat([frame, frame.iloc[idx]], ignore_index=True)

    return frame


# --------------------------------------------------------------------------- #
# Convenience builders used by the CLI, the Makefile and the demo
# --------------------------------------------------------------------------- #
def build_training_dataset(n_rows: int | None = None, seed: int | None = None) -> pd.DataFrame:
    cfg = get_settings().data
    return generate_dataset(
        GenerationSpec(n_rows=n_rows or cfg.train_rows, seed=seed or cfg.random_seed),
        cfg,
    )


def build_production_dataset(
    n_rows: int = 2000,
    drift: DriftMode = "none",
    drift_strength: float = 1.0,
    seed: int = 202,
) -> pd.DataFrame:
    """Later traffic: a different seed, a later date window, optional drift."""
    cfg = get_settings().data
    return generate_dataset(
        GenerationSpec(
            n_rows=n_rows,
            seed=seed,
            drift=drift,
            drift_strength=drift_strength,
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            days_span=120,
        ),
        cfg,
    )


def build_invalid_dataset(n_rows: int = 1500, seed: int = 909) -> pd.DataFrame:
    """A dataset that must fail the validation gate."""
    cfg = get_settings().data
    return generate_dataset(
        GenerationSpec(
            n_rows=n_rows,
            seed=seed,
            missing_fraction=0.22,
            duplicate_fraction=0.08,
            outlier_fraction=0.09,
            invalid_value_fraction=0.05,
        ),
        cfg,
    )


def write_dataset(frame: pd.DataFrame, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    logger.info("dataset.written", extra={"path": str(target), "rows": len(frame)})
    return target


def bootstrap_sample_data(force: bool = False) -> dict[str, Path]:
    """Materialise the standard set of sample datasets under ``data/``.

    Idempotent: existing files are left alone unless ``force`` is set, so a
    developer's edited fixture is never silently overwritten.
    """
    settings = get_settings()
    settings.paths.ensure()
    sample = settings.paths.sample_dir
    raw = settings.paths.raw_dir
    reference = settings.paths.reference_dir

    targets: dict[str, tuple[Path, Callable[[], pd.DataFrame]]] = {
        "train": (
            raw / "loan_default_v1.csv",
            lambda: build_training_dataset(),
        ),
        "train_v2": (
            raw / "loan_default_v2.csv",
            lambda: generate_dataset(
                GenerationSpec(
                    n_rows=settings.data.train_rows + 2000,
                    seed=settings.data.random_seed + 1,
                )
            ),
        ),
        "production": (
            sample / "production_traffic.csv",
            lambda: build_production_dataset(n_rows=2000, drift="none"),
        ),
        "drifted": (
            sample / "production_drifted.csv",
            lambda: build_production_dataset(n_rows=2000, drift="severe", drift_strength=1.0),
        ),
        "concept_drifted": (
            sample / "production_concept_drift.csv",
            lambda: build_production_dataset(n_rows=2000, drift="concept", drift_strength=1.0),
        ),
        "invalid": (sample / "invalid_dataset.csv", build_invalid_dataset),
        "reference": (
            reference / "reference_window.csv",
            lambda: build_production_dataset(n_rows=3000, drift="none", seed=777),
        ),
    }

    written: dict[str, Path] = {}
    for name, (path, builder) in targets.items():
        if path.exists() and not force:
            written[name] = path
            continue
        write_dataset(builder(), path)
        written[name] = path
    return written
