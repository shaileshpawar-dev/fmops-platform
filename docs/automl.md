# AutoML

## What it is

Given a registered dataset, AutoML profiles it, recommends a target column and
a problem type, proposes candidate models, trains each one, ranks them, and
hands the winner to the ordinary approval gate.

```
Dataset
  ↓
Profiler          columns, roles, warnings
  ↓
Target            recommended, with reasons -- confirmed by the user
  ↓
Problem type      inferred from dtype, cardinality and unique ratio
  ↓
Candidates        from the installed algorithms, with reasons
  ↓
Training          app.training.train.train_model, once per candidate
  ↓
Leaderboard       ranked by the configured primary metric
  ↓
Champion          best candidate for this run
  ↓
Registry          existing model registry
  ↓
Approval gate     existing thresholds, unchanged
```

**No model chooses the model.** Every recommendation is a deterministic rule
over observable dataset properties, and every one carries the observations that
produced it. That is what makes them reviewable, repeatable, and arguable.
There is no LLM anywhere in this path.

## What it can and cannot fit

**Binary classification only.** This is a property of the training stack, not a
limitation of the profiler:

- the model factory builds classifiers,
- `split_features_target` casts the label with `.astype(int)`,
- evaluation computes ROC-AUC from `predict_proba[:, 1]`.

The profiler still *detects* regression and multiclass targets. It says so and
refuses, with `422 unsupported_problem_type`, rather than casting a continuous
target to an integer and reporting a meaningless ROC-AUC. Supporting them would
mean new estimators, new metrics and new preprocessing — a change to the
training core, not an addition to it.

## Target recommendation

Columns are scored on signals that are individually weak and useful together:

| Signal | Effect |
|---|---|
| Name in a list of common outcome names (`default`, `churn`, `fraud`, `label`, …) | strong positive |
| Exactly two distinct values | strong positive |
| Positioned last | weak positive, breaks ties only |
| Missing values present | negative — labels are rarely missing |
| Near-unique, or named like a key | disqualified |
| Constant, datetime or entirely empty | disqualified |

Confidence is downgraded to `low` when the runner-up scores within 1.0 of the
winner: two comparable candidates is ambiguity, not confidence.

When nothing plausible is found the API returns **no suggestion at all**.
Training never starts on a guess — the caller must confirm a target.

## Column roles

Every column is `feature`, `target` or `excluded`, and an exclusion always
states why:

- **likely identifier** — near-unique *and* an integer or a string. A float is
  never excluded for uniqueness alone: income and price are naturally almost
  unique, and treating that as an identifier signal silently deletes the best
  features in most real datasets.
- **datetime** — including date-like strings, which arrive from CSV as plain
  objects and would otherwise be one-hot encoded into thousands of columns.
- **constant** — one distinct value carries no signal.

## Warnings

Reported, never acted on automatically: high cardinality, missing values,
constant features, class imbalance, too few rows, and **potential leakage**.

Leakage detection is name-based and therefore a *risk*, not a finding. The
wording is deliberate: a column called `final_outcome` may be perfectly
legitimate, so the platform says "confirm it is available at prediction time"
rather than "leakage detected".

Imbalance is surfaced but the data is **not** resampled. Silently oversampling
would change the model without the user knowing; instead the metric note
explains that ROC-AUC can look strong while the minority class is being missed,
and F1 and recall are shown alongside it.

## Candidates

Only algorithms `available_algorithms()` reports as installable are offered, in
tiers rather than invented scores:

| Tier | Meaning |
|---|---|
| `recommended` | suits this dataset's size and feature mix |
| `baseline` | logistic regression, always included as a reference point — if a boosted model cannot beat it, the extra complexity is not earning anything |
| `optional` | usable, but the dataset is smaller than the algorithm prefers |
| `unsuitable` | not installed, or the problem type is unsupported |

A "suitability score" would imply a precision these heuristics do not have.

## Ranking

```
primary metric (desc) → F1 (desc) → shorter training time → algorithm name
```

Failed candidates are never ranked and can never win. The tie-breakers make the
order deterministic, so two runs over the same results produce the same
leaderboard. The rule is stored on the run and shown in the UI.

## Partial failure

One candidate failing does not sink the run:

```
Model A ✓   Model B ✕   Model C ✓   →  completed_with_warnings
```

If every candidate fails the run is `failed`, with each candidate's error kept.

## The gate is not bypassed

The winner is registered and then judged by `register_and_promote` — the same
absolute thresholds and the same `min_improvement` comparison as any other
model. AutoML picks a candidate; it does not decide what reaches production.

A run whose best candidate is rejected is a **correct** outcome:

```
Best candidate  ROC-AUC 0.9133   F1 0.6313
Production      min_f1 0.65
Result          registered in Development, promotion REJECTED
```

Production thresholds are never relaxed to make a run look successful.

## API

```
GET  /api/v1/automl/profile/{version}[?target=col]   profile + recommendations
POST /api/v1/automl/runs                            202 + run id
GET  /api/v1/automl/runs
GET  /api/v1/automl/runs/{run_id}                   config, profile, leaderboard, gate
GET  /api/v1/automl/runs/{run_id}/candidates
```

The request surface is closed — unknown fields are rejected, and there is no
passthrough into training configuration. `max_models` is capped at 5 so opening
a page cannot queue unbounded training.

## Execution and cost

Runs execute as FastAPI `BackgroundTasks` **inside the API process**, the same
mechanism the training and retraining endpoints use. On the deployed single
Fargate task this means **training competes with request handling for the
task's CPU**, and a run in flight is lost if the process restarts (orphans are
reconciled to `failed` with the reason recorded). Acceptable for this portfolio
deployment; a multi-replica setup needs a real job runner.

Each candidate is a full training run, so *n* candidates cost roughly *n* times
one training run. The UI states how many models a run will train before it
starts.

## Reproducibility

Each run records the dataset version, target, problem type, candidate list,
primary metric, HPO setting, target stage, per-candidate metrics and durations,
the winner, the registered model version, the ranking rule, and timestamps —
all visible on the run detail page.
