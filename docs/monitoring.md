# Monitoring and drift

## The drift taxonomy

Four things get called "drift". They are not the same, and conflating them is
how teams end up with monitoring that looks thorough and detects nothing.

| Type | What changed | Measurable without labels? | How this platform handles it |
|---|---|---|---|
| **Data drift** | P(x) — the input distribution | **yes** | measured continuously |
| **Feature drift** | P(xᵢ) for one feature | **yes** | measured per feature |
| **Prediction drift** | P(ŷ) — the output distribution | **yes** | measured; a *symptom*, not a cause |
| **Concept drift** | P(y \| x) — what the inputs *mean* | **NO** | reported `unavailable` unless labels exist |

### Concept drift cannot be measured from unlabelled data

This is the claim most drift dashboards get wrong, so it is worth being precise.

Concept drift means the relationship between inputs and outcome changed. The
same application that was low-risk last year is high-risk now — **with identical
input distributions**. No amount of input monitoring detects this, because
nothing about the inputs changed.

This platform demonstrates it rather than asserting it. `app/data/generator.py`
has a `concept` drift mode that alters the coefficients mapping features to the
outcome (weakening credit score, strengthening DTI, reversing utilisation) while
leaving every input distribution untouched. A drift scan over that data reports
**zero drifted features**, and there is a test that fails if it ever reports
otherwise:

```python
def test_concept_drift_mode_leaves_inputs_unchanged(settings, valid_frame):
    concept = build_production_dataset(n_rows=1500, drift="concept", seed=44)
    baseline = build_production_dataset(n_rows=1500, drift="none", seed=44)
    results = engine.analyse(baseline, concept, ...)
    assert not any(r.drifted for r in results)
```

So the platform reports:

```
concept drift     : unavailable
```

with this explanation attached to every report:

> Concept drift (a change in P(y|x)) cannot be computed from unlabelled
> production data. No ground-truth labels were available for this window, so only
> data, feature and prediction drift were measured. Prediction drift is a proxy
> signal, not a measurement of concept drift. Submit labels via
> POST /api/v1/feedback to enable it.

Once labels arrive, concept drift becomes `measured` and is quantified as the
relative degradation in live ROC-AUC against the model's offline baseline:

```
concept drift     : measured (0.1255)
   Measured from 369 labelled production rows: live ROC-AUC 0.7717 vs offline
   baseline 0.8824 (12.5% relative degradation).
```

With fewer than 50 labelled rows containing both classes, the status is
`insufficient_labels` rather than a noisy number.

## Choice of statistic

**The headline `score` is Jensen-Shannon distance** between the reference and
current distributions: bounded in [0, 1], symmetric, and defined for both numeric
(binned on reference quantiles) and categorical features — so scores are
comparable across feature types and can be meaningfully averaged.

**The `drifted` flag uses PSI (an effect size) as the primary criterion**, with
the KS / χ² p-value as supporting evidence:

```python
drifted = psi > psi_threshold or (p_value < alpha and psi > PSI_MODERATE)
```

This ordering is deliberate. With production windows of a few thousand rows, KS
p-values are driven below 0.05 by **sample size alone** — a p-value-only rule
flags essentially everything as drifted and the alert becomes worthless. PSI
measures how *large* the shift is, which is what actually matters operationally.

PSI convention: `< 0.10` stable, `0.10–0.25` moderate, `> 0.25` significant.

## The dataset-level verdict

A dataset is "drifted" when **either**:

- the mean drift score exceeds `drift.threshold` (default 0.20), **or**
- the share of drifted features reaches `drift.dataset_drift_share` (default 0.30).

Either rule alone misses real cases: one catastrophically shifted feature is
diluted by an average, and many mildly shifted features never breach a
per-feature threshold. The alert names **which rule fired**:

```
Drift detected because 57% of features drifted (limit 30%).
8 of 14 features drifted: age, annual_income, credit_score, …
Mean drift score 0.1679.
```

A drift alert that claims a threshold was breached when it was not destroys trust
in every subsequent alert.

## The reference window

Production traffic is compared against **the training data the serving model was
actually fitted on**, resolved via the `dataset_version` recorded on the model
version. If that dataset is no longer resolvable the platform falls back to the
latest registered dataset **and logs a warning** — a comparison against the wrong
reference is worse than none.

## Metrics

### Model quality

| Source | Metric | Availability |
|---|---|---|
| Offline (training) | accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, Brier | always |
| Live (labelled traffic) | accuracy, precision, recall, F1, ROC-AUC | **only with labels** |

`GET /api/v1/monitoring/performance` returns `available: false` with an
explanation when no labels exist. It never estimates production accuracy from
unlabelled data.

### System

Latency (p50/p95/p99, histogram), throughput, error rate, HTTP status
distribution, CPU, memory, process RSS, and GPU utilisation **only when a GPU is
actually detected**. A dashboard full of 0% GPU is indistinguishable from an idle
GPU, so absence is reported as `gpu_available: false`.

**Sampling is done off the request path.** A background thread samples on
`monitoring.resource_sample_seconds` (default 15s) and request handlers read the
most recent sample via `latest_resources()`. Measuring the host synchronously on
every request is the wrong shape: a dashboard polling every 15 seconds should
read what the sampler already collected.

**Open file descriptors are opt-in** (`monitoring.sample_open_files`, default
`false`). `psutil.Process.open_files()` enumerates every handle the OS knows
about — **measured at ~1.8s on Windows** — which is fine for a background
diagnostic and completely unacceptable on a request path. Turn it on when you
are hunting a descriptor leak, and expect the sampler to get slower:

```bash
FMOPS_MONITORING__SAMPLE_OPEN_FILES=true
```

### Business / operational

Request volume, per-version traffic split, predicted-positive rate,
model-usage-by-version, LLM token consumption and estimated spend.

## Prometheus

Scrape `/metrics`. Around 30 metric families:

```
fmops_predictions_total{model_name,model_version,variant,outcome}
fmops_prediction_latency_seconds{model_name,model_version,variant}
fmops_prediction_probability{model_name,model_version}     # prediction-drift signal
fmops_model_metric{model_name,model_version,metric}         # offline
fmops_live_model_metric{model_name,model_version,metric}    # from labels
fmops_drift_score{model_name,drift_type}
fmops_feature_drift_score{model_name,feature}
fmops_drift_detected{model_name}
fmops_deployment_traffic_percent{endpoint,model_name,model_version}
fmops_rollbacks_total{endpoint,reason}
fmops_retraining_events_total{trigger,status,decision}
fmops_approval_decisions_total{model_name,decision}
fmops_llm_tokens_total{provider,model,direction}
fmops_llm_cost_usd_total{provider,model}
fmops_llm_safety_findings_total{check,severity}
```

**Cardinality note:** the HTTP metrics label on the *templated* route path
(`/api/v1/models/{name}/versions/{version}`), never the raw path. Labelling on
raw paths creates one time series per model id — the classic way to melt a
Prometheus instance.

### In-process counters vs the persisted log

An important distinction: Prometheus counters are process-local and reset on
restart. **Canary promotion and rollback decisions read the persisted inference
log**, not the counters, so a decision is identical across workers and survives a
restart mid-rollout.

## Grafana

Two provisioned dashboards (`monitoring/grafana/dashboards/`):

- **FMOps – MLOps**: live version, offline vs live quality, drift score and
  per-feature bar gauge, traffic split during a canary, latency percentiles
  against the SLO, retraining outcomes, approval decisions, resources.
- **FMOps – LLMOps**: token throughput, cumulative spend by model, calls by
  prompt version (a prompt rollout looks exactly like a model canary), evaluation
  scores per prompt version, safety findings, blocked requests.

## SLOs and the watchdog

```yaml
monitoring:
  latency_slo_ms: 250.0
  error_rate_slo: 0.02
```

```bash
curl -X POST localhost:8000/api/v1/monitoring/watchdog?window_minutes=15
```

The watchdog evaluates live SLOs and raises alerts. A window with **no traffic**
returns "nothing to evaluate" rather than reporting a perfect 0% error rate —
which would otherwise mask a completely dead endpoint.

## Alerts

Sinks are pluggable: `log`, `database`, `file`, `webhook` (Slack-shaped), `sns`.

**De-duplication.** An alert's fingerprint is derived from its category, title
and context identity fields. An identical fingerprint inside
`dedupe_window_seconds` is suppressed — otherwise a drift scan every five minutes
produces 288 identical pages a day. Suppression is logged at debug level, never
silently.

**Sink failures never propagate.** A webhook being down must not break the drift
scan that raised the alert. Failures are logged as errors and the alert is still
persisted locally.

```bash
curl localhost:8000/api/v1/alerts
curl -X POST localhost:8000/api/v1/alerts/{id}/acknowledge
```

## CloudWatch

When `monitoring.cloudwatch_enabled` is true, a **deliberately chosen subset** of
metrics is mirrored — not everything. CloudWatch custom metrics bill per metric
per month, and mirroring a per-feature drift gauge across 14 features and 3
environments adds up for no operational benefit.

Terraform also creates log metric filters that extract drift detections,
rollbacks, application errors and rejected retraining candidates from the
structured JSON logs, plus alarms and a dashboard.

## Practical guidance

**Scan cadence.** Drift is a slow signal. Scanning every five minutes on a
1000-row window mostly measures sampling noise. Daily, or per-N-thousand
requests, is usually right.

**Window size.** `drift.min_samples` (default 200) is a floor, not a
recommendation. Below ~1000 rows, PSI on a 10-bin histogram is noisy.

**Sampling.** `prediction_log_sample_rate < 1.0` reduces write load but weakens
drift statistics. Production defaults to 0.25 because 25% of high-volume traffic
is still plenty; on a low-volume endpoint, keep it at 1.0.

**Drift is not automatically bad.** A drift alert says the world changed, not
that the model is broken. The model may be perfectly robust to the shift. That is
exactly why the retraining pipeline *compares* rather than *assumes*: drift
triggers an investigation, and the champion/challenger gate decides the outcome.
