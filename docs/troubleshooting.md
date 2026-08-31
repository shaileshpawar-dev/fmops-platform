# Troubleshooting

Every platform error carries a stable `code`. Find it in the response or the log
line, then look it up here.

```json
{
  "error": {
    "code": "model_not_loaded",
    "message": "no model is available to serve: ...",
    "request_id": "a1fc00b523ba4055",
    "details": { "model": "loan_default_classifier" }
  }
}
```

Logs are JSON; every line inside a request carries the same `request_id`:

```bash
docker compose logs api | jq 'select(.request_id=="a1fc00b523ba4055")'
```

---

## Serving

### `model_not_loaded` (503)

**Nothing is deployed and no Production/Staging version exists.**

```bash
fmops models          # is anything registered? in what stage?
fmops status
```

- No versions at all → `make demo`, or `fmops train && fmops promote`.
- Versions exist but all in Development → the approval gate rejected them:
  ```bash
  curl -X POST localhost:8000/api/v1/models/loan_default_classifier/versions/1/evaluate-gate
  ```
  The response lists every check with its observed value and threshold.
- A version is in Production but serving still fails → the artifact is missing;
  see below.

### `model_not_loaded` with "artifact missing"

The registry points at a file that is not there. Usual causes: `artifacts/` was
deleted, a container has no volume mounted, or S3 credentials are absent.

```bash
fmops models
python -c "
from app.registry.factory import get_registry
print(get_registry().get('loan_default_classifier', 1).artifact_uri)"
```

Retrain, or restore the artifact. The registry deliberately does not fall back to
a different version — silently serving a model other than the one the registry
names would be worse than a 503.

### Predictions are slow on the first request

Expected, and already mitigated. Deserialising a pipeline is cheap; the first
`predict` through it is not — scikit-learn and numpy do lazy setup on first call
(~2.5s cold, ~35ms warm). The app runs a throwaway prediction at startup
(`_warm_models`). If you still see it, check for `model_cache.warmed` in the
startup logs.

### `prediction_failed` (400)

The model could not score the input. Usually a schema mismatch after retraining
with different features.

```bash
curl localhost:8000/api/v1/model | jq .feature_names
```

### `request_validation_failed` (422)

The input violated the declared ranges (`credit_score` 300–850, `age` 18–100, …).
`details.violations` names the offending fields. These constraints mirror the
training-time validation suite on purpose: an input that would have been rejected
at training time is rejected at inference time.

---

## Data and training

### `data_validation_failed` (422)

The pipeline stopped at the gate. **This is the system working.**

```bash
fmops data validate --json | jq '.results[] | select(.success==false)'
```

| Failure | Likely cause |
|---|---|
| `column_missing_fraction_below_limit` | upstream nulls; fix the source or raise the limit deliberately |
| `duplicate_row_fraction_below_limit` | double-loaded extract |
| `column_values_between` | unit change, sentinel values (-1, 9999), or corruption |
| `column_values_in_set` | a genuinely new category — decide whether to retrain rather than widen the schema thoughtlessly |
| `class_distribution_within_bounds` | a filtered or mislabelled extract |
| `table_row_count_above_minimum` | truncated extract |

Change limits in `configs/<env>.yaml` under `validation:` — but understand *why*
before you loosen anything. The gate exists to stop exactly this.

### `dataset_not_found` with "content hash mismatch"

The file changed after registration. The platform refuses to load it because the
recorded version no longer describes the bytes on disk. Register the new file as
a new version:

```bash
fmops data versions
python -c "
from app.data.versioning import get_dataset_registry
print(get_dataset_registry().register('data/raw/loan_default_v1.csv').version)"
```

### `training_failed`

`details` carries the algorithm and resolved params. Common causes: a single
class in the training data (the pipeline checks this explicitly), all-NaN
features after engineering, or a bad hyperparameter combination.

### `tuning_failed`

Every trial failed. `details.first_error` has the real reason — usually an
invalid parameter for the chosen estimator. A *single* failed trial is recorded
and the search continues; only a total failure raises.

### `dependency_missing`

An optional backend was requested but is not installed:

```bash
pip install -e ".[boosting]"    # xgboost, lightgbm
pip install -e ".[quality]"     # great-expectations, evidently
pip install -e ".[aws]"         # boto3, sagemaker
pip install -e ".[llm]"         # anthropic, google-genai, openai
```

The platform raises rather than silently substituting a different algorithm,
which would make the registry lie about what was trained.

---

## Approval and promotion

### The model trained but was not promoted

Two independent gates. The output says which one:

```
Comparison (roc_auc): candidate 0.8821 vs baseline 0.8824 (-0.0003) -> reject
Promotion: NO -> stage Development
```

That is the champion/challenger gate, working. The candidate did not beat
production by `min_improvement`.

If you genuinely need to deploy it anyway:

```bash
fmops deploy --version 3 --force --reason "explain yourself here"
```

`--force` is written to the audit log and logged as
`deployment.forced_undeployable_stage`.

### `invalid_stage_transition` (409)

You attempted an illegal move (e.g. Development → Production). The response lists
the legal targets. Promote stepwise, or use `fmops promote` which walks the chain.

### `approval_rejected`

`checks` lists every check with observed vs threshold. Either improve the model
or change the threshold deliberately in `configs/<env>.yaml`.

---

## Deployment

### `deployment_failed`: "only Staging, Production, Validation versions may be deployed"

The version has not cleared the gate. Run `fmops promote`, or `--force` with a
reason.

### Canary rolled back

```bash
curl localhost:8000/api/v1/deployments | jq '.[0].events'
```

Look at the last `canary.step_evaluated` event. If `requests` is small, the
rollback may have been triggered by noise — raise
`canary_min_requests_per_step` or lengthen `canary_step_seconds` so each step
gathers real evidence.

### `rollback_failed`: "no rollback target"

There is no previous version and nothing archived. Deploy a specific version
explicitly:

```bash
fmops rollback --to-version 2
```

### Endpoint health shows `routing_configured: false`

The in-memory routing table is process-local. A fresh CLI process rehydrates it
from the persisted deployment record automatically. If it still reports false,
there is genuinely no active deployment:

```bash
curl localhost:8000/api/v1/deployments/current
```

---

## Drift and retraining

### `insufficient_data`: drift scan needs more rows

```bash
fmops simulate --rows 500 --label-fraction 0.3
fmops drift scan
```

Or lower `FMOPS_DRIFT__MIN_SAMPLES` — but below ~1000 rows PSI on a 10-bin
histogram is mostly noise.

### Drift detected but the retraining trigger does not fire

```bash
fmops retrain --check-only
```

Look for `suppressed_by_cooldown`. After a retraining event, automatic triggers
are suppressed for `cooldown_minutes` (60 by default, 360 in production). Force
past it deliberately:

```bash
fmops retrain --force
```

### Concept drift always says `unavailable`

Working as designed. Concept drift cannot be computed from unlabelled data. Submit
ground truth:

```bash
curl -X POST localhost:8000/api/v1/feedback \
  -d '{"request_id":"<from a prediction response>","actual_label":1}'
```

At least 50 labelled rows with both classes are needed. See
[`monitoring.md`](monitoring.md#concept-drift-cannot-be-measured-from-unlabelled-data).

### False-positive drift on clean traffic

Check whether the reference window is right:

```bash
curl localhost:8000/api/v1/drift/latest | jq '.report.n_reference, .report.n_current'
```

If the reference dataset for the serving model was unresolvable, the platform
falls back to the latest registered dataset and logs
`monitoring.reference_dataset_unavailable` — comparing against the wrong
reference produces exactly this symptom.

### Retraining exits 2 in CI

**That is a success, not a failure.** Exit 2 means a candidate was trained and
correctly rejected. Only exit 1 is a real failure. See
[`mlops.md`](mlops.md#outcomes).

---

## LLMOps

### Responses look generic / `provider: "mock"`

You are on the mock provider. Either no API key is set, or the configured
provider is unavailable and development fell back. Check:

```bash
curl localhost:8000/api/v1/llm/providers | jq
```

Each unavailable provider explains exactly why. In production this raises instead
of falling back.

### `safety_violation` (422)

The safety screen blocked the request. `details.findings` names the checks. To
screen without blocking:

```bash
FMOPS_LLM__SAFETY_BLOCK_ON_VIOLATION=false
```

Note the screen both over- and under-reports; see the limitations in
[`llmops.md`](llmops.md#what-it-is-not).

### `prompt_not_found` with `missing_variables`

The template needs variables you did not supply. Rendering with a literal
`{placeholder}` would silently degrade output, so it raises.

```bash
curl localhost:8000/api/v1/llm/prompts/support_summarizer/1.1.0 | jq .variables
```

### Cost shows $0.00 for a real provider

The model is not in the price table. Look for `cost.model_not_priced` in the
logs — it names the model. Add it:

```bash
FMOPS_LLM__PRICING='{"my-model":{"input":1.0,"output":5.0}}'
```

### `compare()` refuses two evaluations

You are comparing a mock-provider run against a real one. Mock scores measure the
harness, not model quality. Re-run both against the same provider.

---

## Infrastructure

### MLflow: "filesystem tracking backend is in maintenance mode"

MLflow 3.x deprecated the `file:` store. The platform defaults to SQLite, which
also matches the production RDS shape. If you overrode `tracking_uri` with a
`file:` path, use `sqlite:///path/mlflow.db` instead.

### `provider_unavailable`: MLflow unreachable

The platform falls back to the local tracker and logs an ERROR naming the impact
("runs will NOT appear in MLflow for this session"). Runs still complete. Fix the
tracking URI, or set `FMOPS_TRACKING__BACKEND=local` deliberately.

### `database is locked`

SQLite serialises writes. Under concurrent load from multiple API workers this
surfaces. Reduce `FMOPS_SERVER__WORKERS` to 1, or move to Postgres — see
[`deployment.md`](deployment.md).

### AWS: `provider_unavailable` or `configuration_error`

```bash
fmops aws status
```

Reports whether boto3 is installed, whether credentials resolve, the account id,
and which settings are missing.

---

## Diagnostics

```bash
fmops status                                  # everything at a glance
fmops config                                  # effective config, secrets redacted
curl localhost:8000/health | jq               # per-component health
curl localhost:8000/api/v1/dashboard | jq     # the full dashboard payload
curl localhost:8000/api/v1/audit | jq         # who changed what
curl localhost:8000/api/v1/alerts | jq
curl localhost:8000/metrics | grep fmops_

# Reproduce a specific model version
python -c "
from app.registry.factory import get_registry
v = get_registry().get('loan_default_classifier', 3)
print('dataset :', v.dataset_version, v.dataset_hash)
print('commit  :', v.git_commit)
print('params  :', v.params)
print('metrics :', v.metrics)"
```

### Start completely fresh

```bash
make clean-state    # deletes all models, the registry and the database
make demo
```
