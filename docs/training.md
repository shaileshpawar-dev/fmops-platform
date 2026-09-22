# Training from the API

## What a run is

`POST /api/v1/training/runs` records a run and executes
`pipelines.training_pipeline.run` in the background. The API adds a durable id
to poll and nothing else — validation, feature engineering, training, tuning,
evaluation, registration, the champion/challenger comparison and promotion all
stay in the pipeline, which is the same code path the CLI and the retraining
job use.

```bash
curl -X POST localhost:8000/api/v1/training/runs \
  -H 'Content-Type: application/json' \
  -d '{"dataset_version":"v2-f4a45097","algorithm":"logistic_regression",
       "promote":true,"target_stage":"Staging"}'
```

```json
{
  "run_id": "train-726738f5043344cc",
  "status": "queued",
  "poll": "/api/v1/training/runs/train-726738f5043344cc"
}
```

**202, not 201.** The run has been accepted, not completed. A `Location`
header points at the record.

```bash
curl localhost:8000/api/v1/training/runs/train-726738f5043344cc
curl localhost:8000/api/v1/training/runs?limit=25
```

## Request surface

| Field | Meaning |
|---|---|
| `dataset_version` | Registered version. Omit for the latest. |
| `algorithm` | From `GET /api/v1/models/algorithms`. Omit for the configured default. |
| `tune` | Run hyperparameter search before the final fit. |
| `promote` | Register and put the result through the approval gate. |
| `target_stage` | `Staging` or `Production`, if the gate passes. |

Unknown fields are **rejected**, not ignored. There is no free-form parameter
passthrough: a request must not be able to reach arbitrary code, a shell or a
filesystem path. An algorithm whose optional backend is not installed is
refused up front rather than minutes into a background run.

`promote: true` asks for the *attempt*. Promotion still requires clearing the
absolute gate **and** beating the incumbent by `approval.min_improvement`.

## Statuses

```
queued -> running -> completed
                  -> rejected
                  -> failed
```

| Status | Pipeline exit | Meaning |
|---|---|---|
| `completed` | 0 | Ran through. May or may not have been promoted — check `report.promoted`. |
| `rejected` | 2 | Ran fine; the candidate did not clear the gate. A result, not a failure. |
| `failed` | 1 | Validation blocked it, or the run raised. |

Distinguishing `rejected` from `failed` matters: a challenger that loses to the
champion is the system working, and showing it as a failure would train
everyone to ignore the failure count.

## Execution model

A run is queued as a **job** — a database row claimed atomically by the API's
in-process worker — and the response carries its `job_id`. See
[jobs.md](jobs.md). Be clear about what that does and does not give you:

- Jobs, their logs and their results persist in the platform database, and the
  queue is safe for several processes sharing that database.
- A run **running** when the process dies is failed on recovery, with the
  reason recorded, rather than showing progress nothing is making. It can be
  retried; it is never silently resumed.
- Training competes with request handling for the task's CPU. On the deployed
  0.5 vCPU task, a run makes the API slower while it lasts.
- It is not a distributed job system: the worker lives in the API container,
  and the database is local to it.

A run trains the dataset's own target (`target_column`) under a **model name**
(`model_name`), with an optional explicit `positive_label`. The name is checked
before the job is queued: a malformed name, a non-binary target, or a target
that differs from the name's existing lineage is refused with 422/409 at once.

## From the console

The Training page offers only algorithms `available_algorithms()` reports as
installable and only registered dataset versions. Starting a run asks for
confirmation, disables the button while the request is in flight, then polls
the run until it reaches a terminal state and stops. The run detail shows the
pipeline stages, the metrics, the tuning result when tuning ran, and the
approval-gate decision including the champion/challenger comparison.
