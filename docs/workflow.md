# New ML Project — the guided build workflow

## What it is

A ten-step path through the platform for someone who has not used it before:

```
01 Dataset   02 Profile   03 Target    04 Training   05 Evaluation
06 Registry  07 Approval  08 Deploy    09 Predict    10 Monitor
```

It is **entirely frontend**. There is no project entity, no new persistence and
no orchestration layer — every step calls an endpoint the console already used,
and the only backend change it required was an endpoint that reports the upload
limit so the page could stop carrying its own copy of the number.

## What each step actually calls

| Step | Endpoint | What comes back |
|---|---|---|
| 01 Dataset | `GET /api/v1/datasets/limits`, `POST /api/v1/datasets/upload` | version, rows, columns, validation report |
| 02 Profile | `GET /api/v1/automl/profile/{version}` | column roles, warnings, suggested target |
| 03 Target | `GET /api/v1/automl/profile/{version}?target=` | problem type re-derived for the chosen column |
| 04 Training | `GET /api/v1/models/algorithms`, `POST /api/v1/automl/runs` or `POST /api/v1/training/runs` | 202 and a run id |
| 05 Evaluation | `GET .../runs/{id}` (polled) | per-candidate status, leaderboard, ranking rule |
| 06 Registry | `GET /api/v1/models`, `GET /api/v1/models/{name}/versions` | registered version and stage |
| 07 Approval | `POST /api/v1/models/{name}/versions/{v}/evaluate-gate` | checks, thresholds, champion comparison |
| 08 Deploy | `GET /api/v1/deployments/current`, `POST /api/v1/deployments` | rollout result |
| 09 Predict | `GET /openapi.json`, `GET /api/v1/datasets/{v}/preview`, `POST /api/v1/predict` | a real scored response |
| 10 Monitor | `GET /api/v1/monitoring/summary`, `/deployments/current`, `/drift/latest` | live platform state |

## Where progress lives

In `sessionStorage`, in the tab. Reloading mid-run resumes where you were;
closing the tab does not. Nothing about the walkthrough is stored server-side —
only what the real APIs store: dataset versions, runs, model versions,
deployments, audit entries.

A step is reachable only when the thing it operates on exists. The guard is
derived from that state rather than from a step counter, so there is no way to
land on *Deploy* without a registered version, and completed steps stay
revisitable.

## Three things it deliberately does not pretend

**Registration is not a button.** Both AutoML and a training run with promotion
requested register the winner and put it through the gate *inside the run*.
There is no endpoint that registers a finished run after the fact, so step 06
reports what the registry holds rather than offering an action that does not
exist. A manual run started without promotion says so, and offers to run again.

**The model name is global.** Every model is registered under the one
configured name (`tracking.registered_model_name`) and distinguished by
version. There is no per-dataset namespace, and the registry step says so
rather than implying each walkthrough produces its own model.

**Online scoring is schema-bound.** `POST /api/v1/predict` validates against a
single fixed request schema belonging to the reference model. Step 09 reads
that schema from the served OpenAPI document and compares it to the uploaded
columns: when they line up it prefills a form from a real row of the dataset,
and when they do not it names the missing fields and refuses, rather than
rendering a form that would 422. The training path is general; the serving
contract is not.

## The gate is not bypassable from here

The deployment API accepts `force`, which deploys a version that has not
cleared the approval gate. **The workflow never sends it.** When the gate fails,
step 08 renders a locked state explaining why and points at the registry; there
is no override in this path, and a test asserts the parameter appears nowhere in
the module.

Thresholds shown in step 07 come from platform configuration. They are read,
never adjusted to make a candidate pass.

## Cost and execution

Runs execute as FastAPI background tasks in the API process — the same
mechanism the retraining endpoint uses. On a single Fargate task, training
competes with request handling for the same CPU, and a run in flight is lost if
the process restarts (orphans are reconciled to `failed`). Each AutoML
candidate is a full training run, so *n* candidates cost roughly *n* times one
run; the step states how many will be trained before it starts.

## Problem types

Binary classification only, for the reasons in [automl.md](automl.md). A target
that infers to regression or multiclass is detected, named, and refused at step
03 with the reason — it is never cast to an integer and trained anyway.
