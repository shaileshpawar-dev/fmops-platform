# New ML Project — the guided build workflow

## What it is

A ten-step path from a CSV to a deployed, monitored model:

```
01 Dataset   02 Profile   03 Target    04 Training   05 Evaluation
06 Registry  07 Approval  08 Deploy    09 Predict    10 Monitor
```

It is a **guided path over the platform's real APIs**, not a separate system.
There is no "project" entity: every step calls an endpoint the rest of the
console also uses, and everything it creates — the dataset version, the job,
the run, the model versions, the gate decisions, the deployment, the logged
predictions — is stored by those APIs and visible on their own pages.

## What each step actually calls

| Step | Endpoint | What comes back |
|---|---|---|
| 01 Dataset | `GET /api/v1/datasets/limits`, `POST /api/v1/datasets/upload` | version, rows, columns, validation report |
| 02 Profile | `GET /api/v1/automl/profile/{version}` | column roles, warnings, suggested target |
| 03 Target | `GET /api/v1/automl/profile/{version}?target=` | problem type re-derived for the chosen column, its classes |
| 04 Training | `GET /api/v1/models/algorithms`, `POST /api/v1/automl/runs` or `POST /api/v1/training/runs` | 202, a run id, a **job id** and the **model name** it will register under |
| 05 Evaluation | `GET .../runs/{id}` (polled), `GET /api/v1/jobs/{id}/logs` | per-candidate status, leaderboard, ranking rule, the job's log |
| 06 Registry | `GET /api/v1/models/{name}/versions/{v}`, `GET /api/v1/models/{name}/signature` | the registered version, its stage, its recorded input contract |
| 07 Approval | `GET /api/v1/models/{name}/decisions`, `POST .../approve`, `POST .../reject` | the recorded gate decision; a human sign-off where required |
| 08 Deploy | `GET /api/v1/deployments/current?model=`, `POST /api/v1/deployments`, `GET /api/v1/jobs/{id}` | a deployment job, followed to its end |
| 09 Predict | `GET /api/v1/models/{name}/signature`, `POST /api/v1/models/{name}/predict`, `POST /api/v1/feedback` | a scored response from the version that served it |
| 10 Monitor | `GET /api/v1/monitoring/summary?model=`, `/drift/latest?model=`, `/retraining/trigger/evaluate?model=` | that model's traffic, drift and trigger state |

## The model is named before it is trained

Step 03 asks for a **model name** and, optionally, the **positive class**.
Both routes in step 04 send them with the target column, so manual training
trains exactly the problem AutoML would have. Every later step reads the model
by that name — nothing assumes the platform has only one model.

- Left empty, the name is `<target>_classifier` (the reference dataset trained
  on its declared target keeps the reference model's name).
- A name keeps **one target and one pair of classes for life**. Training a
  different target under an existing name is refused before any work starts.
- Left on *automatic*, the positive class is `yes`/`true`/`1` when the target
  has such a pair, otherwise the rarer class. The rule used is recorded with
  the model and shown wherever its predictions are.

## Where progress lives

In `sessionStorage`, in the tab. Reloading mid-run resumes where you were;
closing the tab does not. Nothing about the walkthrough itself is stored
server-side — only what the real APIs store.

A step is reachable only when the thing it operates on exists: *Deploy* needs a
Production version, *Predict* needs a live one. Completed steps stay
revisitable.

## What it deliberately does not pretend

**Registration is not a button.** AutoML and a training run with promotion
requested register the result and put it through the gate *inside the run*.
Step 06 reports what the registry holds. For AutoML, every candidate that
trained is kept as a Development version (so each leaderboard row stays
traceable); only the winner is gated.

**Staging is not a way into production.** Approval into Staging checks the
absolute thresholds only. Approval into Production also compares the version
with the live one on held-out rows neither was trained on. Only a Production
version takes live traffic, so step 08 is locked for a Staging version and
step 07 offers *Promote to Production* instead. A Staging version can still be
*shadowed* from its model page once something is live.

**Where a human is required, a human decides.** With
`approval.require_manual_approval` on (the production profile), a run whose
checks all pass ends as `pending_manual`, shown as *AWAITING MANUAL APPROVAL*.
Approve re-runs the gate first — it cannot push a failing version through —
and the decision, actor and comment are recorded.

**There is no override.** The deployment request has no `force` field, and the
workflow's scripts never send one (a test asserts both).

## Execution

Training, AutoML and deployment run as **jobs**: rows in the platform database,
claimed by the API's in-process worker, with a streamed log, cooperative
cancellation and retry. See [jobs.md](jobs.md). The worker shares the API
container's CPU, so on a small task training slows requests while it runs. A
job interrupted by a restart is marked failed on recovery; it is never silently
resumed. Each AutoML candidate is a full training run.

## Problem types

**Binary classification only**, for the reasons in [automl.md](automl.md). A
target that infers to regression or multiclass is detected, named and refused
at step 03 — never trained anyway.
