# Jobs and automation

Everything slow — training, AutoML, retraining, deployment rollouts and drift
scans — runs as a **job**. A request that starts one returns `202` with the job
id at once; the work happens on a worker, and its progress is observable from
anywhere through the jobs API or the console's **Jobs** page.

## What a job is

A row in the platform database (`jobs`) plus its log lines (`job_logs`):

| Field | Meaning |
|---|---|
| `kind` | `training` · `automl` · `retraining` · `deployment` · `drift_scan` |
| `status` | `queued` → `running` → `succeeded` / `failed` / `cancelled` |
| `resource_id` | the domain record it drives (run id, deployment id, retraining event) |
| `model_name` | the model it acts on |
| `requested_by` | the caller, or `automation` |
| `attempts`, `retry_of` | how many times it was claimed; a retry is a new job linked to the one it replaces |
| `idempotency_key` | a second submit with the same key returns the existing live job |
| `heartbeat_at`, `worker` | liveness of whoever is running it |
| `result`, `error` | what it produced, or why it failed |

The job's own log is every `app` log record emitted while it runs, captured by
a handler bound to that job, so the log shows the real pipeline stages rather
than a progress bar someone wrote.

## How it runs

- **Claiming is atomic.** A worker claims the oldest queued job inside
  `BEGIN IMMEDIATE`, and only while fewer than `jobs.max_running` jobs are
  running across every process sharing the database. Two workers cannot take
  the same job.
- **Heartbeats and recovery.** A running worker heartbeats every
  `jobs.heartbeat_seconds`. A running job whose worker has been silent for
  `jobs.stale_after_seconds` is failed, with its domain record, and the reason
  recorded. At startup, any run left without a job is reconciled the same way.
  Nothing is ever shown as progressing when nothing is running it.
- **Cancellation is cooperative.** `POST /api/v1/jobs/{id}/cancel` sets a flag;
  the running code checks it between stages (and between canary steps) and
  stops cleanly. A cancelled canary puts all traffic back on the live version.
- **Runtime limit.** A job running longer than `jobs.max_runtime_minutes` is
  asked to stop.
- **Retry.** `POST /api/v1/jobs/{id}/retry` on a failed or cancelled job queues
  a new attempt with the same inputs and a fresh domain record.

## Where it runs — and what that means

The worker is a thread inside the API process. That is the honest scope of this
implementation:

- On the deployed single Fargate task, **training competes with request
  handling** for the same CPU. `jobs.max_running` is 1 there.
- Jobs survive in the database across restarts, but a job **running** when the
  process dies is failed on recovery, not resumed.
- Because claims go through the database, more than one process can share the
  queue safely — but the SQLite database itself is local to one container, so
  scaling out needs a shared database first.

## Automation

When `automation.enabled` is on, the worker runs a pass every
`automation.interval_minutes`. The tick is claimed through the database, so
several processes never run it twice. For every model with a Production
version it:

1. queues a **drift scan** once enough new predictions have arrived since the
   last one (`drift.min_samples`);
2. evaluates the **retraining trigger** and queues a retraining job if it fires;
3. runs the **SLO watchdog** on the model's endpoint and raises alerts on a
   breach.

Automation submits with idempotency keys (`automation:drift:<model>`,
`automation:retrain:<model>`), so a slow job is never queued twice. Retraining
needs **new labelled data** since the last attempt — drift alone cannot make it
fire on the same labels again — so it cannot loop.

`GET /api/v1/automation` shows the configuration and the last tick;
`POST /api/v1/automation/run` runs a tick now.

## Endpoints

```
GET  /api/v1/jobs                 ?status= &kind= &model= &limit=
GET  /api/v1/jobs/{id}
GET  /api/v1/jobs/{id}/logs       ?after=<line id>   (stream by polling)
POST /api/v1/jobs/{id}/cancel
POST /api/v1/jobs/{id}/retry
GET  /api/v1/automation
POST /api/v1/automation/run
```
