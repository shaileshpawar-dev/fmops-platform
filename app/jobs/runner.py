"""Job execution.

Execution model
---------------
Jobs are rows in the operational database. Each API process runs one worker
thread that claims queued jobs and one heartbeat thread. That gives:

* **No duplicate execution.** A claim is a single ``BEGIN IMMEDIATE``
  transaction, so two processes (uvicorn workers, replicas sharing a volume)
  cannot take the same job.
* **Bounded load.** ``jobs.max_running`` is enforced globally at claim time. On
  a half-vCPU task, two AutoML runs at once would starve serving; excess work
  waits in ``queued`` rather than competing.
* **Honest liveness.** A running job's worker heartbeats. A job whose heartbeat
  goes stale -- its process died -- is failed and its domain record with it.
  Staleness, not "the process restarted", is the test: with more than one
  worker process, a restart must not fail work another process is doing.
* **Cooperative cancellation.** Cancelling a queued job is immediate. A running
  job stops at its next checkpoint (between AutoML candidates, between canary
  steps, around the training fit); it is never killed mid-write.
* **Per-job logs.** Every log record emitted while a job runs is also stored
  against that job, capped, so the console can show what the job did.

What this is not: a distributed queue. Jobs share the database, so they share
its host. Scaling training out means pointing ``execute`` at SageMaker or a
batch service; the job record, states and API stay the same.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_context, get_logger, log_context
from app.core.utils import jsonable
from app.jobs.store import (
    CANCELLED,
    FAILED,
    SUCCEEDED,
    WORKER_ID,
    JobStore,
    get_job_store,
)

logger = get_logger(__name__)


class JobCancelled(Exception):
    """Raised at a checkpoint when cancellation was requested."""


@dataclass
class JobContext:
    """What a handler gets: the job, and the ways to report back."""

    job: dict[str, Any]
    store: JobStore

    @property
    def id(self) -> str:
        return self.job["id"]

    @property
    def payload(self) -> dict[str, Any]:
        return self.job["payload"]

    def checkpoint(self) -> None:
        """Stop here if cancellation was requested."""
        if self.store.cancel_requested(self.id):
            raise JobCancelled(f"job {self.id} cancelled at a checkpoint")

    def link(self, resource_id: str) -> None:
        """Record which domain object this job is producing."""
        self.job["resource_id"] = resource_id
        self.store.set_resource(self.id, resource_id)

    def cancellable_sleep(self, seconds: float) -> None:
        """Sleep, waking every second to honour a cancellation request."""
        deadline = datetime.now(UTC) + timedelta(seconds=max(0.0, seconds))
        while True:
            self.checkpoint()
            remaining = (deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                return
            threading.Event().wait(min(1.0, remaining))


Handler = Callable[[JobContext], dict[str, Any] | None]
LostHandler = Callable[[dict[str, Any], str], None]

HANDLERS: dict[str, Handler] = {}
ON_LOST: dict[str, LostHandler] = {}


def handler(kind: str, on_lost: LostHandler | None = None) -> Callable[[Handler], Handler]:
    """Register the function that executes one kind of job."""

    def register(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        if on_lost is not None:
            ON_LOST[kind] = on_lost
        return fn

    return register


# --------------------------------------------------------------------------- #
# Log capture
# --------------------------------------------------------------------------- #
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)))
_WRITING = threading.local()


class JobLogHandler(logging.Handler):
    """Stores records emitted inside a job's log context against that job."""

    def __init__(self, store: JobStore, cap: int) -> None:
        super().__init__(level=logging.INFO)
        self.store = store
        self.cap = cap

    def emit(self, record: logging.LogRecord) -> None:
        job_id = get_context().get("job_id")
        if not job_id or getattr(_WRITING, "active", False):
            return
        _WRITING.active = True
        try:
            extras = {
                k: v
                for k, v in vars(record).items()
                if k not in _STANDARD_ATTRS and not k.startswith("_") and k != "message"
            }
            self.store.append_log(
                job_id,
                record.levelname,
                record.getMessage(),
                jsonable(extras),
                self.cap,
            )
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)
        finally:
            _WRITING.active = False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class JobRunner:
    def __init__(
        self, settings: Settings | None = None, store: JobStore | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_job_store()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._log_handler: JobLogHandler | None = None

    @property
    def config(self):
        return self.settings.jobs

    # -- lifecycle ------------------------------------------------------------ #
    def start(self) -> None:
        self._install_log_handler()
        self.recover()
        if self.config.inline or self._threads:
            return
        self._stop.clear()
        for target, name in ((self._work, "fmops-job-worker"), (self._beat, "fmops-job-beat")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info(
            "jobs.runner_started",
            extra={"worker": WORKER_ID, "max_running": self.config.max_running},
        )

    def threads_alive(self) -> bool:
        """Whether this process's worker and heartbeat threads are running."""
        return bool(self._threads) and all(t.is_alive() for t in self._threads)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()

    def _install_log_handler(self) -> None:
        app_logger = logging.getLogger("app")
        if self._log_handler is None:
            self._log_handler = JobLogHandler(self.store, self.config.max_log_lines)
        if self._log_handler not in app_logger.handlers:
            app_logger.addHandler(self._log_handler)
        # A job's log is part of the product, not console verbosity: it must
        # keep its INFO lines even when the console is set to WARNING. The
        # console handler filters by its own level, so stdout is unaffected.
        if app_logger.getEffectiveLevel() > logging.INFO:
            app_logger.setLevel(logging.INFO)

    # -- submission ------------------------------------------------------------ #
    def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        model_name: str | None = None,
        resource_id: str | None = None,
        requested_by: str | None = None,
        idempotency_key: str | None = None,
        retry_of: str | None = None,
    ) -> dict[str, Any]:
        if kind not in HANDLERS:
            raise ValueError(f"no handler registered for job kind {kind!r}")
        job = self.store.create(
            kind,
            payload,
            model_name=model_name,
            resource_id=resource_id,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            retry_of=retry_of,
        )
        logger.info(
            "jobs.submitted",
            extra={"job_id": job["id"], "kind": kind, "model": model_name},
        )
        if self.config.inline:
            # Tests and one-shot CLI use: run now, in this thread, but through
            # exactly the same claim/execute/finish path as a worker.
            claimed = self.store.claim_next(max_running=10_000)
            while claimed is not None:
                self.execute(claimed)
                claimed = self.store.claim_next(max_running=10_000)
        else:
            self._wake.set()
        return self.store.get(job["id"]) or job

    # -- execution --------------------------------------------------------------- #
    def execute(self, job: dict[str, Any]) -> None:
        self._install_log_handler()
        fn = HANDLERS.get(job["kind"])
        context = JobContext(job=job, store=self.store)
        with log_context(job_id=job["id"], job_kind=job["kind"], model=job.get("model_name")):
            if fn is None:
                self.store.finish(job["id"], FAILED, error=f"unknown job kind {job['kind']!r}")
                return
            logger.info("jobs.started", extra={"attempt": job.get("attempts")})
            try:
                result = fn(context)
            except JobCancelled as exc:
                logger.warning("jobs.cancelled", extra={"detail": str(exc)})
                self._lost(job, "cancelled while running")
                self.store.finish(job["id"], CANCELLED, error="cancelled while running")
            except Exception as exc:
                logger.exception("jobs.failed", extra={"error": str(exc)[:500]})
                self.store.finish(job["id"], FAILED, error=_error_text(exc))
            else:
                logger.info("jobs.succeeded")
                self.store.finish(job["id"], SUCCEEDED, result=jsonable(result or {}))

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.store.claim_next(self.config.max_running)
            except Exception as exc:
                logger.error("jobs.claim_failed", extra={"error": str(exc)})
                job = None
            if job is None:
                self._wake.wait(self.config.poll_seconds)
                self._wake.clear()
                continue
            self.execute(job)

    def _beat(self) -> None:
        from app.jobs.automation import claim_tick, run_tick

        while not self._stop.wait(self.config.heartbeat_seconds):
            try:
                self.store.heartbeat(WORKER_ID)
                self.recover()
                self._enforce_runtime_limit()
            except Exception as exc:  # the heartbeat must outlive a bad tick
                logger.error("jobs.heartbeat_failed", extra={"error": str(exc)})
            try:
                if claim_tick(self.settings):
                    run_tick(self.settings)
            except Exception as exc:
                logger.error("automation.tick_failed", extra={"error": str(exc)})

    # -- recovery ------------------------------------------------------------------ #
    def recover(self) -> int:
        """Fail jobs whose worker stopped heartbeating, and their domain records."""
        cutoff = datetime.now(UTC) - timedelta(seconds=self.config.stale_after_seconds)
        stale = self.store.stale(
            cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
        for job in stale:
            message = (
                f"worker {job.get('worker') or 'unknown'} stopped heartbeating while running "
                "this job (process restarted or crashed)"
            )
            self._lost(job, message)
            self.store.finish(job["id"], FAILED, error=message)
            logger.warning(
                "jobs.recovered_stale", extra={"job_id": job["id"], "kind": job["kind"]}
            )
        return len(stale)

    def _enforce_runtime_limit(self) -> None:
        limit = self.config.max_runtime_minutes
        if limit <= 0:
            return
        cutoff = datetime.now(UTC) - timedelta(minutes=limit)
        for job in self.store.overrunning(
            cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        ):
            self.store.request_cancel(job["id"])
            logger.warning(
                "jobs.runtime_limit_exceeded",
                extra={"job_id": job["id"], "limit_minutes": limit},
            )

    @staticmethod
    def _lost(job: dict[str, Any], reason: str) -> None:
        fn = ON_LOST.get(job["kind"])
        if fn is None:
            return
        try:
            fn(job, reason)
        except Exception as exc:
            logger.error("jobs.on_lost_failed", extra={"job_id": job["id"], "error": str(exc)})

    # -- retry ------------------------------------------------------------------------- #
    def retry(self, job_id: str, requested_by: str | None = None) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] not in (FAILED, CANCELLED):
            raise ValueError(
                f"only failed or cancelled jobs can be retried; this one is {job['status']}"
            )
        from app.jobs.handlers import prepare_retry

        payload, resource_id = prepare_retry(job)
        return self.submit(
            job["kind"],
            payload,
            model_name=job.get("model_name"),
            resource_id=resource_id,
            requested_by=requested_by,
            retry_of=job_id,
        )


def _error_text(exc: Exception) -> str:
    message = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
    return str(message)[:1000]


_RUNNER: JobRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_job_runner() -> JobRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                import app.jobs.handlers  # noqa: F401  (registers every handler)

                _RUNNER = JobRunner()
    return _RUNNER


def set_job_runner(runner: JobRunner | None) -> None:
    global _RUNNER
    _RUNNER = runner
