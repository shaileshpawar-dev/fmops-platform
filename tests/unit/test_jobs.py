"""The job runner: claiming, limits, cancellation, recovery, logs, retry."""

from __future__ import annotations

import logging
import threading

import pytest

from app.core.config import JobsConfig
from app.core.db import Database
from app.jobs.runner import JobContext, JobRunner, handler
from app.jobs.store import CANCELLED, FAILED, QUEUED, RUNNING, SUCCEEDED, JobStore

LOST: list[tuple[str, str]] = []


def _lost(job, reason):
    LOST.append((job["id"], reason))


@handler("test_ok", on_lost=_lost)
def _ok(ctx: JobContext):
    logging.getLogger("app.tests.jobs").info("step.done", extra={"n": ctx.payload.get("n")})
    return {"echo": ctx.payload.get("n")}


@handler("test_fail", on_lost=_lost)
def _fail(ctx: JobContext):
    raise RuntimeError("the thing broke")


@handler("test_cancel", on_lost=_lost)
def _cancel(ctx: JobContext):
    # Simulates a request arriving mid-run, then a checkpoint.
    ctx.store.request_cancel(ctx.id)
    ctx.checkpoint()
    return {"unreachable": True}


@handler("test_chatty", on_lost=_lost)
def _chatty(ctx: JobContext):
    log = logging.getLogger("app.tests.jobs")
    for i in range(50):
        log.info("line", extra={"i": i})
    return {}


@pytest.fixture
def store(tmp_path):
    return JobStore(Database(tmp_path / "jobs.db"))


@pytest.fixture
def runner(settings, store):
    LOST.clear()
    cfg = settings.model_copy(
        update={"jobs": JobsConfig(inline=False, max_running=1, max_log_lines=20)}
    )
    return JobRunner(cfg, store)


def test_a_job_runs_to_success_and_keeps_its_result(runner, store):
    job = store.create("test_ok", {"n": 7})
    runner.execute(store.claim_next(1))
    done = store.get(job["id"])
    assert done["status"] == SUCCEEDED
    assert done["result"] == {"echo": 7}
    assert done["attempts"] == 1


def test_a_raising_handler_fails_the_job_with_its_message(runner, store):
    job = store.create("test_fail", {})
    runner.execute(store.claim_next(1))
    done = store.get(job["id"])
    assert done["status"] == FAILED
    assert "the thing broke" in done["error"]


def test_a_claim_is_exclusive_across_connections(tmp_path):
    """Two processes -- two connections to one file -- never take the same job."""
    path = tmp_path / "shared.db"
    first, second = JobStore(Database(path)), JobStore(Database(path))
    for _ in range(20):
        first.create("test_ok", {})
    claimed: list[str] = []
    lock = threading.Lock()

    def drain(s: JobStore) -> None:
        while True:
            job = s.claim_next(max_running=1000)
            if job is None:
                return
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=drain, args=(s,)) for s in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == 20
    assert len(set(claimed)) == 20, "a job was claimed twice"


def test_the_running_cap_holds_back_excess_work(store):
    store.create("test_ok", {})
    store.create("test_ok", {})
    assert store.claim_next(max_running=1) is not None
    assert store.claim_next(max_running=1) is None, "cap of one exceeded"
    assert store.counts()[QUEUED] == 1


def test_cancelling_a_queued_job_is_immediate(runner, store):
    job = store.create("test_ok", {})
    store.request_cancel(job["id"])
    assert store.get(job["id"])["status"] == CANCELLED
    assert store.claim_next(1) is None, "a cancelled job must never start"


def test_a_running_job_stops_at_its_checkpoint(runner, store):
    job = store.create("test_cancel", {})
    runner.execute(store.claim_next(1))
    done = store.get(job["id"])
    assert done["status"] == CANCELLED
    assert done["result"] is None
    assert (job["id"], "cancelled while running") in LOST


def test_a_job_whose_worker_stopped_heartbeating_is_failed_with_its_record(runner, store):
    job = store.create("test_ok", {})
    store.claim_next(1)
    store.db.execute(
        "UPDATE jobs SET heartbeat_at = '2000-01-01T00:00:00.000Z' WHERE id = ?", (job["id"],)
    )
    assert runner.recover() == 1
    done = store.get(job["id"])
    assert done["status"] == FAILED
    assert "stopped heartbeating" in done["error"]
    assert LOST and LOST[-1][0] == job["id"]


def test_a_fresh_heartbeat_is_not_recovered(runner, store):
    job = store.create("test_ok", {})
    store.claim_next(1)
    assert runner.recover() == 0
    assert store.get(job["id"])["status"] == RUNNING


def test_an_idempotency_key_yields_one_job_until_it_fails(runner, store):
    a = store.create("test_fail", {}, idempotency_key="k1")
    b = store.create("test_fail", {}, idempotency_key="k1")
    assert a["id"] == b["id"]
    runner.execute(store.claim_next(1))
    c = store.create("test_fail", {}, idempotency_key="k1")
    assert c["id"] != a["id"], "a failed job's key must be reusable"


def test_logs_are_captured_per_job_and_capped(runner, store):
    job = store.create("test_chatty", {})
    runner.execute(store.claim_next(1))
    lines = store.logs(job["id"], limit=1000)
    assert lines, "nothing was captured"
    assert len(lines) <= 22  # the cap, plus the job's own start/finish lines
    assert any("capped" in line["message"] for line in lines)


def test_log_lines_carry_structured_context(runner, store):
    job = store.create("test_ok", {"n": 3})
    runner.execute(store.claim_next(1))
    step = next(line for line in store.logs(job["id"]) if line["message"] == "step.done")
    assert step["context"].get("n") == 3


def test_retry_queues_a_new_job_and_keeps_the_failed_one(runner, store, monkeypatch):
    job = store.create("test_fail", {"n": 1})
    runner.execute(store.claim_next(1))
    monkeypatch.setattr(runner.config, "inline", False)
    new = runner.retry(job["id"])
    assert new["id"] != job["id"]
    assert new["retry_of"] == job["id"]
    assert new["status"] == QUEUED
    assert store.get(job["id"])["status"] == FAILED


def test_a_running_job_cannot_be_retried(runner, store):
    job = store.create("test_ok", {})
    store.claim_next(1)
    with pytest.raises(ValueError, match="only failed or cancelled"):
        runner.retry(job["id"])
