"""Profile the dashboard aggregation endpoint, layer by layer.

Attributes wall-clock time to the layers that could plausibly own it, so an
optimisation decision rests on a measurement rather than an intuition:

    database        SQL executed through app.core.db.Database
    mlflow          anything routed through the MLflow tracker/registry adapters
    filesystem      open(), scandir(), listdir(), stat(), glob()
    model loading   ModelCache._load (deserialise + materialise artifacts)
    serialization   json.dumps of the assembled payload
    aggregation     total minus the above -- the platform's own Python

Nested calls are counted once. ``Database.scalar()`` delegates to
``query_one()``, so naively wrapping both reports double the queries and
invents an N+1 that is not there; a re-entrancy depth guard prevents that.

Two modes make the before/after honest:

    --mode after    the shipped path: live performance computed once, shared
    --mode before   each section computes its own live performance, and
                    resource sampling includes open_files -- i.e. exactly the
                    behaviour that measured 3.2s, reachable from current code
                    because both are still supported fallbacks

Run:
    python scripts/profile_dashboard.py                  # after, summary
    python scripts/profile_dashboard.py --mode before
    python scripts/profile_dashboard.py --compare        # both, interleaved
    python scripts/profile_dashboard.py --sql --full
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import builtins
import cProfile
import io
import json
import os
import pstats
import time
from collections import Counter
from dataclasses import dataclass, field

LAYERS = ("database", "mlflow", "filesystem", "model_loading")


@dataclass
class Layer:
    seconds: float = 0.0
    calls: int = 0

    @property
    def ms(self) -> float:
        return self.seconds * 1000


@dataclass
class Profile:
    """One measured request, broken down by layer."""

    total_s: float = 0.0
    serialization_s: float = 0.0
    response_bytes: int = 0
    layers: dict[str, Layer] = field(default_factory=lambda: {n: Layer() for n in LAYERS})
    statements: Counter[str] = field(default_factory=Counter)
    files: Counter[str] = field(default_factory=Counter)
    mlflow_imported: bool = False

    @property
    def aggregation_s(self) -> float:
        """Whatever is left once the measurable layers are subtracted."""
        measured = sum(layer.seconds for layer in self.layers.values())
        return max(0.0, self.total_s - measured - self.serialization_s)

    def render(self, title: str) -> None:
        print(f"\n  {title}")
        print(f"  {'-' * 62}")
        rows = [
            ("total", self.total_s, None),
            ("  database", self.layers["database"].seconds, self.layers["database"].calls),
            ("  mlflow", self.layers["mlflow"].seconds, self.layers["mlflow"].calls),
            (
                "  filesystem",
                self.layers["filesystem"].seconds,
                self.layers["filesystem"].calls,
            ),
            (
                "  model loading",
                self.layers["model_loading"].seconds,
                self.layers["model_loading"].calls,
            ),
            ("  aggregation", self.aggregation_s, None),
            ("  serialization", self.serialization_s, None),
        ]
        for name, seconds, calls in rows:
            share = (seconds / self.total_s * 100) if self.total_s else 0.0
            suffix = f"   {calls} calls" if calls is not None else ""
            print(f"  {name:<18}{seconds:8.4f} s{share:7.1f}%{suffix}")
        print(f"  {'response':<18}{self.response_bytes / 1024:8.1f} KB")


class Instrument:
    """Patches the boundaries of each layer for the duration of a block."""

    def __init__(self, profile: Profile) -> None:
        self.p = profile
        self._undo: list[tuple[object, str, object]] = []
        self._depth = 0  # re-entrancy guard: only the outermost call is timed

    # -- helpers ------------------------------------------------------------- #
    def _patch(self, obj: object, name: str, factory) -> None:
        try:
            original = getattr(obj, name)
        except AttributeError:
            return
        self._undo.append((obj, name, original))
        setattr(obj, name, factory(original))

    def _timed(self, layer: str):
        """Wrap a callable so its wall time lands in ``layer``, counted once."""

        def factory(original):
            def wrapper(*args, **kwargs):
                if self._depth:  # nested inside another instrumented call
                    return original(*args, **kwargs)
                self._depth += 1
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self._depth -= 1
                    entry = self.p.layers[layer]
                    entry.seconds += time.perf_counter() - started
                    entry.calls += 1

            return wrapper

        return factory

    # -- lifecycle ----------------------------------------------------------- #
    def __enter__(self) -> Instrument:
        from app.core.db import Database, get_database

        # Statement count comes from sqlite itself, which fires once per
        # statement actually executed -- the only number that can prove or
        # disprove an N+1.
        connection = get_database().connection

        def on_statement(statement: str) -> None:
            # Keep the whole statement. Truncating here once made three
            # different COUNT(*) queries look like one query repeated three
            # times, because the clause that distinguishes them sits past the
            # cut. A truncated key manufactures N+1s that do not exist.
            self.p.statements[" ".join(str(statement).split())] += 1

        connection.set_trace_callback(on_statement)
        self._undo.append((connection, "__trace__", None))

        for method in ("execute", "executemany", "query", "query_one", "scalar"):
            self._patch(Database, method, self._timed("database"))

        from app.deployment.model_cache import ModelCache

        self._patch(ModelCache, "_load", self._timed("model_loading"))

        self.p.mlflow_imported = "mlflow" in sys.modules
        if self.p.mlflow_imported:
            import mlflow

            for method in ("search_runs", "get_run", "search_experiments"):
                self._patch(mlflow, method, self._timed("mlflow"))
            client = getattr(mlflow, "MlflowClient", None)
            if client is not None:
                for method in ("search_runs", "get_run", "search_registered_models"):
                    self._patch(client, method, self._timed("mlflow"))

        fs = self._timed("filesystem")

        def open_factory(original):
            timed = fs(original)

            def wrapper(file, *args, **kwargs):
                self.p.files[str(file)[-68:]] += 1
                return timed(file, *args, **kwargs)

            return wrapper

        self._patch(builtins, "open", open_factory)
        for name in ("scandir", "listdir", "stat"):
            self._patch(os, name, fs)
        return self

    def __exit__(self, *exc) -> bool:
        for obj, name, original in reversed(self._undo):
            if name == "__trace__":
                obj.set_trace_callback(None)
            else:
                setattr(obj, name, original)
        self._undo.clear()
        return False


# ---------------------------------------------------------------------------- #
# the two request shapes
# ---------------------------------------------------------------------------- #
def build_after() -> dict:
    """The shipped path: live performance computed once and shared."""
    from app.api.routes.dashboard import dashboard_data

    return dashboard_data()


def build_before() -> dict:
    """The pre-fix path, assembled from the fallbacks that still exist.

    ``_system_section()`` and ``_retraining_section()`` both compute their own
    live performance when none is handed to them -- which is precisely what
    they used to do. Calling them that way reproduces the duplicated work
    without reverting the commit.
    """
    from app.api.routes import dashboard as d

    payload: dict = {"service": {}}
    payload["model"] = d._safe(d._model_section, "model")
    payload["deployment"] = d._safe(d._deployment_section, "deployment")
    payload["drift"] = d._safe(d._drift_section, "drift")
    payload["system"] = d._safe(d._system_section, "system")  # computes its own
    payload["llm"] = d._safe(d._llm_section, "llm")
    payload["retraining"] = d._safe(d._retraining_section, "retraining")  # and again
    payload["alerts"] = d._safe(d._alerts_section, "alerts")
    return payload


def measure_once(build) -> Profile:
    profile = Profile()
    with Instrument(profile):
        started = time.perf_counter()
        payload = build()
        profile.total_s = time.perf_counter() - started

        started = time.perf_counter()
        body = json.dumps(payload, default=str)
        profile.serialization_s = time.perf_counter() - started
        profile.response_bytes = len(body.encode())
    # Serialization is measured inside the instrumented block but is not part
    # of the request timer, so add it to the total the report divides by.
    profile.total_s += profile.serialization_s
    return profile


def timings(build, n: int) -> list[float]:
    out = []
    for _ in range(n):
        started = time.perf_counter()
        build()
        out.append((time.perf_counter() - started) * 1000)
    return out


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# ---------------------------------------------------------------------------- #
def warm() -> None:
    """Reach steady state: model in cache, lazy imports done, sampler primed."""
    from app.monitoring.resource_monitor import get_resource_monitor

    get_resource_monitor().sample()
    for _ in range(3):
        build_after()


def section_breakdown() -> None:
    from app.api.routes import dashboard as d

    print("\n  per-section (steady state, shipped path):")
    total = 0.0
    live = d._live_performance()
    for name, fn in (
        ("model", d._model_section),
        ("deployment", d._deployment_section),
        ("drift", d._drift_section),
        ("system", lambda: d._system_section(live)),
        ("llm", d._llm_section),
        ("retraining", lambda: d._retraining_section(live)),
        ("alerts", d._alerts_section),
    ):
        started = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - started) * 1000
        total += elapsed
        print(f"      {name:<13}{elapsed:8.1f} ms")
    print(f"      {'TOTAL':<13}{total:8.1f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile /api/v1/dashboard")
    parser.add_argument("--mode", choices=("before", "after"), default="after")
    parser.add_argument("--compare", action="store_true", help="interleaved before/after A/B")
    parser.add_argument("--full", action="store_true", help="print the full cProfile table")
    parser.add_argument("--sql", action="store_true", help="print every SQL statement")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args(argv)

    from app.core.logging import configure_from_settings

    configure_from_settings(force=True)
    import logging

    logging.getLogger().setLevel(logging.ERROR)

    print("=" * 72)
    print("  DASHBOARD PROFILE")
    print("=" * 72)
    warm()

    if args.compare:
        # Interleave so a machine that gets busier partway through penalises
        # both arms equally instead of whichever ran second.
        before, after = [], []
        for _ in range(args.runs):
            started = time.perf_counter()
            build_before()
            before.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            build_after()
            after.append((time.perf_counter() - started) * 1000)
        b, a = median(before), median(after)
        print(f"\n  interleaved A/B over {args.runs} pairs (median ms):")
        print(f"      before  {b:8.1f}")
        print(f"      after   {a:8.1f}")
        print(f"      delta   {b - a:8.1f}  ({(b - a) / b * 100:.1f}% faster)")

        measure_once(build_before).render("BEFORE (live performance computed twice)")
        measure_once(build_after).render("AFTER (computed once, shared)")
        return 0

    build = build_before if args.mode == "before" else build_after

    runs = timings(build, args.runs)
    ordered = sorted(runs)
    print(f"\n  wall clock over {len(runs)} runs, mode={args.mode} (ms):")
    print(f"      min {ordered[0]:8.1f}   median {median(runs):8.1f}   max {ordered[-1]:8.1f}")

    profile = measure_once(build)
    profile.render(f"layer breakdown (mode={args.mode})")

    print(f"\n  SQL statements executed: {sum(profile.statements.values())}")
    repeated = sorted(((n, s) for s, n in profile.statements.items() if n > 1), reverse=True)
    if repeated:
        print("  repeated statements (candidate N+1):")
        for count, statement in repeated[:10]:
            print(f"      {count:3d}x  {statement[:150]}")
    else:
        print("  no statement executed more than once -- no N+1")
    if args.sql:
        print("\n  every statement:")
        for statement, count in profile.statements.most_common():
            print(f"      {count:3d}x  {statement[:150]}")

    if profile.files:
        print(f"\n  filesystem opens: {sum(profile.files.values())}")
        for path, count in profile.files.most_common(10):
            print(f"      {count:3d}x  {path}")
    else:
        print("\n  no filesystem reads on the request path")
    print(f"  mlflow imported in this process: {'yes' if profile.mlflow_imported else 'no'}")

    if args.mode == "after":
        section_breakdown()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(3):
        build()
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(
        30 if args.full else 15
    )
    print("\n  cProfile (cumulative, 3 runs):")
    for line in stream.getvalue().splitlines():
        if line.strip():
            print(f"   {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
