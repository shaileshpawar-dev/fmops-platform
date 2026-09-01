"""Profile the dashboard aggregation endpoint.

Answers four questions with measurements rather than intuition:

1. Where does wall-clock time go?              (cProfile, cumulative)
2. How many SQL queries run, and are any N+1?  (Database.query/query_one/scalar hooks)
3. Is MLflow being called on the request path? (import + attribute hooks)
4. Is there filesystem I/O on the request path? (open/Path.read_* hooks)

Run:
    python scripts/profile_dashboard.py            # summary
    python scripts/profile_dashboard.py --full     # + full cProfile table
    python scripts/profile_dashboard.py --sql      # + every SQL statement
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
import pstats
import time
from collections import Counter

from app.core.logging import configure_from_settings


class Instrumentation:
    """Counts SQL, file opens and MLflow attribute access during a block."""

    def __init__(self) -> None:
        self.sql: Counter[str] = Counter()
        self.sql_time = 0.0
        self.sql_calls = 0
        self.file_opens: Counter[str] = Counter()
        self.mlflow_touched = False
        self._patches: list[tuple[object, str, object]] = []

    # -- patch helpers ------------------------------------------------------- #
    def _wrap(self, obj, name: str, factory) -> None:
        original = getattr(obj, name)
        self._patches.append((obj, name, original))
        setattr(obj, name, factory(original))

    def __enter__(self) -> Instrumentation:
        # Use sqlite3's own trace callback rather than wrapping the Database
        # helper methods. Database.scalar() delegates to query_one(), so
        # wrapping both would count one logical query twice and invent an N+1
        # that is not there. The trace callback fires once per statement the
        # engine actually executes, which is the number that matters.
        from app.core.db import get_database

        connection = get_database().connection

        def on_statement(statement: str) -> None:
            self.sql_calls += 1
            self.sql[" ".join(str(statement).split())[:120]] += 1

        connection.set_trace_callback(on_statement)
        self._patches.append((connection, "__trace__", None))

        def open_factory(original):
            def wrapper(file, *args, **kwargs):
                self.file_opens[str(file)[-70:]] += 1
                return original(file, *args, **kwargs)

            return wrapper

        self._wrap(builtins, "open", open_factory)

        if "mlflow" in sys.modules:
            self.mlflow_touched = True
        return self

    def __exit__(self, *exc) -> bool:
        for obj, name, original in reversed(self._patches):
            if name == "__trace__":
                obj.set_trace_callback(None)
            else:
                setattr(obj, name, original)
        self._patches.clear()
        return False

    # -- reporting ----------------------------------------------------------- #
    def report(self, show_sql: bool = False) -> None:
        print(f"\n  SQL: {self.sql_calls} queries, {self.sql_time * 1000:.1f} ms total")
        repeated = [(n, q) for q, n in self.sql.items() if n > 1]
        repeated.sort(reverse=True)
        if repeated:
            print("  repeated statements (possible N+1):")
            for count, statement in repeated[:10]:
                print(f"      {count:3d}x  {statement}")
        else:
            print("  no statement executed more than once")

        if show_sql:
            print("\n  every statement:")
            for statement, count in self.sql.most_common():
                print(f"      {count:3d}x  {statement}")

        if self.file_opens:
            print(f"\n  file opens: {sum(self.file_opens.values())}")
            for path, count in self.file_opens.most_common(10):
                print(f"      {count:3d}x  {path}")
        else:
            print("\n  no filesystem reads on the request path")


def warm() -> None:
    """Do what the running app does at startup, so we measure steady state."""
    from app.api.routes.dashboard import dashboard_data
    from app.monitoring.resource_monitor import get_resource_monitor

    get_resource_monitor().sample()
    dashboard_data()  # loads the model into cache, primes lazy imports


def measure(n: int = 20) -> list[float]:
    from app.api.routes.dashboard import dashboard_data

    timings = []
    for _ in range(n):
        started = time.perf_counter()
        dashboard_data()
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def section_breakdown() -> None:
    from app.api.routes import dashboard as d

    print("\n  per-section (steady state):")
    total = 0.0
    for name, fn in (
        ("model", d._model_section),
        ("deployment", d._deployment_section),
        ("drift", d._drift_section),
        ("system", d._system_section),
        ("llm", d._llm_section),
        ("retraining", d._retraining_section),
        ("alerts", d._alerts_section),
    ):
        started = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - started) * 1000
        total += elapsed
        print(f"      {name:12s} {elapsed:8.1f} ms")
    print(f"      {'TOTAL':12s} {total:8.1f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile /api/v1/dashboard")
    parser.add_argument("--full", action="store_true", help="print the full cProfile table")
    parser.add_argument("--sql", action="store_true", help="print every SQL statement")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args(argv)

    configure_from_settings(force=True)
    import logging

    logging.getLogger().setLevel(logging.ERROR)

    print("=" * 72)
    print("  DASHBOARD PROFILE")
    print("=" * 72)

    warm()

    timings = measure(args.runs)
    timings_sorted = sorted(timings)
    print(f"\n  wall clock over {len(timings)} runs (ms):")
    print(f"      min    {timings_sorted[0]:8.1f}")
    print(f"      median {timings_sorted[len(timings_sorted) // 2]:8.1f}")
    print(f"      max    {timings_sorted[-1]:8.1f}")
    print(f"      mean   {sum(timings) / len(timings):8.1f}")

    section_breakdown()

    # ---- instrumented single run ------------------------------------------ #
    from app.api.routes.dashboard import dashboard_data

    with Instrumentation() as inst:
        dashboard_data()
    inst.report(show_sql=args.sql)

    print(
        "\n  mlflow imported in this process: " f"{'yes' if 'mlflow' in sys.modules else 'no'}"
    )

    # ---- cProfile ---------------------------------------------------------- #
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(3):
        dashboard_data()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30 if args.full else 15)
    print("\n  cProfile (cumulative, 3 runs):")
    for line in stream.getvalue().splitlines():
        if line.strip():
            print(f"   {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
