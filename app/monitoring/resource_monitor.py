"""System resource monitoring.

CPU, memory and disk come from psutil. GPU metrics are reported only when a GPU
is actually detected -- via ``nvidia-smi`` or pynvml if present. When there is no
GPU the platform reports ``gpu_available: false`` rather than emitting zeros,
because a dashboard full of 0% GPU is indistinguishable from an idle GPU.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.monitoring.metrics import get_metrics
from app.schemas.evaluation import ResourceUsage

logger = get_logger(__name__)


class ResourceMonitor:
    """Samples process and host resource usage."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._process = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._gpu_checked = False
        self._gpu_available = False

    @property
    def process(self):
        if self._process is None:
            import psutil

            self._process = psutil.Process()
            # Prime cpu_percent; the first call always returns 0.0.
            self._process.cpu_percent(interval=None)
        return self._process

    def sample(self) -> ResourceUsage:
        try:
            import psutil
        except ImportError:
            logger.warning("resource_monitor.psutil_missing")
            return ResourceUsage()

        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            process = self.process
            with process.oneshot():
                rss_mb = process.memory_info().rss / (1024 * 1024)
                cpu = process.cpu_percent(interval=None)
                threads = process.num_threads()
                try:
                    open_files = len(process.open_files())
                except (psutil.AccessDenied, OSError):
                    open_files = 0

            gpu = self._sample_gpu()
            usage = ResourceUsage(
                cpu_percent=round(float(cpu), 2),
                memory_percent=round(float(memory.percent), 2),
                memory_used_mb=round(memory.used / (1024 * 1024), 1),
                memory_total_mb=round(memory.total / (1024 * 1024), 1),
                disk_percent=round(float(disk.percent), 2),
                process_rss_mb=round(rss_mb, 1),
                open_files=open_files,
                threads=threads,
                gpu_available=bool(gpu),
                gpu=gpu,
            )
        except Exception as exc:
            logger.error("resource_monitor.sample_failed", extra={"error": str(exc)})
            return ResourceUsage()

        self._export(usage)
        return usage

    def _export(self, usage: ResourceUsage) -> None:
        metrics = get_metrics()
        metrics.cpu_percent.set(usage.cpu_percent)
        metrics.memory_percent.set(usage.memory_percent)
        metrics.process_memory_mb.set(usage.process_rss_mb)
        for device in usage.gpu:
            metrics.gpu_utilization.labels(device=str(device.get("index", 0))).set(
                float(device.get("utilization_percent", 0.0))
            )

    def _sample_gpu(self) -> list[dict[str, Any]]:
        """Query NVIDIA GPUs. Returns [] on any machine without one."""
        if self._gpu_checked and not self._gpu_available:
            return []

        smi = shutil.which("nvidia-smi")
        if smi is None:
            self._gpu_checked = True
            self._gpu_available = False
            return []

        try:
            proc = subprocess.run(  # noqa: S603
                [
                    smi,
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("resource_monitor.gpu_query_failed", extra={"error": str(exc)})
            self._gpu_checked = True
            self._gpu_available = False
            return []

        if proc.returncode != 0:
            self._gpu_checked = True
            self._gpu_available = False
            return []

        devices: list[dict[str, Any]] = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "utilization_percent": float(parts[2]),
                        "memory_used_mb": float(parts[3]),
                        "memory_total_mb": float(parts[4]),
                        "temperature_c": float(parts[5]),
                    }
                )
            except ValueError:
                continue

        self._gpu_checked = True
        self._gpu_available = bool(devices)
        return devices

    # -- background sampling -------------------------------------------------- #
    def start(self) -> None:
        """Sample on an interval in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        interval = max(5, self.settings.monitoring.resource_sample_seconds)

        def _loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.sample()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("resource_monitor.loop_error", extra={"error": str(exc)})

        self._thread = threading.Thread(
            target=_loop, name="fmops-resource-monitor", daemon=True
        )
        self._thread.start()
        logger.info("resource_monitor.started", extra={"interval_seconds": interval})

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("resource_monitor.stopped")


_MONITOR: ResourceMonitor | None = None


def get_resource_monitor() -> ResourceMonitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = ResourceMonitor()
    return _MONITOR


def sample_resources() -> ResourceUsage:
    return get_resource_monitor().sample()
