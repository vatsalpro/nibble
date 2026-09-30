"""
Unified Profiler for SnapForge.
Collects time-series samples of CPU, RAM, GPU, and NPU during benchmark runs or background monitoring.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from app.profiling.cpu_monitor import CPUMonitor
from app.profiling.memory import MemoryTracker
from app.profiling.gpu_monitor import GPUMonitor
from app.profiling.npu_monitor import NPUMonitor


class SystemProfiler:
    """Collects system-level telemetry periodically."""

    def __init__(self, interval_sec: float = 0.2):
        self.interval_sec = interval_sec
        self.cpu_monitor = CPUMonitor()
        self.memory_tracker = MemoryTracker()
        self.gpu_monitor = GPUMonitor()
        self.npu_monitor = NPUMonitor()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def start(self):
        """Start background profiling thread."""
        with self._lock:
            self._samples.clear()
            self._running = True
            self.memory_tracker = MemoryTracker()  # reset baseline
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return aggregated stats."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        return self.get_summary()

    def _monitor_loop(self):
        while True:
            with self._lock:
                if not self._running:
                    break

            sample = self.sample_now()
            with self._lock:
                self._samples.append(sample)

            time.sleep(self.interval_sec)

    def sample_now(self) -> Dict[str, Any]:
        """Take a single snapshot of all telemetry."""
        cpu = self.cpu_monitor.sample()
        mem = self.memory_tracker.get_system_memory_info()
        gpu = self.gpu_monitor.sample()
        npu = self.npu_monitor.sample()

        return {
            "timestamp": time.time(),
            "cpu_util_pct": cpu["system_util_pct"],
            "process_cpu_pct": cpu["process_util_pct"],
            "cpu_freq_mhz": cpu["frequency_mhz"],
            "ram_used_gb": mem["used_gb"],
            "ram_total_gb": mem["total_gb"],
            "ram_pct": mem["percent"],
            "process_rss_mb": mem["process_rss_mb"],
            "gpu_status": gpu["status"],
            "gpu_util_pct": gpu.get("util_pct"),
            "npu_status": npu["status"],
            "npu_device": npu.get("device_name"),
            "npu_details": npu.get("details")
        }

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate samples collected during the run."""
        with self._lock:
            samples = list(self._samples)

        if not samples:
            snap = self.sample_now()
            return {
                "sample_count": 0,
                "avg_cpu_pct": snap["cpu_util_pct"],
                "max_cpu_pct": snap["cpu_util_pct"],
                "peak_process_rss_mb": snap["process_rss_mb"],
                "gpu_status": snap["gpu_status"],
                "npu_status": snap["npu_status"],
                "npu_details": snap["npu_details"],
                "samples": []
            }

        cpu_vals = [s["cpu_util_pct"] for s in samples]
        proc_rss_vals = [s["process_rss_mb"] for s in samples]

        return {
            "sample_count": len(samples),
            "avg_cpu_pct": round(sum(cpu_vals) / len(cpu_vals), 1),
            "max_cpu_pct": round(max(cpu_vals), 1),
            "peak_process_rss_mb": round(max(proc_rss_vals), 1),
            "gpu_status": samples[-1]["gpu_status"],
            "npu_status": samples[-1]["npu_status"],
            "npu_details": samples[-1]["npu_details"],
            "samples": samples
        }
