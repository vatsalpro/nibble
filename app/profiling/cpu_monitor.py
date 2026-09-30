"""
CPU telemetry monitor for SnapForge.
Measures real-time CPU utilization, clock frequencies, and per-core loads.
"""

import os
import psutil
from typing import Dict, Any, List


class CPUMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        # prime psutil cpu percent
        psutil.cpu_percent(interval=None)

    def sample(self) -> Dict[str, Any]:
        """Sample current CPU metrics."""
        try:
            sys_cpu = psutil.cpu_percent(interval=None)
            proc_cpu = self.process.cpu_percent(interval=None)
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            freq = psutil.cpu_freq()
            freq_current = freq.current if freq else 0.0

            return {
                "system_util_pct": round(sys_cpu, 1),
                "process_util_pct": round(proc_cpu, 1),
                "frequency_mhz": round(freq_current, 1),
                "per_core_pct": [round(c, 1) for c in per_core],
                "status": "Available"
            }
        except Exception as e:
            return {
                "system_util_pct": 0.0,
                "process_util_pct": 0.0,
                "frequency_mhz": 0.0,
                "per_core_pct": [],
                "status": f"Error: {e}"
            }
