"""
Memory tracking for SnapForge.
Measures process RSS memory, peak memory, and system memory allocations.
"""

import os
import psutil
from typing import Dict, Any


class MemoryTracker:
    """Tracks memory usage of the running process and system."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_bytes = self.get_current_bytes()
        self.peak_bytes = self.baseline_bytes

    def get_current_bytes(self) -> int:
        try:
            return self.process.memory_info().rss
        except Exception:
            return 0

    def sample(self) -> float:
        """Sample current memory in MB and update peak."""
        current = self.get_current_bytes()
        if current > self.peak_bytes:
            self.peak_bytes = current
        return current / (1024 * 1024)

    def get_peak_mb(self) -> float:
        return round(self.peak_bytes / (1024 * 1024), 2)

    def get_delta_mb(self) -> float:
        delta = max(0, self.peak_bytes - self.baseline_bytes)
        return round(delta / (1024 * 1024), 2)

    def get_system_memory_info(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "available_gb": round(vm.available / (1024 ** 3), 2),
            "used_gb": round(vm.used / (1024 ** 3), 2),
            "percent": vm.percent,
            "process_rss_mb": round(self.get_current_bytes() / (1024 * 1024), 2),
            "process_peak_mb": self.get_peak_mb()
        }
