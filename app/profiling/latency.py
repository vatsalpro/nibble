"""
Latency timing and percentile statistics for Nibble.
Uses time.perf_counter_ns for high-precision nanosecond measurements.
Calculates Median, Mean, Trimmed Mean, P90, P95, P99, Min, Max, and mathematically consistent Throughput (FPS).
"""

import time
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from nibble.benchmark.metrics import (
    calculate_summary_stats,
    BenchmarkMetrics,
    calculate_fps
)


@dataclass
class LatencyStats:
    warmup_runs: int
    measured_runs: int
    total_time_ms: float
    median_ms: float
    mean_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    throughput_fps: float
    raw_latencies_ms: List[float]
    trimmed_mean_ms: float = 0.0
    batch_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "batch_size": self.batch_size,
            "total_time_ms": round(self.total_time_ms, 2),
            "median_ms": round(self.median_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "trimmed_mean_ms": round(self.trimmed_mean_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
            "throughput_fps": round(self.throughput_fps, 2),
            "raw_latencies_ms": [round(x, 4) for x in self.raw_latencies_ms]
        }


class LatencyTimer:
    """High-resolution monotonic timer for inference benchmarking."""

    def __init__(self):
        self._start_ns: int = 0
        self._stop_ns: int = 0
        self.recorded_samples: List[float] = []

    def start(self):
        self._start_ns = time.perf_counter_ns()

    def stop(self) -> float:
        self._stop_ns = time.perf_counter_ns()
        elapsed_ms = (self._stop_ns - self._start_ns) / 1_000_000.0
        self.recorded_samples.append(elapsed_ms)
        return elapsed_ms

    @contextmanager
    def time_scope(self):
        """Context manager for timing a block of code."""
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            stop = time.perf_counter_ns()
            elapsed_ms = (stop - start) / 1_000_000.0
            self.recorded_samples.append(elapsed_ms)

    def get_summary_stats(self, warmup_runs: int = 0, batch_size: int = 1) -> Dict[str, Any]:
        """Compute summary statistics for all samples recorded by this timer."""
        stats = self.calculate_stats(self.recorded_samples, warmup_runs=warmup_runs, batch_size=batch_size)
        return stats.to_dict()

    @staticmethod
    def calculate_stats(
        latencies_ms: List[float],
        warmup_runs: int = 0,
        batch_size: int = 1
    ) -> LatencyStats:
        """
        Compute statistics from measured latency samples.
        Throughput (FPS) is mathematically synchronized with median latency:
            FPS = (batch_size * 1000.0) / median_ms
        """
        m: BenchmarkMetrics = calculate_summary_stats(
            raw_latencies_ms=latencies_ms,
            warmup_runs=warmup_runs,
            batch_size=batch_size
        )
        return LatencyStats(
            warmup_runs=m.warmup_runs,
            measured_runs=m.measured_runs,
            total_time_ms=m.total_time_ms,
            median_ms=m.median_ms,
            mean_ms=m.mean_ms,
            trimmed_mean_ms=m.trimmed_mean_ms,
            p90_ms=m.p90_ms,
            p95_ms=m.p95_ms,
            p99_ms=m.p99_ms,
            min_ms=m.min_ms,
            max_ms=m.max_ms,
            std_dev_ms=m.std_dev_ms,
            throughput_fps=m.throughput_fps,
            raw_latencies_ms=m.raw_latencies_ms,
            batch_size=m.batch_size
        )
