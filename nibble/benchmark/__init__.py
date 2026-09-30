"""
Nibble Benchmark Package.
Provides mathematically consistent latency, throughput, and comparison calculations.
"""

from nibble.benchmark.metrics import (
    calculate_fps,
    calculate_speedup,
    calculate_latency_reduction,
    calculate_percent_change,
    calculate_p95,
    calculate_trimmed_mean,
    calculate_summary_stats,
    BenchmarkMetrics
)
from nibble.benchmark.bundle import create_benchmark_bundle

__all__ = [
    "calculate_fps",
    "calculate_speedup",
    "calculate_latency_reduction",
    "calculate_percent_change",
    "calculate_p95",
    "calculate_trimmed_mean",
    "calculate_summary_stats",
    "BenchmarkMetrics",
    "create_benchmark_bundle",
]
