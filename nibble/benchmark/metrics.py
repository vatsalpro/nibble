"""
Nibble Benchmark Metrics Engine.
Ensures mathematically sound, consistent calculations for inference latency,
percentiles, throughput (FPS), speedup factor, and latency delta.

Key Invariant:
Throughput (FPS) for single-sample inference is defined relative to the primary
latency metric (median latency) to prevent contradictory reporting where a model
with a higher median latency is reported as having higher throughput.
    FPS = (batch_size * 1000.0) / median_latency_ms
"""

import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


def calculate_fps(latency_ms: float, batch_size: int = 1) -> float:
    """
    Calculate throughput in frames/inferences per second.
    FPS = (batch_size * 1000.0) / latency_ms
    """
    if latency_ms <= 0.0:
        return 0.0
    safe_batch = max(1, batch_size)
    return round((safe_batch * 1000.0) / latency_ms, 2)


def calculate_speedup(baseline_latency_ms: float, optimized_latency_ms: float) -> float:
    """
    Calculate speedup ratio: baseline_latency / optimized_latency.
    Example: 10 ms -> 5 ms = 2.0x speedup.
    Example: 10 ms -> 12 ms = 0.83x speedup.
    """
    if baseline_latency_ms <= 0.0 or optimized_latency_ms <= 0.0:
        return 1.0
    return round(baseline_latency_ms / optimized_latency_ms, 2)


def calculate_latency_reduction(baseline_latency_ms: float, optimized_latency_ms: float) -> float:
    """
    Calculate percentage reduction in latency.
    Positive value indicates improvement (faster).
    Negative value indicates regression (slower).
    Example: 10 ms -> 5 ms = +50.0% reduction.
    Example: 10 ms -> 12 ms = -20.0% reduction.
    """
    if baseline_latency_ms <= 0.0:
        return 0.0
    reduction = ((baseline_latency_ms - optimized_latency_ms) / baseline_latency_ms) * 100.0
    return round(reduction, 1)


def calculate_percent_change(baseline: float, current: float) -> float:
    """
    Calculate standard relative percent change: ((current - baseline) / |baseline|) * 100.
    """
    if baseline == 0.0:
        return 0.0 if current == 0.0 else 100.0
    return round(((current - baseline) / abs(baseline)) * 100.0, 1)


def calculate_percentile(samples: List[float], percentile: float) -> float:
    """
    Calculate arbitrary percentile (0-100) using linear interpolation.
    """
    if not samples:
        return 0.0
    if len(samples) == 1:
        return round(samples[0], 4)
    if percentile <= 0.0:
        return round(min(samples), 4)
    if percentile >= 100.0:
        return round(max(samples), 4)

    s = sorted(samples)
    n = len(s)
    k = (n - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(s[int(k)], 4)
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return round(d0 + d1, 4)


def calculate_p95(samples: List[float]) -> float:
    """Calculate 95th percentile latency."""
    return calculate_percentile(samples, 95.0)


def calculate_trimmed_mean(samples: List[float], trim_ratio: float = 0.1) -> float:
    """
    Calculate trimmed mean by discarding trim_ratio fraction of extreme samples from both tails.
    Default trim_ratio=0.1 trims 10% lowest and 10% highest for an outlier-resistant metric.
    """
    if not samples:
        return 0.0
    n = len(samples)
    if n < 4 or trim_ratio <= 0.0:
        return round(sum(samples) / n, 4)
    s = sorted(samples)
    k = int(n * trim_ratio)
    if 2 * k >= n:
        return round(sum(samples) / n, 4)
    trimmed = s[k : n - k]
    return round(sum(trimmed) / len(trimmed), 4)


@dataclass
class BenchmarkMetrics:
    warmup_runs: int
    measured_runs: int
    batch_size: int
    total_time_ms: float
    min_ms: float
    mean_ms: float
    median_ms: float
    trimmed_mean_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    std_dev_ms: float
    throughput_fps: float
    throughput_mean_fps: float
    raw_latencies_ms: List[float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def calculate_summary_stats(
    raw_latencies_ms: List[float],
    warmup_runs: int = 10,
    batch_size: int = 1
) -> BenchmarkMetrics:
    """
    Compute rigorous statistical metrics from measured latency samples.
    Throughput is mathematically consistent with median latency:
        throughput_fps = (batch_size * 1000.0) / median_ms
    """
    if not raw_latencies_ms:
        return BenchmarkMetrics(
            warmup_runs=warmup_runs,
            measured_runs=0,
            batch_size=batch_size,
            total_time_ms=0.0,
            min_ms=0.0,
            mean_ms=0.0,
            median_ms=0.0,
            trimmed_mean_ms=0.0,
            p90_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            max_ms=0.0,
            std_dev_ms=0.0,
            throughput_fps=0.0,
            throughput_mean_fps=0.0,
            raw_latencies_ms=[]
        )

    n = len(raw_latencies_ms)
    s = sorted(raw_latencies_ms)
    total_time = sum(raw_latencies_ms)
    mean_val = total_time / n

    # Exact median
    if n % 2 == 1:
        median_val = s[n // 2]
    else:
        median_val = (s[(n // 2) - 1] + s[n // 2]) / 2.0

    p90_val = calculate_percentile(raw_latencies_ms, 90.0)
    p95_val = calculate_percentile(raw_latencies_ms, 95.0)
    p99_val = calculate_percentile(raw_latencies_ms, 99.0)
    trimmed_mean = calculate_trimmed_mean(raw_latencies_ms, trim_ratio=0.1)

    min_val = s[0]
    max_val = s[-1]

    # Sample standard deviation
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in raw_latencies_ms) / (n - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    # Consistent throughput based on median latency
    throughput_median = calculate_fps(median_val, batch_size=batch_size)
    throughput_mean = calculate_fps(mean_val, batch_size=batch_size)

    return BenchmarkMetrics(
        warmup_runs=warmup_runs,
        measured_runs=n,
        batch_size=batch_size,
        total_time_ms=round(total_time, 3),
        min_ms=round(min_val, 4),
        mean_ms=round(mean_val, 4),
        median_ms=round(median_val, 4),
        trimmed_mean_ms=round(trimmed_mean, 4),
        p90_ms=round(p90_val, 4),
        p95_ms=round(p95_val, 4),
        p99_ms=round(p99_val, 4),
        max_ms=round(max_val, 4),
        std_dev_ms=round(std_dev, 4),
        throughput_fps=throughput_median,
        throughput_mean_fps=throughput_mean,
        raw_latencies_ms=[round(x, 4) for x in raw_latencies_ms]
    )
