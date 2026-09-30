"""Unit tests for nibble.benchmark.metrics."""
import unittest
from nibble.benchmark.metrics import (
    calculate_fps,
    calculate_speedup,
    calculate_latency_reduction,
    calculate_percent_change,
    calculate_percentile,
    calculate_p95,
    calculate_trimmed_mean,
    calculate_summary_stats,
    BenchmarkMetrics
)


class TestBenchmarkMetrics(unittest.TestCase):

    def test_calculate_fps_single_sample(self):
        self.assertEqual(calculate_fps(1.0, batch_size=1), 1000.0)
        self.assertEqual(calculate_fps(10.0, batch_size=1), 100.0)
        self.assertEqual(calculate_fps(0.1, batch_size=1), 10000.0)
        self.assertEqual(calculate_fps(0.0, batch_size=1), 0.0)
        self.assertEqual(calculate_fps(-5.0, batch_size=1), 0.0)

    def test_calculate_fps_batched(self):
        self.assertEqual(calculate_fps(10.0, batch_size=4), 400.0)
        self.assertEqual(calculate_fps(5.0, batch_size=8), 1600.0)

    def test_calculate_speedup(self):
        # Faster
        self.assertEqual(calculate_speedup(10.0, 5.0), 2.0)
        # Slower
        self.assertEqual(calculate_speedup(10.0, 12.5), 0.8)
        # Identical
        self.assertEqual(calculate_speedup(10.0, 10.0), 1.0)
        # Edge cases
        self.assertEqual(calculate_speedup(0.0, 5.0), 1.0)
        self.assertEqual(calculate_speedup(10.0, 0.0), 1.0)

    def test_calculate_latency_reduction(self):
        # 50% improvement
        self.assertEqual(calculate_latency_reduction(10.0, 5.0), 50.0)
        # 20% slowdown
        self.assertEqual(calculate_latency_reduction(10.0, 12.0), -20.0)
        # Identical
        self.assertEqual(calculate_latency_reduction(10.0, 10.0), 0.0)
        # Zero baseline
        self.assertEqual(calculate_latency_reduction(0.0, 5.0), 0.0)

    def test_calculate_percent_change(self):
        self.assertEqual(calculate_percent_change(100.0, 150.0), 50.0)
        self.assertEqual(calculate_percent_change(100.0, 50.0), -50.0)
        self.assertEqual(calculate_percent_change(0.0, 0.0), 0.0)

    def test_calculate_percentile_and_p95(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.assertAlmostEqual(calculate_percentile(samples, 50.0), 5.5, places=2)
        p95 = calculate_p95(samples)
        self.assertAlmostEqual(p95, 9.55, places=2)
        # Single element
        self.assertEqual(calculate_p95([4.2]), 4.2)
        # Empty
        self.assertEqual(calculate_p95([]), 0.0)

    def test_calculate_trimmed_mean(self):
        # 10 samples with a massive outlier at the end
        samples = [1.0, 1.0, 1.1, 1.0, 1.1, 1.0, 1.0, 1.1, 1.0, 100.0]
        normal_mean = sum(samples) / len(samples)
        trimmed = calculate_trimmed_mean(samples, trim_ratio=0.1)
        self.assertGreater(normal_mean, 10.0)
        self.assertLess(trimmed, 1.2)  # Extreme outlier safely pruned

    def test_summary_stats_mathematical_consistency(self):
        # Model A: faster median
        samples_a = [0.080, 0.081, 0.082, 0.083, 0.350]  # median 0.082
        # Model B: slower median
        samples_b = [0.090, 0.091, 0.091, 0.092, 0.100]  # median 0.091

        stats_a = calculate_summary_stats(samples_a)
        stats_b = calculate_summary_stats(samples_b)

        # Fundamental mathematical invariant:
        # If Model A has a lower median latency than Model B, Model A MUST have higher throughput!
        self.assertLess(stats_a.median_ms, stats_b.median_ms)
        self.assertGreater(stats_a.throughput_fps, stats_b.throughput_fps)
        self.assertEqual(stats_a.measured_runs, 5)
        self.assertIsInstance(stats_a.to_dict(), dict)


if __name__ == "__main__":
    unittest.main()
