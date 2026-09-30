"""Unit tests for benchmarking statistics and accuracy validation."""
import unittest
import numpy as np
from pathlib import Path
from app.profiling.latency import LatencyTimer, LatencyStats
from app.profiling.memory import MemoryTracker
from app.profiling.accuracy import AccuracyValidator
from app.backends.cpu_backend import CPUBackend


class TestBenchmarkAndAccuracy(unittest.TestCase):
    def test_latency_statistics(self):
        latencies = [10.0, 12.0, 11.0, 15.0, 13.0, 11.5, 12.5, 14.0, 10.5, 12.0]
        stats = LatencyTimer.calculate_stats(latencies, warmup_runs=2)
        self.assertIsInstance(stats, LatencyStats)
        self.assertAlmostEqual(stats.median_ms, 12.0, places=1)
        self.assertGreater(stats.throughput_fps, 0)
        self.assertGreaterEqual(stats.p95_ms, stats.median_ms)
        self.assertGreaterEqual(stats.max_ms, stats.min_ms)

    def test_accuracy_validator(self):
        arr1 = {"out": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)}
        arr2 = {"out": np.array([[1.01, 1.99, 3.02], [3.98, 5.01, 5.99]], dtype=np.float32)}
        comp = AccuracyValidator.compare_outputs(arr1, arr2)
        self.assertGreater(comp["overall_cosine_similarity"], 0.99)
        self.assertLess(comp["overall_mae"], 0.05)
        self.assertIn("fidelity_grade", comp)
        self.assertEqual(comp["top_prediction_preserved"], "YES")
        self.assertFalse(comp["contains_nan_or_inf"])
        self.assertGreaterEqual(comp["overall_relative_error"], 0.0)

    def test_accuracy_nan_inf_detection(self):
        arr1 = {"out": np.array([1.0, 2.0, float("nan")], dtype=np.float32)}
        arr2 = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        comp = AccuracyValidator.compare_outputs(arr1, arr2)
        self.assertTrue(comp["contains_nan_or_inf"])
        self.assertEqual(comp["top_prediction_preserved"], "NO")
        self.assertIn("Corrupted", comp["fidelity_grade"])

    def test_accuracy_zero_vectors(self):
        arr1 = {"out": np.zeros((4,), dtype=np.float32)}
        arr2 = {"out": np.zeros((4,), dtype=np.float32)}
        comp = AccuracyValidator.compare_outputs(arr1, arr2)
        self.assertEqual(comp["overall_cosine_similarity"], 1.0)
        self.assertEqual(comp["overall_relative_error"], 0.0)

    def test_cpu_backend_benchmark(self):
        p = Path("E:/snapdragon/models/resnet_classifier.onnx")
        cpu = CPUBackend()
        res = cpu.benchmark(p, warmup_runs=2, measured_runs=5)
        self.assertEqual(res["label_type"], "Measured")
        self.assertGreater(res["stats"]["median_ms"], 0)


if __name__ == "__main__":
    unittest.main()
