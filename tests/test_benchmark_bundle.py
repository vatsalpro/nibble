"""Unit tests for Benchmark Bundle generation."""
import unittest
import json
from pathlib import Path
from nibble.benchmark.bundle import create_benchmark_bundle


class TestBenchmarkBundle(unittest.TestCase):
    def test_create_bundle(self):
        orig_path = Path("models/resnet_classifier.onnx")
        opt_path = Path("models/resnet_classifier.onnx")
        self.assertTrue(orig_path.exists())

        dummy_comparison = {
            "original_median_ms": 1.25,
            "optimized_median_ms": 1.05,
            "latency_reduction_pct": 16.0,
            "speedup_factor": 1.19,
            "original_throughput_fps": 800.0,
            "optimized_throughput_fps": 952.4,
            "size_reduction_pct": 0.0,
            "original_stats": {
                "median_ms": 1.25,
                "mean_ms": 1.26,
                "p95_ms": 1.35,
                "min_ms": 1.20,
                "max_ms": 1.50,
                "throughput_fps": 800.0,
            },
            "optimized_stats": {
                "median_ms": 1.05,
                "mean_ms": 1.06,
                "p95_ms": 1.15,
                "min_ms": 1.00,
                "max_ms": 1.30,
                "throughput_fps": 952.4,
            },
            "accuracy": {
                "overall_cosine_similarity": 0.9999,
                "overall_mae": 0.0001,
                "overall_rmse": 0.0002,
                "overall_relative_error": 0.0001,
                "top_prediction_preserved": "YES",
                "fidelity_grade": "Excellent (Indistinguishable from FP32)",
                "contains_nan_or_inf": False,
            }
        }

        reports_dir = Path("reports")
        bundle_dir = create_benchmark_bundle(
            original_model_path=orig_path,
            optimized_model_path=opt_path,
            comparison_data=dummy_comparison,
            provider="CPUExecutionProvider",
            intra_op_threads=4,
            inter_op_threads=1,
            warmup_runs=5,
            measured_runs=10,
            batch_size=1,
            reports_dir=reports_dir,
        )

        self.assertTrue(bundle_dir.exists())
        result_json = bundle_dir / "result.json"
        summary_md = bundle_dir / "summary.md"
        self.assertTrue(result_json.exists())
        self.assertTrue(summary_md.exists())

        # Check JSON content
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("environment", data)
        self.assertIn("execution_configuration", data)
        self.assertIn("models", data)
        self.assertEqual(len(data["models"]["original"]["sha256"]), 64)
        self.assertIn("scorecard", data)

        # Check Markdown content
        md_text = summary_md.read_text(encoding="utf-8")
        self.assertIn("Reproducibility & Integrity Hashes", md_text)
        self.assertIn("SHA-256 Digest", md_text)
        self.assertIn("[MEASURED]", md_text)


if __name__ == "__main__":
    unittest.main()
