"""Unit tests for ReportGenerator."""
import unittest
import os
from pathlib import Path
from app.reports.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    def test_generate_all_formats(self):
        proj_data = {"name": "Test Project", "status": "Optimized"}
        hw_data = {"cpu_model": "AMD Ryzen 5", "cpu_arch": "AMD64", "total_ram_gb": 16.0, "npu_status": "Unavailable"}
        model_data = {"name": "test_net", "format": "ONNX", "file_size_mb": 1.2, "total_params": 5000}
        compat_data = {"weighted_npu_score": 95.0, "npu_score": 90.0, "gpu_score": 95.0, "cpu_score": 100.0}
        opt_data = {"precision": "INT8"}
        bench_data = {"original_median_ms": 10.0, "optimized_median_ms": 3.0, "latency_reduction_pct": 70.0, "speedup_factor": 3.33}

        files = ReportGenerator.generate_full_report(
            project_data=proj_data,
            hardware_data=hw_data,
            model_data=model_data,
            compat_data=compat_data,
            opt_data=opt_data,
            bench_data=bench_data,
            output_format="ALL"
        )

        self.assertIn("PDF", files)
        self.assertIn("JSON", files)
        self.assertIn("CSV", files)

        for fmt, path_str in files.items():
            p = Path(path_str)
            self.assertTrue(p.exists(), f"File for {fmt} was not created: {path_str}")
            self.assertGreater(p.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
