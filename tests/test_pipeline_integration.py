"""Integration test for SnapForge end-to-end pipeline."""
import unittest
from pathlib import Path
from app.core.project_manager import ProjectManager
from app.core.model_manager import ModelManager
from app.core.optimization_manager import OptimizationManager
from app.core.benchmark_manager import BenchmarkManager
from app.reports.report_generator import ReportGenerator
from app.core.hardware_manager import HardwareManager


class TestEndToEndPipeline(unittest.TestCase):
    def test_complete_workflow(self):
        model_path = Path("E:/snapdragon/models/resnet_classifier.onnx")
        self.assertTrue(model_path.exists())

        # 1. Project Creation
        proj = ProjectManager.create_project("E2E Test Project", model_path=str(model_path))
        self.assertIsNotNone(proj.id)

        # 2. Model Import & Inspection
        import_res = ModelManager.import_model(model_path, project_id=proj.id)
        meta = import_res["metadata"]
        compat = import_res["compatibility"]
        self.assertGreater(meta.total_params, 0)
        self.assertGreater(compat.weighted_npu_score, 0)

        # 3. Optimization
        opt_res = OptimizationManager.run_optimization(
            input_model_path=model_path,
            project_id=proj.id,
            model_id=import_res["model_id"],
            precision="INT8"
        )
        self.assertTrue(Path(opt_res["optimized_model_path"]).exists())
        self.assertGreater(opt_res["size_reduction_pct"], 0.0)

        # 4. Differential Benchmark
        bench_res = BenchmarkManager.compare_before_after(
            original_model_path=str(model_path),
            optimized_model_path=opt_res["optimized_model_path"],
            warmup_runs=3,
            measured_runs=10
        )
        self.assertGreater(bench_res["original_median_ms"], 0)
        self.assertGreater(bench_res["optimized_median_ms"], 0)
        self.assertEqual(bench_res["label_type"], "Measured")

        # 5. Report Generation
        hw = HardwareManager.get_hardware_profile().to_dict()
        reports = ReportGenerator.generate_full_report(
            project_data={"name": proj.name, "status": "Benchmarked"},
            hardware_data=hw,
            model_data=meta.to_dict(),
            compat_data=compat.to_dict(),
            opt_data=opt_res,
            bench_data=bench_res,
            output_format="ALL",
            project_id=proj.id
        )
        self.assertTrue(Path(reports["PDF"]).exists())
        self.assertTrue(Path(reports["JSON"]).exists())
        self.assertTrue(Path(reports["CSV"]).exists())


if __name__ == "__main__":
    unittest.main()
