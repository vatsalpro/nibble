"""Unit tests for OperatorFusionEngine, GraphOptimizer, and QuantizationEngine."""
import unittest
from pathlib import Path
import onnx
from app.optimization.operator_fusion import OperatorFusionEngine
from app.optimization.graph_optimizer import GraphOptimizer
from app.optimization.quantization import QuantizationEngine


class TestOptimizationEngines(unittest.TestCase):
    def setUp(self):
        self.resnet_path = Path("E:/snapdragon/models/resnet_classifier.onnx")
        self.out_dir = Path("E:/snapdragon/tests/output")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def test_conv_batchnorm_fusion(self):
        fused_path = self.out_dir / "test_fused.onnx"
        res = OperatorFusionEngine.fuse_conv_batchnorm(self.resnet_path, fused_path)
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["fused_count"], 0)
        self.assertTrue(fused_path.exists())

    def test_graph_optimizer(self):
        opt_path = self.out_dir / "test_opt.onnx"
        res = GraphOptimizer.optimize(self.resnet_path, opt_path, opt_level="BASIC")
        self.assertTrue(res["success"])
        self.assertTrue(opt_path.exists())

    def test_int8_dynamic_quantization(self):
        q_path = self.out_dir / "test_int8.onnx"
        res = QuantizationEngine.quantize_int8_dynamic(self.resnet_path, q_path)
        self.assertTrue(res["success"])
        self.assertEqual(res["precision"], "INT8 (Dynamic)")
        self.assertGreater(res["size_reduction_pct"], 0.0)
        self.assertTrue(q_path.exists())


if __name__ == "__main__":
    unittest.main()
