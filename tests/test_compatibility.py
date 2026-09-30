"""Unit tests for SnapdragonCompatibilityEngine."""
import unittest
from pathlib import Path
from app.models.model_inspector import ModelInspector
from app.models.compatibility import SnapdragonCompatibilityEngine, SupportLevel


class TestCompatibilityEngine(unittest.TestCase):
    def setUp(self):
        self.resnet_path = Path("E:/snapdragon/models/resnet_classifier.onnx")
        self.yolo_nms_path = Path("E:/snapdragon/models/yolov8_with_nms.onnx")

    def test_resnet_compatibility(self):
        meta = ModelInspector.inspect(self.resnet_path)
        compat = SnapdragonCompatibilityEngine.analyze(meta)
        self.assertEqual(compat.weighted_npu_score, 100.0)
        self.assertGreater(compat.npu_score, 90.0)
        self.assertEqual(compat.unsupported_nodes_count, 0)
        self.assertIsNone(compat.primary_bottleneck)

    def test_yolo_nms_bottleneck_detection(self):
        meta = ModelInspector.inspect(self.yolo_nms_path)
        compat = SnapdragonCompatibilityEngine.analyze(meta)
        # NMS should be detected as unsupported bottleneck
        self.assertGreater(compat.unsupported_nodes_count, 0)
        self.assertIsNotNone(compat.primary_bottleneck)
        self.assertIn("NonMaxSuppression", compat.primary_bottleneck)
        self.assertTrue(compat.hybrid_partition_candidate)


if __name__ == "__main__":
    unittest.main()
