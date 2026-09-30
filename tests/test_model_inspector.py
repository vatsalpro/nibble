"""Unit tests for ModelInspector."""
import unittest
from pathlib import Path
from app.models.model_inspector import ModelInspector, ModelMetadata


class TestModelInspector(unittest.TestCase):
    def setUp(self):
        self.model_path = Path("E:/snapdragon/models/resnet_classifier.onnx")
        self.assertTrue(self.model_path.exists(), "Sample test model resnet_classifier.onnx must exist.")

    def test_inspect_resnet(self):
        meta = ModelInspector.inspect(self.model_path)
        self.assertIsInstance(meta, ModelMetadata)
        self.assertEqual(meta.format, "ONNX")
        self.assertGreater(meta.total_params, 0)
        self.assertGreater(meta.file_size_bytes, 0)
        self.assertGreater(len(meta.nodes), 0)
        self.assertGreater(len(meta.inputs), 0)
        self.assertGreater(len(meta.outputs), 0)
        self.assertIn("Conv", meta.op_counts)

        # Hardened metrics verification
        self.assertEqual(len(meta.model_sha256), 64)
        self.assertGreaterEqual(meta.estimated_macs, 0)
        self.assertGreaterEqual(meta.weight_memory_mb, 0.0)
        self.assertGreaterEqual(meta.activation_memory_mb, 0.0)
        self.assertIsInstance(meta.largest_tensors, list)
        self.assertTrue(len(meta.largest_tensors) <= 5)
        if meta.largest_tensors:
            t0 = meta.largest_tensors[0]
            self.assertIn("name", t0)
            self.assertIn("shape", t0)
            self.assertIn("param_count", t0)
            self.assertIn("size_mb", t0)
            self.assertIn("dtype", t0)

    def test_inspect_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            ModelInspector.inspect("nonexistent_model.onnx")


if __name__ == "__main__":
    unittest.main()
