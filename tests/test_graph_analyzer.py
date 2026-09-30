"""Unit tests for GraphAnalyzer and hybrid partitioner."""
import unittest
from pathlib import Path
from app.models.model_inspector import ModelInspector
from app.models.compatibility import SnapdragonCompatibilityEngine
from app.models.graph_analyzer import GraphAnalyzer, GraphStructure


class TestGraphAnalyzer(unittest.TestCase):
    def test_graph_partitioning(self):
        yolo_path = Path("E:/snapdragon/models/yolov8_with_nms.onnx")
        meta = ModelInspector.inspect(yolo_path)
        compat = SnapdragonCompatibilityEngine.analyze(meta)
        graph = GraphAnalyzer.analyze_graph(meta, compat)

        self.assertIsInstance(graph, GraphStructure)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        self.assertGreater(len(graph.partitions), 0)
        self.assertGreaterEqual(graph.npu_compute_pct, 90.0)


if __name__ == "__main__":
    unittest.main()
