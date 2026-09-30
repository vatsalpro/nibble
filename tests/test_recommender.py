"""Unit tests for OptimizationRecommender and OptimizationScorecard."""
import unittest
from nibble.optimizer.recommender import OptimizationRecommender, OptimizationScorecard


class TestOptimizationRecommender(unittest.TestCase):
    def test_fp16_cpu_slowdown_recommendation(self):
        """Test that FP16 model with size reduction but CPU latency slowdown is honestly evaluated."""
        card = OptimizationRecommender.evaluate(
            model_name="test_model",
            optimization_type="FP16 Quantization",
            original_size_bytes=10_000_000,
            optimized_size_bytes=5_200_000, # -48%
            original_latency_ms=10.0,
            optimized_latency_ms=11.2, # +12% slowdown on CPU
            original_throughput_fps=100.0,
            optimized_throughput_fps=89.3,
            cosine_similarity=0.9992,
            top_prediction_preserved="YES",
            contains_nan_or_inf=False,
            execution_provider="CPUExecutionProvider",
        )
        self.assertIsInstance(card, OptimizationScorecard)
        self.assertFalse(card.is_recommended)
        self.assertEqual(card.recommendation_grade, "SPECIALIZED (NPU/GPU ONLY)")
        self.assertFalse(card.target_recommendations["host_cpu"]["recommended"])
        self.assertTrue(card.target_recommendations["snapdragon_npu"]["recommended"])
        self.assertTrue(card.target_recommendations["directml_gpu"]["recommended"])
        self.assertIn("NPU", card.summary_narrative)

    def test_genuine_speedup_recommendation(self):
        """Test that genuine speedup with high fidelity is highly recommended."""
        card = OptimizationRecommender.evaluate(
            model_name="test_model",
            optimization_type="Graph Optimization",
            original_size_bytes=10_000_000,
            optimized_size_bytes=9_800_000,
            original_latency_ms=10.0,
            optimized_latency_ms=7.5, # 25% faster
            original_throughput_fps=100.0,
            optimized_throughput_fps=133.3,
            cosine_similarity=1.0,
            top_prediction_preserved="YES",
            contains_nan_or_inf=False,
            execution_provider="CPUExecutionProvider",
        )
        self.assertTrue(card.is_recommended)
        self.assertEqual(card.recommendation_grade, "HIGHLY RECOMMENDED")
        self.assertGreater(card.speedup_ratio, 1.3)
        self.assertTrue(card.target_recommendations["host_cpu"]["recommended"])

    def test_accuracy_corruption_rejection(self):
        """Test that numerical corruption causes unconditional rejection."""
        card = OptimizationRecommender.evaluate(
            model_name="test_model",
            optimization_type="Aggressive Quantization",
            original_size_bytes=10_000_000,
            optimized_size_bytes=2_000_000,
            original_latency_ms=10.0,
            optimized_latency_ms=4.0, # very fast
            original_throughput_fps=100.0,
            optimized_throughput_fps=250.0,
            cosine_similarity=0.62, # poor similarity
            top_prediction_preserved="NO",
            contains_nan_or_inf=False,
            execution_provider="CPUExecutionProvider",
        )
        self.assertFalse(card.is_recommended)
        self.assertEqual(card.recommendation_grade, "REJECTED (ACCURACY LOSS)")
        self.assertFalse(card.target_recommendations["host_cpu"]["recommended"])
        self.assertFalse(card.target_recommendations["snapdragon_npu"]["recommended"])


if __name__ == "__main__":
    unittest.main()
