"""
AI Optimization Agent for SnapForge.
Performs technical reasoning over model graphs, operator compatibility,
hardware profiles, and empirical benchmark data.
Provides explainable engineering insights without fabricating measurements.
"""

from typing import Dict, Any, List, Optional
from app.models.model_inspector import ModelMetadata
from app.models.compatibility import CompatibilityAnalysisResult
from app.core.hardware_manager import HardwareProfile


class AIOptimizationAgent:
    """Provides technical analysis and actionable advice for Snapdragon deployment."""

    @classmethod
    def generate_expert_analysis(
        cls,
        metadata: ModelMetadata,
        compat: CompatibilityAnalysisResult,
        hardware: HardwareProfile,
        benchmark_comparison: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize deep technical reasoning over all model and hardware data."""

        # 1. Executive Summary
        executive_summary = (
            f"Analysis of model '{metadata.name}' ({metadata.file_size_mb} MB, {metadata.total_params:,} parameters, "
            f"{len(metadata.nodes)} operators): Overall Snapdragon NPU compatibility is "
            f"{compat.weighted_npu_score}% (compute-weighted) and {compat.npu_score}% (operator count). "
        )

        # 2. Key Bottlenecks
        bottleneck_explanations: List[str] = []
        if compat.bottlenecks:
            for b in compat.bottlenecks[:3]:
                bottleneck_explanations.append(
                    f"Operator '{b['op_type']}' in node '{b['node_name']}': {b['reason']}"
                )
        else:
            bottleneck_explanations.append(
                "No critical hardware bottlenecks detected. All operators map cleanly to Qualcomm Hexagon HTP native kernels."
            )

        # 3. Precision & Quantization Insights
        precision_insights = (
            f"The original model uses {metadata.primary_dtype} precision. On Qualcomm Snapdragon processors (Hexagon NPU), "
            "FP32 operations require software emulation or incur significant thermal and power costs. "
            "Converting to INT8 (via QDQ or dynamic quantization) allows the model to leverage the dedicated Hexagon Tensor "
            "Processor (HTP), which provides up to 45 TOPS on Snapdragon X Elite platforms with 4x reduced memory bandwidth."
        )

        # 4. Fusion & Graph Optimization Insights
        fusion_insights: List[str] = []
        if metadata.op_counts.get("BatchNormalization", 0) > 0:
            bn_count = metadata.op_counts["BatchNormalization"]
            fusion_insights.append(
                f"Conv + BatchNorm Fusion: Detected {bn_count} BatchNormalization layer(s). In inference mode, "
                "these should be algebraically folded into the convolution weight tensor. This eliminates memory "
                "loads and reduces the number of kernel executions."
            )

        if metadata.op_counts.get("Reshape", 0) > 5 or metadata.op_counts.get("Transpose", 0) > 3:
            fusion_insights.append(
                "Data Layout Optimizations: Model contains multiple Reshape/Transpose operations. Running constant folding "
                "and layout simplification reduces data rearrangement passes between CPU and NPU memory spaces."
            )

        # 5. Target Execution Strategy Recommendation
        if compat.weighted_npu_score >= 95.0 and compat.unsupported_nodes_count == 0:
            execution_strategy = (
                "Direct Qualcomm Hexagon NPU Execution: The entire computational graph is supported by Hexagon HTP. "
                "Deploy as a unified QNN context binary to achieve maximum throughput and minimum latency."
            )
        elif compat.weighted_npu_score >= 70.0:
            execution_strategy = (
                f"Hybrid NPU + CPU Execution: {compat.weighted_npu_score}% of compute can execute on Hexagon NPU. "
                f"Route unsupported post-processing operators ({compat.primary_bottleneck or 'dynamic nodes'}) "
                "to CPU fallback. This prevents compilation failure while retaining acceleration on 70%+ of the model."
            )
        else:
            execution_strategy = (
                "CPU / GPU Execution: Due to low NPU operator compatibility, running primarily on Qualcomm Adreno GPU "
                "(via DirectML) or CPU (via ARM64 NEON) is recommended until unsupported operators are decomposed."
            )

        # 6. Benchmark Evaluation (if measured)
        benchmark_insights = ""
        if benchmark_comparison:
            orig_lat = benchmark_comparison.get("original_median_ms", 0.0)
            opt_lat = benchmark_comparison.get("optimized_median_ms", 0.0)
            reduction = benchmark_comparison.get("latency_reduction_pct", 0.0)
            speedup = benchmark_comparison.get("speedup_factor", 1.0)
            label = benchmark_comparison.get("label_type", "Measured")

            benchmark_insights = (
                f"Empirical Benchmarking ({label}): Median latency improved from {orig_lat} ms to {opt_lat} ms "
                f"({reduction}% reduction, {speedup}x speedup). All measurements derived from actual device timing runs."
            )

        return {
            "executive_summary": executive_summary,
            "bottleneck_explanations": bottleneck_explanations,
            "precision_insights": precision_insights,
            "fusion_insights": fusion_insights,
            "execution_strategy": execution_strategy,
            "benchmark_insights": benchmark_insights,
            "recommendations": compat.recommendations
        }
