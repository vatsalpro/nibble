"""
Optimization Recommendation & Honest Scorecard Engine for Nibble.
Evaluates size vs. latency vs. fidelity trade-offs across execution targets.
Prevents deceptive "speedup" reporting when optimizations trade off CPU latency for memory savings.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from nibble.benchmark.metrics import (
    calculate_percent_change,
    calculate_speedup,
    calculate_latency_reduction,
)


@dataclass
class OptimizationScorecard:
    model_name: str
    optimization_type: str
    original_size_bytes: int
    optimized_size_bytes: int
    size_change_pct: float
    original_latency_ms: float
    optimized_latency_ms: float
    latency_change_pct: float
    speedup_ratio: float
    original_throughput_fps: float
    optimized_throughput_fps: float
    cosine_similarity: float
    top_prediction_preserved: str
    contains_nan_or_inf: bool
    is_recommended: bool
    recommendation_grade: str
    target_recommendations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tradeoffs: List[str] = field(default_factory=list)
    summary_narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OptimizationRecommender:
    """Evaluates optimization outcomes and generates objective scorecards."""

    @staticmethod
    def evaluate(
        model_name: str,
        optimization_type: str,
        original_size_bytes: int,
        optimized_size_bytes: int,
        original_latency_ms: float,
        optimized_latency_ms: float,
        original_throughput_fps: float,
        optimized_throughput_fps: float,
        cosine_similarity: float,
        top_prediction_preserved: str = "YES",
        contains_nan_or_inf: bool = False,
        execution_provider: str = "CPUExecutionProvider",
    ) -> OptimizationScorecard:
        """Evaluate optimization metrics and return an honest scorecard."""
        size_change = calculate_percent_change(original_size_bytes, optimized_size_bytes)
        lat_change = calculate_percent_change(original_latency_ms, optimized_latency_ms)
        speedup = calculate_speedup(original_latency_ms, optimized_latency_ms)

        tradeoffs: List[str] = []
        target_recs: Dict[str, Dict[str, Any]] = {}

        # 1. Check for numerical corruption
        if contains_nan_or_inf or cosine_similarity < 0.85 or top_prediction_preserved == "NO":
            tradeoffs.append("CRITICAL: Output numerical corruption or classification class shift detected.")
            is_rec = False
            rec_grade = "REJECTED (ACCURACY LOSS)"
            narrative = (
                f"Optimization '{optimization_type}' failed quality thresholds. "
                f"Cosine similarity is {cosine_similarity:.4f} and top prediction preserved is {top_prediction_preserved}. "
                "This model should NOT be deployed in production without recalibration."
            )
            for target in ("host_cpu", "snapdragon_npu", "directml_gpu"):
                target_recs[target] = {
                    "recommended": False,
                    "reason": "Accuracy degradation exceeds acceptable deployment limits."
                }
            return OptimizationScorecard(
                model_name=model_name,
                optimization_type=optimization_type,
                original_size_bytes=original_size_bytes,
                optimized_size_bytes=optimized_size_bytes,
                size_change_pct=size_change,
                original_latency_ms=original_latency_ms,
                optimized_latency_ms=optimized_latency_ms,
                latency_change_pct=lat_change,
                speedup_ratio=speedup,
                original_throughput_fps=original_throughput_fps,
                optimized_throughput_fps=optimized_throughput_fps,
                cosine_similarity=cosine_similarity,
                top_prediction_preserved=top_prediction_preserved,
                contains_nan_or_inf=contains_nan_or_inf,
                is_recommended=is_rec,
                recommendation_grade=rec_grade,
                target_recommendations=target_recs,
                tradeoffs=tradeoffs,
                summary_narrative=narrative,
            )

        # 2. Evaluate Size vs Latency trade-offs
        is_cpu = "cpu" in execution_provider.lower()
        is_fp16 = "fp16" in optimization_type.lower() or "float16" in optimization_type.lower()

        if size_change < -5.0:
            tradeoffs.append(f"Memory footprint reduced by {abs(size_change):.1f}%.")
        elif size_change > 5.0:
            tradeoffs.append(f"Memory footprint increased by {size_change:.1f}%.")

        if lat_change < -5.0:
            tradeoffs.append(f"Inference latency decreased by {abs(lat_change):.1f}% ({speedup:.2f}x speedup).")
        elif lat_change > 5.0:
            tradeoffs.append(f"Inference latency increased by {lat_change:.1f}% on current execution provider ({execution_provider}).")
        else:
            tradeoffs.append("Inference latency is statistically unchanged (within +/- 5%).")

        # 3. Determine honest target recommendations
        if is_fp16 and is_cpu and lat_change > 0:
            # Special case: FP16 on x86 CPU typically suffers software conversion penalty
            is_rec = False
            rec_grade = "SPECIALIZED (NPU/GPU ONLY)"
            narrative = (
                f"FP16 reduced model size by {abs(size_change):.1f}%, but increased CPU inference latency "
                f"by {lat_change:.1f}% due to software FP16 emulation on this host processor. "
                "Do NOT use this FP16 model for x86 CPU deployment. "
                "However, this model IS recommended for Snapdragon Hexagon NPU or DirectML GPU deployment, "
                "where native FP16 vector hardware will deliver significant acceleration and memory bandwidth reduction."
            )
            target_recs["host_cpu"] = {
                "recommended": False,
                "reason": f"Latency regression ({lat_change:+.1f}%) due to lack of native x86 FP16 execution pipeline."
            }
            target_recs["snapdragon_npu"] = {
                "recommended": True,
                "reason": "Hexagon NPU natively executes FP16 tensors with 2x memory bandwidth throughput."
            }
            target_recs["directml_gpu"] = {
                "recommended": True,
                "reason": "DirectML GPU compute units natively accelerate FP16 arithmetic."
            }

        elif lat_change <= -5.0:
            is_rec = True
            rec_grade = "HIGHLY RECOMMENDED"
            narrative = (
                f"Optimization '{optimization_type}' delivered a genuine {abs(lat_change):.1f}% latency reduction "
                f"({speedup:.2f}x speedup) with {cosine_similarity:.4f} output similarity. "
                "Recommended for production deployment on this hardware."
            )
            target_recs["host_cpu"] = {
                "recommended": True,
                "reason": f"Demonstrated {speedup:.2f}x speedup on host CPU with preserved accuracy."
            }
            target_recs["snapdragon_npu"] = {
                "recommended": True,
                "reason": "High operator coverage and optimized graph structure translate to efficient NPU execution."
            }
            target_recs["directml_gpu"] = {
                "recommended": True,
                "reason": "Clean fused operator graph is suitable for GPU execution."
            }

        elif size_change < -20.0 and abs(lat_change) <= 5.0:
            is_rec = True
            rec_grade = "RECOMMENDED (MEMORY CONSTRAINED)"
            narrative = (
                f"Optimization '{optimization_type}' reduced storage footprint by {abs(size_change):.1f}% "
                "with neutral latency change and high numerical accuracy. Recommended for memory-constrained deployments."
            )
            target_recs["host_cpu"] = {
                "recommended": True,
                "reason": "Neutral latency impact with significant memory footprint reduction."
            }
            target_recs["snapdragon_npu"] = {
                "recommended": True,
                "reason": "Smaller weight footprint optimizes Hexagon NPU TCM (Tightly Coupled Memory) utilization."
            }
            target_recs["directml_gpu"] = {
                "recommended": True,
                "reason": "Reduces GPU VRAM consumption with neutral compute latency."
            }

        else:
            is_rec = False
            rec_grade = "NOT RECOMMENDED"
            narrative = (
                f"Optimization '{optimization_type}' did not produce significant latency ({lat_change:+.1f}%) "
                f"or size ({size_change:+.1f}%) advantages on current execution hardware."
            )
            target_recs["host_cpu"] = {
                "recommended": False,
                "reason": "No meaningful latency or memory advantages observed."
            }
            target_recs["snapdragon_npu"] = {
                "recommended": False,
                "reason": "Verify compatibility before target deployment."
            }
            target_recs["directml_gpu"] = {
                "recommended": False,
                "reason": "No significant performance benefit observed."
            }

        return OptimizationScorecard(
            model_name=model_name,
            optimization_type=optimization_type,
            original_size_bytes=original_size_bytes,
            optimized_size_bytes=optimized_size_bytes,
            size_change_pct=size_change,
            original_latency_ms=original_latency_ms,
            optimized_latency_ms=optimized_latency_ms,
            latency_change_pct=lat_change,
            speedup_ratio=speedup,
            original_throughput_fps=original_throughput_fps,
            optimized_throughput_fps=optimized_throughput_fps,
            cosine_similarity=cosine_similarity,
            top_prediction_preserved=top_prediction_preserved,
            contains_nan_or_inf=contains_nan_or_inf,
            is_recommended=is_rec,
            recommendation_grade=rec_grade,
            target_recommendations=target_recs,
            tradeoffs=tradeoffs,
            summary_narrative=narrative,
        )
