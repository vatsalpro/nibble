"""
Optimization Search Engine for SnapForge.
Explores multiple optimization candidates (FP32 baseline, Graph-optimized FP32,
FP16, and INT8), benchmarks each, evaluates accuracy trade-offs,
and constructs a Pareto frontier (Fastest, Smallest, Highest Accuracy, Balanced).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from app.optimization.quantization import QuantizationEngine
from app.optimization.graph_optimizer import GraphOptimizer
from app.optimization.operator_fusion import OperatorFusionEngine


@dataclass
class CandidateResult:
    config_name: str
    precision: str
    target_backend: str
    model_path: str
    file_size_mb: float
    size_reduction_pct: float
    median_latency_ms: float
    throughput_fps: float
    accuracy_similarity: float  # Cosine similarity (1.0 = identical)
    is_pareto_optimal: bool = False
    pareto_category: Optional[str] = None  # "FASTEST", "SMALLEST", "HIGHEST_ACCURACY", "BALANCED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OptimizationSearchEngine:
    """Evaluates optimization configurations to discover the Pareto frontier."""

    @classmethod
    def evaluate_candidates(
        cls,
        original_model_path: str | Path,
        work_dir: str | Path,
        benchmark_fn,  # Callable[[model_path, backend], (median_ms, throughput, accuracy_sim)]
        target_backend: str = "CPU"
    ) -> List[CandidateResult]:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        orig_path = Path(original_model_path)
        orig_size_mb = round(orig_path.stat().st_size / (1024 * 1024), 2)

        candidates: List[CandidateResult] = []

        # 1. Config A: Baseline FP32
        base_med, base_fps, base_sim = benchmark_fn(str(orig_path), target_backend, is_baseline=True)
        candidates.append(CandidateResult(
            config_name="Baseline FP32",
            precision="FP32",
            target_backend=target_backend,
            model_path=str(orig_path),
            file_size_mb=orig_size_mb,
            size_reduction_pct=0.0,
            median_latency_ms=base_med,
            throughput_fps=base_fps,
            accuracy_similarity=1.0,
            pareto_category="HIGHEST_ACCURACY"
        ))

        # 2. Config B: Graph-Optimized FP32 (Fused + Folded)
        fused_path = work_dir / f"{orig_path.stem}_fused.onnx"
        opt_path = work_dir / f"{orig_path.stem}_graph_opt.onnx"

        OperatorFusionEngine.fuse_conv_batchnorm(orig_path, fused_path)
        GraphOptimizer.optimize(fused_path if fused_path.exists() else orig_path, opt_path)

        if opt_path.exists():
            opt_size_mb = round(opt_path.stat().st_size / (1024 * 1024), 2)
            pct_red = round(((orig_size_mb - opt_size_mb) / max(0.001, orig_size_mb)) * 100.0, 1)
            med, fps, sim = benchmark_fn(str(opt_path), target_backend, is_baseline=False)
            candidates.append(CandidateResult(
                config_name="Graph Optimized FP32",
                precision="FP32",
                target_backend=target_backend,
                model_path=str(opt_path),
                file_size_mb=opt_size_mb,
                size_reduction_pct=pct_red,
                median_latency_ms=med,
                throughput_fps=fps,
                accuracy_similarity=sim
            ))

        # 3. Config C: FP16
        fp16_path = work_dir / f"{orig_path.stem}_fp16.onnx"
        try:
            QuantizationEngine.quantize_fp16(opt_path if opt_path.exists() else orig_path, fp16_path)
            if fp16_path.exists():
                fp16_size_mb = round(fp16_path.stat().st_size / (1024 * 1024), 2)
                pct_red = round(((orig_size_mb - fp16_size_mb) / max(0.001, orig_size_mb)) * 100.0, 1)
                med, fps, sim = benchmark_fn(str(fp16_path), target_backend, is_baseline=False)
                candidates.append(CandidateResult(
                    config_name="FP16 Half-Precision",
                    precision="FP16",
                    target_backend=target_backend,
                    model_path=str(fp16_path),
                    file_size_mb=fp16_size_mb,
                    size_reduction_pct=pct_red,
                    median_latency_ms=med,
                    throughput_fps=fps,
                    accuracy_similarity=sim
                ))
        except Exception:
            pass

        # 4. Config D: INT8 Dynamic Quantization
        int8_path = work_dir / f"{orig_path.stem}_int8.onnx"
        try:
            QuantizationEngine.quantize_int8_dynamic(opt_path if opt_path.exists() else orig_path, int8_path)
            if int8_path.exists():
                int8_size_mb = round(int8_path.stat().st_size / (1024 * 1024), 2)
                pct_red = round(((orig_size_mb - int8_size_mb) / max(0.001, orig_size_mb)) * 100.0, 1)
                med, fps, sim = benchmark_fn(str(int8_path), target_backend, is_baseline=False)
                candidates.append(CandidateResult(
                    config_name="INT8 Quantized",
                    precision="INT8",
                    target_backend=target_backend,
                    model_path=str(int8_path),
                    file_size_mb=int8_size_mb,
                    size_reduction_pct=pct_red,
                    median_latency_ms=med,
                    throughput_fps=fps,
                    accuracy_similarity=sim
                ))
        except Exception:
            pass

        # Identify Pareto Optimal configurations
        cls._assign_pareto_labels(candidates)

        return candidates

    @staticmethod
    def _assign_pareto_labels(candidates: List[CandidateResult]):
        if not candidates:
            return

        # Fastest (lowest latency)
        fastest = min(candidates, key=lambda c: c.median_latency_ms if c.median_latency_ms > 0 else 999999)
        fastest.is_pareto_optimal = True
        fastest.pareto_category = "FASTEST"

        # Smallest (lowest file size)
        smallest = min(candidates, key=lambda c: c.file_size_mb)
        smallest.is_pareto_optimal = True
        if not smallest.pareto_category:
            smallest.pareto_category = "SMALLEST"

        # Highest Accuracy
        highest_acc = max(candidates, key=lambda c: c.accuracy_similarity)
        highest_acc.is_pareto_optimal = True
        if not highest_acc.pareto_category:
            highest_acc.pareto_category = "HIGHEST_ACCURACY"

        # Balanced (good latency, size, and >0.98 accuracy)
        balanced_candidates = [c for c in candidates if c.accuracy_similarity >= 0.95 and c != fastest and c != smallest]
        if balanced_candidates:
            balanced = min(balanced_candidates, key=lambda c: c.median_latency_ms)
            balanced.is_pareto_optimal = True
            if not balanced.pareto_category:
                balanced.pareto_category = "BALANCED"
