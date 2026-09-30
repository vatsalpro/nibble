"""
AI Optimization Planner for SnapForge.
The algorithmic brain of SnapForge:
Given a model, detected hardware, and user objective, formulates
a tailored, hardware-informed optimization strategy for Snapdragon deployment.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from app.models.model_inspector import ModelMetadata
from app.models.compatibility import CompatibilityAnalysisResult
from app.core.hardware_manager import HardwareProfile, HardwareManager


class OptimizationObjective:
    MAX_PERFORMANCE = "Maximum Performance"
    MAX_BATTERY = "Maximum Battery Efficiency"
    MAX_ACCURACY = "Maximum Accuracy"
    BALANCED = "Balanced"


@dataclass
class PlanStep:
    step_number: int
    name: str
    action_type: str  # "quantization", "graph_optimization", "fusion", "replacement", "backend_selection", "benchmark"
    description: str
    rationale: str
    accuracy_risk: str  # "None", "Very Low", "Low", "Moderate"
    expected_benefit: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationPlan:
    objective: str
    target_backend: str
    target_precision: str
    steps: List[PlanStep] = field(default_factory=list)
    estimated_speedup_factor: float = 1.0
    estimated_size_reduction_pct: float = 0.0
    summary: str = ""
    hardware_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OptimizationPlanner:
    """Generates structured optimization plans for Snapdragon PCs."""

    @classmethod
    def create_plan(
        cls,
        metadata: ModelMetadata,
        compat: CompatibilityAnalysisResult,
        objective: str = OptimizationObjective.BALANCED,
        hardware: Optional[HardwareProfile] = None
    ) -> OptimizationPlan:
        if hardware is None:
            hardware = HardwareManager.get_hardware_profile()

        steps: List[PlanStep] = []
        step_idx = 1

        # 1. Conv + BatchNorm Fusion (always safe & eliminates graph nodes)
        if metadata.op_counts.get("BatchNormalization", 0) > 0:
            bn_count = metadata.op_counts["BatchNormalization"]
            steps.append(PlanStep(
                step_number=step_idx,
                name="Conv + BatchNormalization Folding",
                action_type="fusion",
                description=f"Fold {bn_count} BatchNormalization layer(s) directly into preceding Conv weights and bias.",
                rationale="BatchNorm in inference is a linear transform. Merging it into Conv weights eliminates kernel launches and memory bandwidth without altering outputs.",
                accuracy_risk="None",
                expected_benefit="Zero runtime memory overhead for BatchNorm, reduces graph size."
            ))
            step_idx += 1

        # 2. Graph Optimization & Dead Node Removal
        steps.append(PlanStep(
            step_number=step_idx,
            name="Constant Folding & Redundant Operator Elimination",
            action_type="graph_optimization",
            description="Evaluate constant expressions at compile time, eliminate dead nodes, and remove redundant Identity/Transpose nodes.",
            rationale="Pre-computing constant subgraphs reduces runtime evaluation overhead and simplifies the graph for Qualcomm Hexagon HTP ingestion.",
            accuracy_risk="None",
            expected_benefit="5% - 15% reduction in total operator count."
        ))
        step_idx += 1

        # 3. Precision Selection & Quantization Strategy
        target_precision = "INT8"
        est_size_red = 70.0
        est_speedup = 2.5

        if objective == OptimizationObjective.MAX_ACCURACY:
            target_precision = "FP16"
            est_size_red = 50.0
            est_speedup = 1.6
            steps.append(PlanStep(
                step_number=step_idx,
                name="FP16 Precision Conversion",
                action_type="quantization",
                description="Convert weights and floating point operations to IEEE 754 Half-Precision (FP16).",
                rationale="FP16 preserves 99.9%+ numerical accuracy while cutting memory bandwidth by 50% and doubling vector SIMD throughput on Snapdragon Adreno GPU / Hexagon HTP.",
                accuracy_risk="Very Low",
                expected_benefit="50% model size reduction, ~1.5x - 2.0x throughput improvement."
            ))
        elif objective == OptimizationObjective.MAX_BATTERY:
            target_precision = "INT8"
            est_size_red = 75.0
            est_speedup = 3.5
            steps.append(PlanStep(
                step_number=step_idx,
                name="Static INT8 (QDQ) Quantization",
                action_type="quantization",
                description="Quantize weights and activations to INT8 with calibration dataset.",
                rationale="Qualcomm Hexagon HTP operates at peak TOPS/Watt efficiency in INT8. Minimizing memory bandwidth directly preserves battery on Snapdragon laptops.",
                accuracy_risk="Low",
                expected_benefit="75% model size reduction, optimal energy efficiency per inference."
            ))
        elif objective == OptimizationObjective.MAX_PERFORMANCE:
            target_precision = "INT8"
            est_size_red = 75.0
            est_speedup = 4.0
            steps.append(PlanStep(
                step_number=step_idx,
                name="INT8 Quantization for Hexagon Tensor Acceleration",
                action_type="quantization",
                description="Quantize dense and convolutional layers to 8-bit integers.",
                rationale="Hexagon Tensor Processor achieves maximum throughput on INT8 matrix operations.",
                accuracy_risk="Low",
                expected_benefit="Maximized inference throughput and lowest latency."
            ))
        else:  # Balanced
            target_precision = "INT8"
            est_size_red = 72.0
            est_speedup = 2.8
            steps.append(PlanStep(
                step_number=step_idx,
                name="Dynamic / Static INT8 Quantization",
                action_type="quantization",
                description="Apply 8-bit quantization with accuracy verification.",
                rationale="Optimal trade-off: 70%+ memory footprint reduction with minimal measurable loss in output similarity.",
                accuracy_risk="Low",
                expected_benefit="~3x latency reduction and 70% smaller memory footprint."
            ))
        step_idx += 1

        # 4. Target Execution Backend Selection
        target_backend = "Snapdragon NPU"
        if compat.weighted_npu_score < 70.0 or compat.unsupported_nodes_count > 0:
            target_backend = "Hybrid NPU + CPU"
            steps.append(PlanStep(
                step_number=step_idx,
                name="Hybrid Subgraph Partitioning",
                action_type="backend_selection",
                description=f"Partition graph: route {compat.weighted_npu_score}% compute to Hexagon NPU, fallback remaining {round(100 - compat.weighted_npu_score, 1)}% ops to CPU.",
                rationale=f"Model contains {compat.unsupported_nodes_count} operator(s) unsupported on NPU (such as {compat.primary_bottleneck or 'dynamic ops'}). Hybrid partitioning prevents execution failure while keeping compute-heavy layers on NPU.",
                accuracy_risk="None",
                expected_benefit="Full model executability with NPU acceleration on supported subgraphs."
            ))
        else:
            steps.append(PlanStep(
                step_number=step_idx,
                name="Qualcomm QNN Hexagon NPU Compilation",
                action_type="backend_selection",
                description="Direct deployment to Qualcomm Hexagon HTP via QNN Execution Provider.",
                rationale=f"Model is {compat.weighted_npu_score}% compatible with zero fatal bottlenecks. Full NPU execution avoids host-to-device memory roundtrips.",
                accuracy_risk="None",
                expected_benefit="Sub-millisecond inference and zero CPU host utilization."
            ))
        step_idx += 1

        # 5. Benchmarking & Accuracy Validation
        steps.append(PlanStep(
            step_number=step_idx,
            name="Differential Benchmarking & Output Similarity Validation",
            action_type="benchmark",
            description="Measure latency percentiles (Median, P95) and compute Cosine Similarity & MAE between original and optimized models.",
            rationale="Ensures optimization did not degrade accuracy and verifies real hardware speedup on the actual device.",
            accuracy_risk="None",
            expected_benefit="Empirical verification of latency improvement and numerical fidelity."
        ))

        # Hardware Diagnostic Note
        hw_notes = ""
        if not hardware.npu_present:
            hw_notes = (
                f"Note: Current host hardware ({hardware.cpu_model}) does not possess Qualcomm Hexagon NPU. "
                "Optimization and local benchmarking will execute via the CPU Execution Provider with [Measured] labels, "
                "while Qualcomm NPU target profiles will be generated for Snapdragon deployment."
            )
        else:
            hw_notes = f"Detected Snapdragon platform ({hardware.snapdragon_model or hardware.cpu_model}). NPU status: {hardware.npu_status}."

        summary_text = (
            f"Tailored plan for objective '{objective}': Apply Conv+BatchNorm fusion, {target_precision} quantization, "
            f"and compile for target '{target_backend}'. Estimated analytical size reduction: {est_size_red}%, "
            f"projected speedup: {est_speedup}x."
        )

        return OptimizationPlan(
            objective=objective,
            target_backend=target_backend,
            target_precision=target_precision,
            steps=steps,
            estimated_speedup_factor=est_speedup,
            estimated_size_reduction_pct=est_size_red,
            summary=summary_text,
            hardware_notes=hw_notes
        )
