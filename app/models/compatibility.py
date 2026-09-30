"""
Snapdragon Compatibility Engine for SnapForge.
Evaluates computational graphs against documented Qualcomm Hexagon NPU,
Qualcomm Adreno GPU, and CPU execution capabilities.
Calculates both operator count score and compute-weighted (FLOPs) score.
Identifies execution bottlenecks and fallback points.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from app.models.model_inspector import ModelMetadata, NodeInfo


class SupportLevel:
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class OpCapability:
    npu: str
    gpu: str
    cpu: str
    npu_constraints: str = ""
    gpu_constraints: str = ""


# Real Qualcomm Hexagon NPU (QNN HTP v68/v69/v73/v75) & Adreno GPU Capability Registry
QUALCOMM_CAPABILITY_REGISTRY: Dict[str, OpCapability] = {
    # Convolution & Pooling
    "Conv": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Best with INT8/FP16; 1D, 2D, depthwise fully accelerated."),
    "ConvTranspose": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported on HTP; stride/padding constraints apply."),
    "MaxPool": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Fully accelerated."),
    "AveragePool": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Fully accelerated."),
    "GlobalAveragePool": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Fully accelerated."),
    "GlobalMaxPool": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Fully accelerated."),

    # Matrix Multiplication & Dense
    "Gemm": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Optimal in INT8 or FP16 on Hexagon Tensor Core."),
    "MatMul": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Batch matrix multiply accelerated on HTP."),

    # Activations
    "Relu": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Zero-cost when fused with Conv/Gemm."),
    "LeakyRelu": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated on HTP."),
    "PRelu": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported with broadcast parameters."),
    "Clip": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Often fused with prior operator."),
    "Sigmoid": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Look-up table or polynomial approximation on DSP."),
    "Tanh": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Look-up table accelerated."),
    "Softmax": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported along last axis; INT8 requires scale tracking."),
    "HardSigmoid": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported piecewise approximation."),
    "HardSwish": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported natively on HTP."),
    "Gelu": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported in opset 20+; older opsets decompose to Erf/Tanh."),
    "Elu": OpCapability(SupportLevel.PARTIAL, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Requires float emulation on older Hexagon cores."),
    "Erf": OpCapability(SupportLevel.PARTIAL, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Approximated on HTP; may fall back to CPU if accuracy threshold strict."),

    # Element-wise Arithmetic
    "Add": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Broadcast and scalar addition accelerated."),
    "Sub": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated."),
    "Mul": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated."),
    "Div": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated."),
    "Abs": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated."),
    "Neg": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated."),
    "Sqrt": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated on HTP."),
    "Exp": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "LUT accelerated on HTP."),
    "Log": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "LUT accelerated."),
    "Pow": OpCapability(SupportLevel.PARTIAL, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Integer power supported; arbitrary float exponent may fall back."),

    # Normalization
    "BatchNormalization": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Should ideally be fused into Conv weights during optimization."),
    "InstanceNormalization": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated on HTP."),
    "LayerNormalization": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported on HTP v68+."),
    "GroupNormalization": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported on HTP."),

    # Tensor Manipulation & Shape
    "Reshape": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Metadata-only tensor view on HTP."),
    "Transpose": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Hardware DMA accelerated; avoid back-to-back transposes."),
    "Flatten": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Zero-copy tensor reshape."),
    "Squeeze": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Metadata-only shape manipulation."),
    "Unsqueeze": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Metadata-only shape manipulation."),
    "Concat": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Channel or batch concatenation accelerated."),
    "Split": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated on HTP."),
    "Pad": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Constant, reflect, and edge padding supported."),
    "Slice": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Static step and bound slices accelerated."),
    "Gather": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Accelerated for embedding lookups."),
    "Shape": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Folded at compile time if static."),
    "Constant": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Folded into model initializers."),
    "Identity": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "No-op, removed during graph optimization."),
    "Cast": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported; avoid frequent cast churn between FP32 and INT8."),

    # Vision & Advanced Ops
    "Resize": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Nearest and Bilinear modes supported; Bicubic may fall back."),
    "Upsample": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Deprecated in ONNX, mapped to Resize."),
    "ReduceMean": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Spatial reduction accelerated."),
    "ReduceSum": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported on HTP."),
    "ReduceMax": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported on HTP."),
    "ReduceMin": OpCapability(SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Supported on HTP."),
    "ArgMax": OpCapability(SupportLevel.PARTIAL, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Often executed on CPU fallback in post-processing."),
    "TopK": OpCapability(SupportLevel.PARTIAL, SupportLevel.PARTIAL, SupportLevel.SUPPORTED, "Limited K size on HTP; CPU fallback common."),
    "NonZero": OpCapability(SupportLevel.PARTIAL, SupportLevel.PARTIAL, SupportLevel.SUPPORTED, "Dynamic non-zero shape causes pipeline stalls on NPU."),
    "Where": OpCapability(SupportLevel.PARTIAL, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Conditional selection supported if broadcastable."),
    "ScatterND": OpCapability(SupportLevel.PARTIAL, SupportLevel.PARTIAL, SupportLevel.SUPPORTED, "Sparse updates slow on tensor cores."),
    "GatherND": OpCapability(SupportLevel.PARTIAL, SupportLevel.PARTIAL, SupportLevel.SUPPORTED, "Multi-dimensional indexing partial support."),
    "GridSample": OpCapability(SupportLevel.PARTIAL, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Bilinear mode supported on newer HTP."),

    # Object Detection Post-Processing & Dynamic Control Flow
    "NonMaxSuppression": OpCapability(SupportLevel.UNSUPPORTED, SupportLevel.PARTIAL, SupportLevel.SUPPORTED, "Dynamic bounding box filtering; recommended for CPU execution target."),
    "RoiAlign": OpCapability(SupportLevel.PARTIAL, SupportLevel.SUPPORTED, SupportLevel.SUPPORTED, "Requires fixed box counts for HTP pipeline."),
    "Loop": OpCapability(SupportLevel.UNSUPPORTED, SupportLevel.UNSUPPORTED, SupportLevel.SUPPORTED, "Dynamic control flow unsupported on NPU graph compiler."),
    "If": OpCapability(SupportLevel.UNSUPPORTED, SupportLevel.UNSUPPORTED, SupportLevel.SUPPORTED, "Conditional branching unsupported on HTP."),
    "Scan": OpCapability(SupportLevel.UNSUPPORTED, SupportLevel.UNSUPPORTED, SupportLevel.SUPPORTED, "Recurrent scan unsupported."),
}


@dataclass
class NodeCompatibility:
    node_name: str
    op_type: str
    npu_status: str
    gpu_status: str
    cpu_status: str
    npu_notes: str
    gpu_notes: str
    estimated_flops: int
    data_type: str
    is_bottleneck: bool = False
    has_known_flops: bool = True
    compute_cost_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompatibilityAnalysisResult:
    model_name: str
    total_nodes: int
    npu_score: float             # Unweighted % of nodes supported
    weighted_npu_score: float    # FLOPs-weighted % of compute supported
    operator_coverage_pct: float = 0.0
    estimated_compute_coverage_pct: float = 0.0
    gpu_score: float = 0.0
    cpu_score: float = 100.0
    supported_nodes_count: int = 0
    partial_nodes_count: int = 0
    unsupported_nodes_count: int = 0
    unknown_compute_nodes_count: int = 0
    static_analysis_disclaimer: str = "Static analysis only. Actual Snapdragon NPU execution has not been tested."
    node_evaluations: List[NodeCompatibility] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    primary_bottleneck: Optional[str] = None
    hybrid_partition_candidate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SnapdragonCompatibilityEngine:
    """Analyzes ONNX models for Qualcomm Snapdragon X-series hardware compatibility."""

    @classmethod
    def analyze(cls, metadata: ModelMetadata) -> CompatibilityAnalysisResult:
        node_evals: List[NodeCompatibility] = []
        bottlenecks: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        total_nodes = len(metadata.nodes)
        total_flops = max(1, metadata.total_flops)

        supported_count = 0
        partial_count = 0
        unsupported_count = 0

        supported_flops = 0
        partial_flops = 0
        unsupported_flops = 0

        gpu_supported_count = 0
        cpu_supported_count = 0

        for node in metadata.nodes:
            op = node.op_type
            cap = QUALCOMM_CAPABILITY_REGISTRY.get(op)

            if cap is None:
                # Custom or uncataloged operator
                npu_stat = SupportLevel.UNKNOWN
                gpu_stat = SupportLevel.PARTIAL
                cpu_stat = SupportLevel.SUPPORTED
                npu_note = "Operator not in standard Qualcomm Hexagon HTP registry. Probable CPU fallback required."
                gpu_note = "May execute via DirectML if registered."
            else:
                npu_stat = cap.npu
                gpu_stat = cap.gpu
                cpu_stat = cap.cpu
                npu_note = cap.npu_constraints
                gpu_note = cap.gpu_constraints

            is_bottleneck = False
            if npu_stat == SupportLevel.UNSUPPORTED:
                unsupported_count += 1
                unsupported_flops += node.estimated_flops
                is_bottleneck = True
                bottlenecks.append({
                    "node_name": node.name,
                    "op_type": op,
                    "reason": npu_note or "Operator unsupported on Qualcomm Hexagon NPU.",
                    "estimated_flops": node.estimated_flops
                })
            elif npu_stat == SupportLevel.PARTIAL or npu_stat == SupportLevel.UNKNOWN:
                partial_count += 1
                partial_flops += node.estimated_flops
                if node.estimated_flops > 0.05 * total_flops:
                    is_bottleneck = True
                    bottlenecks.append({
                        "node_name": node.name,
                        "op_type": op,
                        "reason": f"Partial compatibility with significant compute ({node.estimated_flops} FLOPs). {npu_note}",
                        "estimated_flops": node.estimated_flops
                    })
            else:
                supported_count += 1
                supported_flops += node.estimated_flops

            if gpu_stat == SupportLevel.SUPPORTED:
                gpu_supported_count += 1
            if cpu_stat == SupportLevel.SUPPORTED:
                cpu_supported_count += 1

            has_known = node.estimated_flops > 0 or op in ("Conv", "Gemm", "MatMul")
            cost_label = f"{node.estimated_flops:,} FLOPs" if node.estimated_flops > 0 else "Unknown compute cost"

            node_evals.append(NodeCompatibility(
                node_name=node.name,
                op_type=op,
                npu_status=npu_stat,
                gpu_status=gpu_stat,
                cpu_status=cpu_stat,
                npu_notes=npu_note,
                gpu_notes=gpu_note,
                estimated_flops=node.estimated_flops,
                data_type=node.data_type,
                is_bottleneck=is_bottleneck,
                has_known_flops=has_known,
                compute_cost_label=cost_label
            ))

        # Calculate scores
        if total_nodes > 0:
            npu_score = round((supported_count / total_nodes) * 100.0, 1)
            gpu_score = round((gpu_supported_count / total_nodes) * 100.0, 1)
            cpu_score = round((cpu_supported_count / total_nodes) * 100.0, 1)
        else:
            npu_score = 0.0
            gpu_score = 0.0
            cpu_score = 100.0

        # FLOPs-weighted score (Supported = 1.0, Partial = 0.5, Unsupported = 0.0)
        weighted_supported_flops = supported_flops + (0.5 * partial_flops)
        weighted_npu_score = round(min(100.0, (weighted_supported_flops / total_flops) * 100.0), 1)

        # Explicit dual metrics per Section 9
        operator_coverage_pct = npu_score
        estimated_compute_coverage_pct = weighted_npu_score

        # Primary bottleneck
        primary_bottleneck = None
        if bottlenecks:
            # Sort by compute cost
            sorted_bottlenecks = sorted(bottlenecks, key=lambda b: b["estimated_flops"], reverse=True)
            top = sorted_bottlenecks[0]
            primary_bottleneck = f"{top['op_type']} ('{top['node_name']}') — {top['reason']}"

        # Generate Actionable Recommendations
        if metadata.primary_dtype == "float32":
            recommendations.append("Apply INT8 or FP16 Quantization: Snapdragon Hexagon HTP achieves up to 4x throughput and power efficiency in INT8.")

        # BatchNormalization check
        if metadata.op_counts.get("BatchNormalization", 0) > 0:
            recommendations.append(f"Fuse BatchNormalization ({metadata.op_counts['BatchNormalization']} nodes): Fold weights into adjacent Conv layers to eliminate runtime memory overhead.")

        # NMS check
        if metadata.op_counts.get("NonMaxSuppression", 0) > 0:
            recommendations.append("Configure Hybrid Execution: Route NonMaxSuppression post-processing to CPU while keeping feature backbone on Snapdragon NPU.")

        # Dynamic reshape check
        if metadata.op_counts.get("Reshape", 0) > 5:
            recommendations.append("Graph Optimization: Run constant folding and shape simplification to eliminate redundant Transpose and Reshape operations.")

        hybrid_candidate = (unsupported_count > 0 and weighted_npu_score >= 60.0)

        if hybrid_candidate:
            recommendations.append(f"Hybrid NPU + CPU Candidate: {weighted_npu_score}% of compute runs efficiently on Hexagon NPU; route remaining {round(100 - weighted_npu_score, 1)}% to CPU fallback.")

        return CompatibilityAnalysisResult(
            model_name=metadata.name,
            total_nodes=total_nodes,
            npu_score=npu_score,
            weighted_npu_score=weighted_npu_score,
            operator_coverage_pct=operator_coverage_pct,
            estimated_compute_coverage_pct=estimated_compute_coverage_pct,
            gpu_score=gpu_score,
            cpu_score=cpu_score,
            supported_nodes_count=supported_count,
            partial_nodes_count=partial_count,
            unsupported_nodes_count=unsupported_count,
            unknown_compute_nodes_count=sum(1 for n in node_evals if not n.has_known_flops),
            static_analysis_disclaimer="Static analysis only. Actual Snapdragon NPU execution has not been tested.",
            node_evaluations=node_evals,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            primary_bottleneck=primary_bottleneck,
            hybrid_partition_candidate=hybrid_candidate
        )
