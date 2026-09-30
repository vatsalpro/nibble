"""
Computational Graph Analyzer for SnapForge.
Builds node topological dependencies, edge flows, subgraph partitions,
and identifies NPU vs CPU boundary crossings for hybrid execution.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Set, Optional
from app.models.model_inspector import ModelMetadata, NodeInfo
from app.models.compatibility import SnapdragonCompatibilityEngine, CompatibilityAnalysisResult, SupportLevel


@dataclass
class GraphEdge:
    source_node: str
    target_node: str
    tensor_name: str
    shape: List[Any] = field(default_factory=list)
    dtype: str = "float32"


@dataclass
class GraphPartition:
    partition_id: int
    target_backend: str  # "NPU", "CPU", "GPU"
    node_names: List[str]
    input_tensors: List[str]
    output_tensors: List[str]
    estimated_flops: int
    flops_percentage: float


@dataclass
class GraphStructure:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    partitions: List[Dict[str, Any]]
    npu_compute_pct: float
    cpu_fallback_compute_pct: float
    boundary_crossings: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GraphAnalyzer:
    """Analyzes ONNX computational graph topology and partition boundaries."""

    @classmethod
    def analyze_graph(cls, metadata: ModelMetadata, compat: Optional[CompatibilityAnalysisResult] = None) -> GraphStructure:
        if compat is None:
            compat = SnapdragonCompatibilityEngine.analyze(metadata)

        compat_map = {n.node_name: n for n in compat.node_evaluations}

        # Build tensor producer -> list of consumers map
        tensor_producer: Dict[str, str] = {}
        tensor_consumers: Dict[str, List[str]] = {}

        # Initial graph inputs
        for inp in metadata.inputs:
            tensor_producer[inp.name] = "__GRAPH_INPUT__"

        for node in metadata.nodes:
            for out in node.outputs:
                tensor_producer[out] = node.name
            for inp in node.inputs:
                if inp not in tensor_consumers:
                    tensor_consumers[inp] = []
                tensor_consumers[inp].append(node.name)

        # Build edges
        edges: List[GraphEdge] = []
        for node in metadata.nodes:
            for inp in node.inputs:
                prod = tensor_producer.get(inp)
                if prod:
                    edges.append(GraphEdge(
                        source_node=prod,
                        target_node=node.name,
                        tensor_name=inp
                    ))

        # Build UI node representations with color status
        ui_nodes = []
        for node in metadata.nodes:
            eval_info = compat_map.get(node.name)
            npu_stat = eval_info.npu_status if eval_info else SupportLevel.UNKNOWN
            
            # Status color tag: green, yellow, red
            if npu_stat == SupportLevel.SUPPORTED:
                color = "#00D26A"  # Green
                status_label = "NPU Supported"
            elif npu_stat == SupportLevel.PARTIAL:
                color = "#FF9F1C"  # Yellow/Amber
                status_label = "Partial Fallback"
            elif npu_stat == SupportLevel.UNSUPPORTED:
                color = "#E63946"  # Red
                status_label = "NPU Unsupported"
            else:
                color = "#8D99AE"  # Gray / Unknown
                status_label = "Unknown"

            ui_nodes.append({
                "id": node.name,
                "label": f"{node.op_type}\n({node.name[:18]})",
                "op_type": node.op_type,
                "name": node.name,
                "color": color,
                "status": status_label,
                "npu_status": npu_stat,
                "gpu_status": eval_info.gpu_status if eval_info else "UNKNOWN",
                "cpu_status": eval_info.cpu_status if eval_info else "SUPPORTED",
                "input_shapes": node.input_shapes,
                "output_shapes": node.output_shapes,
                "param_count": node.param_count,
                "estimated_flops": node.estimated_flops,
                "data_type": node.data_type,
                "notes": eval_info.npu_notes if eval_info else ""
            })

        # Subgraph Partitioning for Hybrid Execution
        partitions = cls._partition_subgraphs(metadata.nodes, compat_map)

        total_flops = max(1, metadata.total_flops)
        npu_flops = sum(p.estimated_flops for p in partitions if p.target_backend == "NPU")
        cpu_flops = sum(p.estimated_flops for p in partitions if p.target_backend == "CPU")

        npu_pct = round((npu_flops / total_flops) * 100.0, 1)
        cpu_pct = round((cpu_flops / total_flops) * 100.0, 1)
        boundary_crossings = max(0, len(partitions) - 1)

        return GraphStructure(
            nodes=ui_nodes,
            edges=[asdict(e) for e in edges],
            partitions=[asdict(p) for p in partitions],
            npu_compute_pct=npu_pct,
            cpu_fallback_compute_pct=cpu_pct,
            boundary_crossings=boundary_crossings
        )

    @classmethod
    def _partition_subgraphs(
        cls,
        nodes: List[NodeInfo],
        compat_map: Dict[str, Any]
    ) -> List[GraphPartition]:
        """Group consecutive nodes with identical execution target into subgraphs."""
        partitions: List[GraphPartition] = []
        if not nodes:
            return partitions

        current_target = None
        current_node_names: List[str] = []
        current_flops = 0
        current_inputs: Set[str] = set()
        current_outputs: Set[str] = set()
        partition_id = 0

        for node in nodes:
            eval_info = compat_map.get(node.name)
            is_npu = eval_info and eval_info.npu_status == SupportLevel.SUPPORTED
            target = "NPU" if is_npu else "CPU"

            if target != current_target and current_node_names:
                partitions.append(GraphPartition(
                    partition_id=partition_id,
                    target_backend=current_target,
                    node_names=current_node_names,
                    input_tensors=list(current_inputs),
                    output_tensors=list(current_outputs),
                    estimated_flops=current_flops,
                    flops_percentage=0.0
                ))
                partition_id += 1
                current_node_names = []
                current_flops = 0
                current_inputs = set()
                current_outputs = set()

            current_target = target
            current_node_names.append(node.name)
            current_flops += node.estimated_flops
            current_inputs.update(node.inputs)
            current_outputs.update(node.outputs)

        if current_node_names and current_target:
            partitions.append(GraphPartition(
                partition_id=partition_id,
                target_backend=current_target,
                node_names=current_node_names,
                input_tensors=list(current_inputs),
                output_tensors=list(current_outputs),
                estimated_flops=current_flops,
                flops_percentage=0.0
            ))

        total_flops = max(1, sum(p.estimated_flops for p in partitions))
        for p in partitions:
            p.flops_percentage = round((p.estimated_flops / total_flops) * 100.0, 1)

        return partitions
