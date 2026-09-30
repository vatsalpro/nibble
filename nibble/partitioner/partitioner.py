"""
Graph Partitioning Architecture for Heterogeneous NPU + CPU Execution.
Defines the future contract for splitting an ONNX computational graph into
an NPU-accelerated primary subgraph and a CPU fallback subgraph.

Reserved for future Snapdragon copilot+ PC hardware deployment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class SubgraphPartition:
    partition_id: int
    target_backend: str  # "QNN_HTP", "CPU", "DIRECTML"
    node_names: List[str]
    input_tensors: List[str]
    output_tensors: List[str]
    estimated_compute_flops: int = 0


@dataclass
class PartitionPlan:
    model_name: str
    total_nodes: int
    npu_nodes_count: int
    cpu_fallback_nodes_count: int
    partitions: List[SubgraphPartition] = field(default_factory=list)
    boundary_crossings: int = 0
    is_supported: bool = False
    notes: str = ""


class GraphPartitioner:
    """
    Architectural interface for heterogeneous NPU+CPU graph partitioning.
    Not yet implemented in the MVP to prevent simulated or fake hybrid claims.
    """

    @classmethod
    def partition_graph(cls, model_path: str | Path) -> PartitionPlan:
        """
        Partition an ONNX model into NPU and CPU subgraphs.
        Currently returns NotImplemented to prevent dishonest claims of fake hybrid execution.
        """
        raise NotImplementedError(
            "True hybrid NPU+CPU execution is experimental and reserved for future Snapdragon Copilot+ PC deployment. "
            "To maintain strict engineering integrity, Nibble does not simulate hybrid execution by running models entirely on CPU."
        )
