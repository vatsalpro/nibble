"""
Hybrid Execution Backend for Nibble.
Clearly classified as EXPERIMENTAL / NOT IMPLEMENTED to prevent fake hybrid execution claims.
Does NOT simulate hybrid execution by running models entirely on CPU.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from app.backends.base_backend import BaseBackend


class HybridBackend(BaseBackend):
    """
    Experimental Hybrid NPU + CPU Execution Backend.
    Marked as NOT IMPLEMENTED for the MVP to prevent running entirely on CPU and falsely labeling it Hybrid.
    """

    def __init__(self):
        super().__init__(name="Hybrid (Snapdragon NPU + CPU)")

    def is_available(self) -> bool:
        return False

    def status(self) -> str:
        return "EXPERIMENTAL / NOT IMPLEMENTED"

    def reason(self) -> str:
        return (
            "True heterogeneous NPU+CPU execution requires on-device ONNX graph partitioning and zero-copy "
            "subgraph memory handoff. Nibble does not simulate hybrid execution by running models on CPU."
        )

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device_name": "Heterogeneous Snapdragon NPU + CPU Partitioner",
            "status": self.status(),
            "reason": self.reason(),
            "future_target": "Snapdragon Copilot+ PC with QNN Execution Provider"
        }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "target": "Graph Partitioning Subgraphs (QNN HTP + CPU fallback)",
            "status": "Not Implemented"
        }

    def validate_model(self, model_path: str | Path) -> Tuple[bool, str]:
        return False, f"Hybrid execution is not implemented: {self.reason()}"

    def load_model(self, model_path: str | Path, options: Optional[Dict[str, Any]] = None):
        raise NotImplementedError(f"Hybrid execution is not available: {self.reason()}")

    def run_inference(self, input_feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError(f"Hybrid execution is not available: {self.reason()}")

    def benchmark(
        self,
        model_path: str | Path,
        warmup_runs: int = 10,
        measured_runs: int = 50,
        sample_input: Optional[Dict[str, np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"Hybrid execution is not available: {self.reason()}")
