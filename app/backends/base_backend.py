"""
Base Execution Backend for SnapForge.
Defines the standard abstract contract for all execution targets:
CPU, GPU, Snapdragon NPU, Hybrid, and Qualcomm AI Hub.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BaseBackend(ABC):
    """Abstract Base Class for model execution backends."""

    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Return information about the physical execution device."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return supported data types, precision modes, and operator capabilities."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend can execute on the current host machine."""
        pass

    @abstractmethod
    def validate_model(self, model_path: str | Path) -> Tuple[bool, str]:
        """Validate whether the model can run on this backend."""
        pass

    @abstractmethod
    def load_model(self, model_path: str | Path, options: Optional[Dict[str, Any]] = None) -> Any:
        """Load and initialize the model session."""
        pass

    @abstractmethod
    def run_inference(self, input_feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute a single inference pass and return output dictionary."""
        pass

    @abstractmethod
    def benchmark(
        self,
        model_path: str | Path,
        warmup_runs: int = 10,
        measured_runs: int = 100,
        sample_input: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Benchmark latency and throughput on this backend."""
        pass
