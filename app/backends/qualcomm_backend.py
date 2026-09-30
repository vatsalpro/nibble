"""
Dedicated Qualcomm Integration Backend for SnapForge.
Manages Qualcomm Hexagon QNN configurations, HTP performance modes,
and local QNN compiler tools.

Strictly isolated: on non-Snapdragon systems (such as AMD or Intel PCs),
cleanly reports unavailable and raises SnapdragonExecutionUnavailableError
if execution is attempted.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from app.backends.base_backend import BaseBackend
from app.backends.npu_backend import NPUBackend


class SnapdragonExecutionUnavailableError(RuntimeError):
    """Raised when an attempt is made to execute models on the Qualcomm backend on non-Snapdragon hardware."""
    pass


class QualcommBackend(BaseBackend):
    """Encapsulates Qualcomm-specific toolchains and HTP compiler options."""

    def __init__(self):
        super().__init__(name="Qualcomm AI Engine")
        from app.core.hardware_manager import HardwareManager
        self.hw_profile = HardwareManager.get_hardware_profile()
        self.npu_backend = NPUBackend()

    def is_available(self) -> bool:
        """Available ONLY if host is genuine Snapdragon and Hexagon NPU is ready."""
        return bool(self.hw_profile.is_snapdragon and self.npu_backend.is_available())

    def status(self) -> str:
        """Returns uppercase status string."""
        return "AVAILABLE" if self.is_available() else "NOT AVAILABLE"

    def reason(self) -> str:
        """Explains why the backend is available or unavailable."""
        if self.is_available():
            return "Qualcomm Hexagon NPU and QNN Execution Provider are active."
        if not self.hw_profile.is_snapdragon:
            return "Qualcomm Snapdragon hardware was not detected."
        return "Qualcomm QNNExecutionProvider / QNN SDK is not configured in this environment."

    def get_device_info(self) -> Dict[str, Any]:
        info = self.npu_backend.get_device_info()
        info["status"] = self.status()
        info["reason"] = self.reason()
        info["is_snapdragon_detected"] = self.hw_profile.is_snapdragon
        return info

    def get_capabilities(self) -> Dict[str, Any]:
        return self.npu_backend.get_capabilities()

    def validate_model(self, model_path: str | Path) -> Tuple[bool, str]:
        if not self.is_available():
            return False, f"Qualcomm backend unavailable: {self.reason()}"
        return self.npu_backend.validate_model(model_path)

    def load_model(self, model_path: str | Path, options: Optional[Dict[str, Any]] = None):
        if not self.is_available():
            raise SnapdragonExecutionUnavailableError(
                f"Qualcomm NPU execution is not supported on this {self.hw_profile.cpu_vendor} device. {self.reason()}"
            )
        return self.npu_backend.load_model(model_path, options)

    def run_inference(self, input_feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.is_available():
            raise SnapdragonExecutionUnavailableError(
                f"Qualcomm NPU execution is not supported on this {self.hw_profile.cpu_vendor} device. {self.reason()}"
            )
        return self.npu_backend.run_inference(input_feed)

    def benchmark(
        self,
        model_path: str | Path,
        warmup_runs: int = 10,
        measured_runs: int = 50,
        sample_input: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise SnapdragonExecutionUnavailableError(
                f"Qualcomm NPU execution is not supported on this {self.hw_profile.cpu_vendor} device. {self.reason()}"
            )
        return self.npu_backend.benchmark(
            model_path=model_path,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            sample_input=sample_input
        )

    def get_htp_compilation_flags(self, precision: str = "INT8", performance_mode: str = "burst") -> Dict[str, str]:
        """Generate official Qualcomm QNN HTP compiler options for future Snapdragon target deployment."""
        return {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": performance_mode.lower(),  # "burst", "sustained_high_performance", "power_saver"
            "htp_graph_finalization_optimization_mode": "3",  # O3 maximum optimization
            "vtcm_size_in_mb": "8",  # Vector Tightly Coupled Memory allocation
            "precision": precision.upper()
        }
