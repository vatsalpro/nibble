"""Backends package initialization."""
from app.backends.base_backend import BaseBackend
from app.backends.cpu_backend import CPUBackend
from app.backends.directml_backend import DirectMLBackend
from app.backends.gpu_backend import GPUBackend
from app.backends.npu_backend import NPUBackend
from app.backends.hybrid_backend import HybridBackend
from app.backends.qualcomm_backend import QualcommBackend, SnapdragonExecutionUnavailableError
from app.backends.qai_hub_client import QualcommAIHubClient

__all__ = [
    "BaseBackend", "CPUBackend", "DirectMLBackend", "GPUBackend",
    "NPUBackend", "HybridBackend", "QualcommBackend", "SnapdragonExecutionUnavailableError",
    "QualcommAIHubClient"
]
