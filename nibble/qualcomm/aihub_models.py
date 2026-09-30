"""
Qualcomm AI Hub Data Models for Nibble.
Defines typed models representing cloud devices, models, jobs, and execution profile results.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class AIHubDevice:
    name: str
    os: str = "Windows 11 ARM64"
    chipset: str = "Qualcomm Snapdragon"
    npu_tops: Optional[float] = None
    is_available: bool = True
    attributes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIHubModel:
    model_id: str
    name: str
    sha256: str
    size_bytes: int
    size_mb: float = 0.0
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    uploaded_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIHubJob:
    job_id: str
    job_type: str  # "profile", "compile", "inference"
    model_id: str
    device_name: str
    status: str  # "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
    url: str = ""
    created_at: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIHubProfileResult:
    job_id: str
    device_name: str
    runtime: str
    execution_target: str  # "NPU_CONFIRMED", "QNN_CONFIRMED", "CPU_EXECUTION", "GPU_EXECUTION", "UNKNOWN"
    verification_notes: str
    median_latency_ms: Optional[float]
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    throughput_fps: Optional[float] = None
    npu_layers_count: int = 0
    cpu_layers_count: int = 0
    total_layers_count: int = 0
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIHubValidationResult:
    model_name: str
    model_sha256: str
    static_npu_score_pct: float
    actual_npu_verified: bool
    actual_latency_ms: Optional[float]
    actual_device: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
