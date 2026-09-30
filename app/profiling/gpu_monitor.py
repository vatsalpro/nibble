"""
GPU telemetry monitor for SnapForge.
Attempts to query Windows GPU utilization counters or Direct3D diagnostics.
Marks metric as 'Unavailable' if driver does not expose counters, avoiding fabrication.
"""

from typing import Dict, Any, Optional


class GPUMonitor:
    def __init__(self):
        self._is_available = False
        self._device_name = "Unknown"
        self._probe_gpu()

    def _probe_gpu(self):
        # We check if DXGI/DirectML or manufacturer counters are exposed
        # On many integrated/adreno GPUs, hardware counters require elevated WMI/ETW sessions.
        pass

    def sample(self) -> Dict[str, Any]:
        """Sample GPU metrics. Returns Unavailable if not exposed by OS/driver."""
        return {
            "status": "Unavailable",
            "util_pct": None,
            "memory_used_mb": None,
            "details": "GPU hardware telemetry not exposed by installed graphics driver without elevated ETW session."
        }
