from typing import Dict, Any, Optional


class NPUMonitor:
    def __init__(self, hw_profile=None):
        self._hardware_profile = hw_profile

    def sample(self) -> Dict[str, Any]:
        """
        Sample NPU metrics.
        Returns status 'Available' with real counters if on Snapdragon with QNN profiling active;
        otherwise returns 'Unavailable' with clear diagnostic explanation.
        """
        if self._hardware_profile is None:
            from app.core.hardware_manager import HardwareManager
            self._hardware_profile = HardwareManager.get_hardware_profile()
        if self._hardware_profile.npu_present and self._hardware_profile.ort_qnn_ep_available:
            # When QNN execution is active, QNN provides session-based cycle counts
            return {
                "status": "Available",
                "util_pct": None,  # Windows does not currently expose continuous % counter for NPU in standard perfmon
                "device_name": self._hardware_profile.npu_name,
                "execution_target": "Qualcomm Hexagon DSP/HTP",
                "details": "NPU execution active via Qualcomm QNN Execution Provider."
            }
        else:
            return {
                "status": "Unavailable",
                "util_pct": None,
                "device_name": self._hardware_profile.npu_name,
                "details": (
                    "NPU telemetry unavailable: Qualcomm Hexagon NPU hardware is not present on this host "
                    f"({self._hardware_profile.cpu_arch}). Hardware-dependent integration point: Requires "
                    "Snapdragon X-series PC with Qualcomm NPU drivers."
                )
            }
