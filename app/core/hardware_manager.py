"""
Hardware Manager for SnapForge.
Performs genuine hardware discovery on Windows:
- CPU model, architecture, cores, frequency
- Snapdragon platform detection (Snapdragon X Elite, X Plus, etc.)
- GPU detection (Qualcomm Adreno, AMD, NVIDIA, Intel)
- NPU detection (Qualcomm Hexagon NPU, QNN SDK, ONNX Runtime QNN EP, DirectML)
- RAM, Battery, OS, and Qualcomm toolchain status.
"""

import os
import sys
import platform
import subprocess
import shutil
import psutil
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class HardwareProfile:
    # CPU
    cpu_vendor: str = "Unknown"
    cpu_model: str = "Unknown"
    cpu_arch: str = "Unknown"
    is_arm64: bool = False
    is_snapdragon: bool = False
    snapdragon_model: Optional[str] = None
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: float = 0.0

    # Device & OEM
    oem_manufacturer: str = "Unknown"
    system_model: str = "Unknown"

    # GPU
    gpu_devices: List[str] = field(default_factory=list)
    has_adreno_gpu: bool = False

    # NPU
    npu_present: bool = False
    npu_name: str = "Not detected"
    npu_status: str = "Not detected"
    npu_details: str = ""
    is_hexagon: bool = False

    # RAM & Battery & OS
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    has_battery: bool = False
    battery_percent: Optional[float] = None
    battery_charging: Optional[bool] = None
    os_name: str = "Windows"
    os_release: str = ""
    os_build: str = ""

    # Qualcomm Toolchain
    qnn_sdk_installed: bool = False
    qnn_sdk_path: Optional[str] = None
    ort_qnn_ep_available: bool = False
    ort_dml_ep_available: bool = False
    available_ort_providers: List[str] = field(default_factory=list)
    qai_hub_cli_available: bool = False
    qai_hub_token_configured: bool = False

    @property
    def qnn_status(self) -> str:
        return "Active" if self.ort_qnn_ep_available else "Not active"

    @property
    def snapdragon_npu_display_status(self) -> str:
        if self.is_snapdragon and self.npu_present:
            return "Available"
        return "Not detected"

    @property
    def hardware_badge_text(self) -> str:
        if self.is_snapdragon:
            return f"Snapdragon PC ({self.snapdragon_model or 'Snapdragon'})"
        return f"Development Machine: {self.cpu_vendor} (Non-Snapdragon)"

    @property
    def target_badge_text(self) -> str:
        return "Snapdragon Target: Ready for deployment on Snapdragon hardware"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["qnn_status"] = self.qnn_status
        d["hardware_badge_text"] = self.hardware_badge_text
        d["target_badge_text"] = self.target_badge_text
        return d


class HardwareManager:
    """Detects actual system hardware and Qualcomm runtime components."""

    _cached_profile: Optional[HardwareProfile] = None

    @classmethod
    def get_hardware_profile(cls, force_refresh: bool = False) -> HardwareProfile:
        if cls._cached_profile is not None and not force_refresh:
            return cls._cached_profile

        profile = HardwareProfile()
        cls._detect_cpu_and_system(profile)
        cls._detect_memory_and_battery(profile)
        cls._detect_gpus(profile)
        cls._detect_npu_and_qualcomm(profile)
        cls._cached_profile = profile
        return profile

    get_profile = get_hardware_profile

    @staticmethod
    def _detect_cpu_and_system(profile: HardwareProfile):
        profile.cpu_arch = platform.machine()
        profile.os_name = platform.system()
        profile.os_release = platform.release()
        profile.os_build = platform.version()

        # Architecture check
        is_arm = profile.cpu_arch.upper() in ("ARM64", "AARCH64")
        profile.is_arm64 = is_arm

        # CPU Cores & Frequency
        profile.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        profile.cpu_cores_logical = psutil.cpu_count(logical=True) or 1
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            profile.cpu_freq_mhz = cpu_freq.current or cpu_freq.max or 0.0

        # Query Windows Management Information via PowerShell for accurate hardware strings
        try:
            cmd = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0 and res.stdout.strip():
                profile.cpu_model = res.stdout.strip()
            else:
                profile.cpu_model = platform.processor() or "Generic Processor"
        except Exception:
            profile.cpu_model = platform.processor() or "Generic Processor"

        # Check OEM manufacturer and computer model
        try:
            cmd = "Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model | ConvertTo-Json"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0 and res.stdout.strip():
                import json
                info = json.loads(res.stdout.strip())
                profile.oem_manufacturer = info.get("Manufacturer", "Unknown").strip()
                profile.system_model = info.get("Model", "Unknown").strip()
        except Exception:
            pass

        # Determine CPU Vendor accurately
        cpu_str_lower = profile.cpu_model.lower()
        if "amd" in cpu_str_lower or "ryzen" in cpu_str_lower or "radeon" in cpu_str_lower or "advanced micro devices" in cpu_str_lower:
            profile.cpu_vendor = "AMD"
        elif "intel" in cpu_str_lower or "core" in cpu_str_lower or "xeon" in cpu_str_lower:
            profile.cpu_vendor = "Intel"
        elif "snapdragon" in cpu_str_lower or "qualcomm" in cpu_str_lower:
            profile.cpu_vendor = "Qualcomm"
        elif "apple" in cpu_str_lower:
            profile.cpu_vendor = "Apple"
        else:
            profile.cpu_vendor = "Generic"

        # Check if Qualcomm Snapdragon (ARM64 does NOT imply Snapdragon)
        if "snapdragon" in cpu_str_lower or "qualcomm" in cpu_str_lower:
            profile.is_snapdragon = True
            profile.cpu_vendor = "Qualcomm"
            if "x elite" in cpu_str_lower:
                profile.snapdragon_model = "Snapdragon X Elite"
            elif "x plus" in cpu_str_lower:
                profile.snapdragon_model = "Snapdragon X Plus"
            elif "8cx" in cpu_str_lower:
                profile.snapdragon_model = "Snapdragon 8cx"
            else:
                profile.snapdragon_model = profile.cpu_model
        else:
            profile.is_snapdragon = False
            profile.snapdragon_model = None

    @staticmethod
    def _detect_memory_and_battery(profile: HardwareProfile):
        mem = psutil.virtual_memory()
        profile.total_ram_gb = round(mem.total / (1024 ** 3), 2)
        profile.available_ram_gb = round(mem.available / (1024 ** 3), 2)

        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                profile.has_battery = True
                profile.battery_percent = round(battery.percent, 1)
                profile.battery_charging = battery.power_plugged
        except Exception:
            profile.has_battery = False

    @staticmethod
    def _detect_gpus(profile: HardwareProfile):
        try:
            cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
                profile.gpu_devices = lines
                for dev in lines:
                    if "adreno" in dev.lower() or "qualcomm" in dev.lower():
                        profile.has_adreno_gpu = True
        except Exception:
            profile.gpu_devices = ["Unknown Display Adapter"]

    @staticmethod
    def _detect_npu_and_qualcomm(profile: HardwareProfile):
        # 1. Check ONNX Runtime Execution Providers
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            profile.available_ort_providers = providers
            profile.ort_qnn_ep_available = "QNNExecutionProvider" in providers
            profile.ort_dml_ep_available = "DmlExecutionProvider" in providers
        except Exception:
            profile.available_ort_providers = []

        # 2. Check Qualcomm Neural Processing SDK / QNN environment variables or paths
        qnn_env = os.environ.get("QNN_SDK_ROOT") or os.environ.get("SNPE_ROOT")
        default_qnn_paths = [
            r"C:\Qualcomm\QNN",
            r"C:\Program Files\Qualcomm\QNN",
            r"C:\Qualcomm\NeuralProcessingSDK"
        ]
        if qnn_env and os.path.exists(qnn_env):
            profile.qnn_sdk_installed = True
            profile.qnn_sdk_path = qnn_env
        else:
            for p in default_qnn_paths:
                if os.path.exists(p):
                    profile.qnn_sdk_installed = True
                    profile.qnn_sdk_path = p
                    break

        # 3. Check Qualcomm AI Hub token and CLI
        qai_token = os.environ.get("QAI_HUB_API_TOKEN")
        profile.qai_hub_token_configured = bool(qai_token and len(qai_token.strip()) > 5)
        profile.qai_hub_cli_available = shutil.which("qai-hub") is not None

        # 4. Check Windows PnP device list for NPU / Hexagon hardware
        found_npu_device = False
        npu_device_name = ""
        try:
            cmd = "Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match '\\bNPU\\b|Hexagon|Qualcomm.*Neural|Snapdragon.*NPU|Neural Processing Unit' } | Select-Object -ExpandProperty Name"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0 and res.stdout.strip():
                detected = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
                if detected:
                    found_npu_device = True
                    npu_device_name = detected[0]
        except Exception:
            pass

        # Determine overall NPU status
        if profile.is_snapdragon:
            if found_npu_device and ("hexagon" in npu_device_name.lower() or "qualcomm" in npu_device_name.lower()):
                profile.npu_present = True
                profile.npu_name = npu_device_name
                profile.is_hexagon = True
                if profile.ort_qnn_ep_available or profile.qnn_sdk_installed:
                    profile.npu_status = "Available"
                    profile.npu_details = f"{npu_device_name} (Runtime ready: QNN active)"
                else:
                    profile.npu_status = "Hardware Detected (Runtime Missing)"
                    profile.npu_details = f"{npu_device_name} detected, but QNNExecutionProvider / QNN SDK is not configured in Python."
            elif profile.ort_qnn_ep_available:
                profile.npu_present = True
                profile.npu_name = "Qualcomm Hexagon NPU (via QNN EP)"
                profile.npu_status = "Available"
                profile.is_hexagon = True
                profile.npu_details = "QNNExecutionProvider is active and ready for model compilation."
            else:
                profile.npu_present = False
                profile.npu_name = "Not detected"
                profile.npu_status = "Not detected"
                profile.is_hexagon = False
                profile.npu_details = "Qualcomm Hexagon NPU hardware or driver not detected on this Snapdragon device."
        else:
            profile.npu_present = False
            profile.npu_name = "Not detected"
            profile.npu_status = "Not detected"
            profile.is_hexagon = False
            if profile.cpu_vendor == "AMD":
                profile.npu_details = (
                    "Development Hardware: AMD x64. "
                    "Snapdragon NPU: Not detected. "
                    "Qualcomm QNN: Not active. "
                    "Qualcomm backend is isolated and ready for testing on Snapdragon hardware."
                )
            else:
                profile.npu_details = (
                    f"No Qualcomm Hexagon NPU hardware detected on this host ({profile.cpu_arch}). "
                    "Hardware-dependent integration point: Requires Snapdragon X-series ARM64 Windows device "
                    "(e.g., HP OmniBook X) with Qualcomm QNN Execution Provider."
                )
