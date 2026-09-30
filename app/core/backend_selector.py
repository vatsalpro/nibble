"""
Automatic Backend Selector for SnapForge.
Evaluates model architecture, Snapdragon compatibility, hardware availability,
and latency/battery constraints to recommend and select the optimal execution backend.
"""

from typing import Dict, Any, Tuple
from app.models.model_inspector import ModelMetadata
from app.models.compatibility import CompatibilityAnalysisResult
from app.core.hardware_manager import HardwareProfile, HardwareManager


class BackendSelector:
    """Intelligently routes models to CPU, GPU, Snapdragon NPU, or Hybrid targets."""

    @classmethod
    def select_best_backend(
        cls,
        metadata: ModelMetadata,
        compat: CompatibilityAnalysisResult,
        hardware: HardwareProfile,
        objective: str = "Balanced"
    ) -> Dict[str, Any]:
        """
        Determines the optimal execution target with transparent reasoning.
        """
        reasons = []

        # 1. Check if host has Snapdragon NPU
        if hardware.is_snapdragon and hardware.npu_present and hardware.ort_qnn_ep_available:
            # On Snapdragon device with active NPU
            if compat.weighted_npu_score >= 95.0 and compat.unsupported_nodes_count == 0:
                target = "Snapdragon NPU"
                reasons.append(
                    f"Selected Snapdragon NPU because the model is {compat.weighted_npu_score}% compatible "
                    "with zero unsupported operators. The Hexagon Tensor Processor provides peak TOPS/Watt efficiency."
                )
            elif compat.weighted_npu_score >= 60.0:
                target = "Hybrid (Snapdragon NPU + CPU)"
                reasons.append(
                    f"Selected Hybrid (NPU + CPU) because {compat.weighted_npu_score}% of compute runs natively on Hexagon NPU, "
                    f"while {compat.unsupported_nodes_count} operator(s) (such as {compat.primary_bottleneck or 'dynamic ops'}) "
                    "are safely routed to CPU fallback."
                )
            else:
                if hardware.has_adreno_gpu and hardware.ort_dml_ep_available:
                    target = "Snapdragon GPU (Adreno)"
                    reasons.append(
                        "Selected Qualcomm Adreno GPU because NPU compatibility is below 60%, and GPU supports "
                        "the remaining floating-point operations via DirectML."
                    )
                else:
                    target = "CPU"
                    reasons.append(
                        "Selected CPU because model compatibility with Hexagon NPU is low and GPU EP is not active."
                    )
        else:
            # On host development PC without Snapdragon Hexagon NPU
            target = "CPU"
            reasons.append(
                f"Selected CPU because Qualcomm Hexagon NPU hardware is not present on this host machine ({hardware.cpu_model}, {hardware.cpu_arch}). "
                "Hardware-dependent integration point: Snapdragon deployment profile generated for target HP Snapdragon PCs."
            )
            if hardware.gpu_devices and hardware.ort_dml_ep_available:
                target = "GPU (DirectML)"
                reasons.append("GPU DirectML execution is available on host.")

        # Factor in User Objective
        if objective == "Maximum Battery Efficiency":
            reasons.append("Objective is Battery Efficiency: Snapdragon NPU operates at optimal energy per inference.")
        elif objective == "Maximum Accuracy":
            reasons.append("Objective is Maximum Accuracy: Preserving high precision layers during execution.")

        reasoning_text = " ".join(reasons)

        return {
            "selected_backend": target,
            "reasoning": reasoning_text,
            "weighted_npu_compatibility": compat.weighted_npu_score,
            "hardware_platform": hardware.snapdragon_model or hardware.cpu_model,
            "npu_status": hardware.npu_status
        }
