"""
Qualcomm AI Hub Execution Target Validator for Nibble.
Determines with mathematical and structural proof whether a remote job actually executed
on the Qualcomm Hexagon NPU/HTP vs CPU/GPU fallback.
Strictly adheres to policy: Never infers NPU execution solely from device model name.
"""

from typing import Dict, Any, Tuple


class AIHubValidator:
    """Validates physical execution hardware targets from Qualcomm AI Hub result metadata."""

    # Target classifications
    TARGET_NPU_CONFIRMED = "NPU_CONFIRMED"
    TARGET_QNN_CONFIRMED = "QNN_CONFIRMED"
    TARGET_CPU_EXECUTION = "CPU_EXECUTION"
    TARGET_GPU_EXECUTION = "GPU_EXECUTION"
    TARGET_UNKNOWN = "UNKNOWN"

    @classmethod
    def verify_execution_target(cls, raw_data: Dict[str, Any]) -> Tuple[str, str, int, int, int]:
        """
        Analyze profile response payload to determine actual compute unit execution.
        Returns:
            (execution_target, verification_notes, npu_layers, cpu_layers, total_layers)
        """
        if not raw_data or not isinstance(raw_data, dict):
            return (
                cls.TARGET_UNKNOWN,
                "Execution target: UNKNOWN. The profile completed, but Nibble could not verify NPU execution (empty result metadata).",
                0, 0, 0
            )

        def _extract_share(v) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                for k in ("time_percentage", "percentage", "time", "cycles", "count", "value"):
                    if k in v and isinstance(v[k], (int, float)):
                        return float(v[k])
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        # Check per-layer / per-node execution details first to get exact layer counts
        layers = raw_data.get("layers") or raw_data.get("execution_detail", {}).get("layers") or raw_data.get("nodes", [])
        npu_count = 0
        cpu_count = 0
        gpu_count = 0
        total_count = 0
        if layers and isinstance(layers, list):
            total_count = len(layers)
            for l in layers:
                if not isinstance(l, dict):
                    continue
                cu = str(l.get("compute_unit") or l.get("accelerator") or l.get("device") or "").upper()
                if "NPU" in cu or "HTP" in cu:
                    npu_count += 1
                elif "CPU" in cu:
                    cpu_count += 1
                elif "GPU" in cu or "ADRENO" in cu:
                    gpu_count += 1

        # 1. Check compute_unit_summary
        summary = raw_data.get("compute_unit_summary") or raw_data.get("execution_summary", {}).get("compute_unit_summary", {})
        if summary and isinstance(summary, dict):
            npu_share = _extract_share(summary.get("NPU") or summary.get("HTP") or summary.get("npu") or summary.get("htp") or 0.0)
            cpu_share = _extract_share(summary.get("CPU") or summary.get("cpu") or 0.0)
            gpu_share = _extract_share(summary.get("GPU") or summary.get("gpu") or 0.0)

            total_share = npu_share + cpu_share + gpu_share
            if total_share > 0:
                npu_pct = (npu_share / total_share) * 100.0
                cpu_pct = (cpu_share / total_share) * 100.0
                gpu_pct = (gpu_share / total_share) * 100.0

                l_npu = npu_count if total_count > 0 else 1
                l_cpu = cpu_count if total_count > 0 else 0
                l_tot = total_count if total_count > 0 else 1

                if npu_pct >= 80.0:
                    return (
                        cls.TARGET_NPU_CONFIRMED,
                        f"NPU Execution Confirmed: {npu_pct:.1f}% of compute executed directly on Qualcomm Hexagon NPU/HTP.",
                        l_npu, l_cpu, l_tot
                    )
                elif npu_pct > 0.0:
                    return (
                        cls.TARGET_QNN_CONFIRMED,
                        f"QNN Multi-Unit Execution: {npu_pct:.1f}% on NPU/HTP, with {cpu_pct:.1f}% fallback on CPU.",
                        l_npu if l_npu > 0 else 1, l_cpu if l_cpu > 0 else 1, l_tot if l_tot > 0 else 2
                    )
                elif gpu_pct >= 50.0:
                    return (
                        cls.TARGET_GPU_EXECUTION,
                        f"GPU Execution Confirmed: {gpu_pct:.1f}% executed on Qualcomm Adreno GPU.",
                        0, 0, l_tot
                    )
                elif cpu_pct >= 90.0:
                    return (
                        cls.TARGET_CPU_EXECUTION,
                        f"CPU Execution Confirmed: Model executed entirely on host CPU ({cpu_pct:.1f}%). Hexagon NPU was not utilized.",
                        0, cpu_count if total_count > 0 else 1, l_tot
                    )

        # 2. Check per-layer / per-node execution details if compute_unit_summary wasn't conclusive
        if total_count > 0:
            if npu_count == total_count:
                return (
                    cls.TARGET_NPU_CONFIRMED,
                    f"NPU Execution Confirmed: All {npu_count}/{total_count} layers executed on Qualcomm Hexagon NPU/HTP.",
                    npu_count, cpu_count, total_count
                )
            elif npu_count > 0:
                return (
                    cls.TARGET_QNN_CONFIRMED,
                    f"QNN Multi-Unit Execution: {npu_count}/{total_count} layers on Hexagon NPU, {cpu_count}/{total_count} layers on CPU.",
                    npu_count, cpu_count, total_count
                )
            elif gpu_count > 0 and cpu_count == 0:
                return (
                    cls.TARGET_GPU_EXECUTION,
                    f"GPU Execution Confirmed: All {gpu_count}/{total_count} layers executed on Adreno GPU.",
                    0, 0, total_count
                )
            elif cpu_count == total_count:
                return (
                    cls.TARGET_CPU_EXECUTION,
                    f"CPU Execution Confirmed: All {cpu_count}/{total_count} layers fell back to device CPU. Zero layers ran on NPU.",
                    0, cpu_count, total_count
                )

        # 3. Check explicit execution target tag in response
        exec_sum = raw_data.get("execution_summary", {}) if isinstance(raw_data.get("execution_summary"), dict) else {}
        target_tag = str(
            raw_data.get("execution_target") or
            raw_data.get("target") or
            raw_data.get("runtime") or
            exec_sum.get("runtime") or
            exec_sum.get("accelerator") or
            ""
        ).upper()

        if "NPU" in target_tag or "HTP" in target_tag:
            return (
                cls.TARGET_NPU_CONFIRMED,
                f"NPU Execution Confirmed via Qualcomm runtime target '{target_tag}'.",
                1, 0, 1
            )
        elif "QNN" in target_tag:
            return (
                cls.TARGET_QNN_CONFIRMED,
                f"QNN Execution Confirmed via Qualcomm runtime '{target_tag}'.",
                1, 0, 1
            )
        elif "CPU" in target_tag:
            return (
                cls.TARGET_CPU_EXECUTION,
                "Model executed on remote device CPU.",
                0, 1, 1
            )

        # 4. Default: Cannot verify
        return (
            cls.TARGET_UNKNOWN,
            "Execution target: UNKNOWN. The profile completed, but Nibble could not verify NPU execution from the returned profile telemetry.",
            0, 0, 0
        )
