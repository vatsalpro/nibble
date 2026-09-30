"""
Qualcomm Snapdragon NPU Execution Backend for SnapForge.
Designed specifically for Qualcomm Hexagon NPU (HTP / DSP) on Snapdragon X-series PCs.
Probes for QNNExecutionProvider and Qualcomm QNN SDK.
Integrates real execution when available; provides clear diagnostics and explicit labeling
(Measured vs Estimated vs Unavailable) when hardware is not present.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import onnx
import onnxruntime as ort
from app.backends.base_backend import BaseBackend
from app.profiling.latency import LatencyTimer
from app.profiling.memory import MemoryTracker


class NPUBackend(BaseBackend):
    """Executes or compiles models for Qualcomm Hexagon NPU via QNN Execution Provider."""

    def __init__(self):
        super().__init__(name="Snapdragon NPU")
        from app.core.hardware_manager import HardwareManager
        self.session: Optional[ort.InferenceSession] = None
        self.hw_profile = HardwareManager.get_hardware_profile()
        self.has_qnn_ep: bool = "QNNExecutionProvider" in ort.get_available_providers()
        self.is_hexagon_present: bool = self.hw_profile.is_hexagon or (self.hw_profile.is_snapdragon and self.hw_profile.npu_present)

    def is_available(self) -> bool:
        """NPU is genuinely available only if Hexagon hardware and QNN EP are both present."""
        return self.is_hexagon_present and self.has_qnn_ep

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device_name": "Qualcomm Hexagon NPU" if self.hw_profile.is_snapdragon else self.hw_profile.npu_name,
            "architecture": "Qualcomm Hexagon Tensor Processor (HTP)",
            "npu_status": self.hw_profile.npu_status,
            "qnn_ep_installed": self.has_qnn_ep,
            "qnn_sdk_path": self.hw_profile.qnn_sdk_path,
            "is_snapdragon_host": self.hw_profile.is_snapdragon,
            "integration_requirements": (
                "To execute natively on Hexagon NPU: "
                "1. HP Snapdragon X Elite / X Plus PC running Windows 11 ARM64. "
                "2. Qualcomm Hexagon NPU driver installed. "
                "3. ONNX Runtime built with QNN Execution Provider (onnxruntime-qnn)."
            )
        }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "target_core": "Hexagon HTP (Hexagon Tensor Processor)",
            "compute_capacity": "Up to 45 TOPS on Snapdragon X Elite",
            "optimal_precisions": ["INT8 (QDQ)", "FP16 (Half)"],
            "execution_modes": ["Burst", "Sustained High Performance", "Power Saver"],
            "graph_compilation": "QNN Context Binary (.bin / .dlc)",
            "operator_support": "Optimized for Conv2D, DepthwiseConv, Gemm, MatMul, Relu, Softmax, LayerNorm"
        }

    def validate_model(self, model_path: str | Path) -> Tuple[bool, str]:
        if not self.is_available():
            return False, (
                f"Snapdragon NPU execution unavailable on this machine ({self.hw_profile.cpu_arch}). "
                "Hardware-dependent integration point: Requires Qualcomm Snapdragon X-series ARM64 device."
            )
        return True, "Model ready for Qualcomm QNN compilation."

    def load_model(self, model_path: str | Path, options: Optional[Dict[str, Any]] = None) -> ort.InferenceSession:
        if not self.is_available():
            raise RuntimeError(
                f"Cannot initialize NPU session: Qualcomm Hexagon NPU hardware is not present on this host ({self.hw_profile.cpu_arch}). "
                "Hardware-dependent integration point: Requires Snapdragon X-series PC with QNNExecutionProvider."
            )

        p = Path(model_path)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Qualcomm QNN Execution Provider provider options
        qnn_options = {
            "backend_path": "QnnHtp.dll",  # HTP backend for Hexagon
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3"
        }
        if options:
            qnn_options.update(options)

        self.session = ort.InferenceSession(
            str(p),
            sess_opts,
            providers=["QNNExecutionProvider"],
            provider_options=[qnn_options]
        )
        self.is_initialized = True
        return self.session

    def run_inference(self, input_feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.session:
            raise RuntimeError("NPU session not loaded.")
        output_names = [out.name for out in self.session.get_outputs()]
        outputs = self.session.run(output_names, input_feed)
        return dict(zip(output_names, outputs))

    def benchmark(
        self,
        model_path: str | Path,
        warmup_runs: int = 10,
        measured_runs: int = 100,
        sample_input: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Executes genuine benchmark on Hexagon NPU if present.
        If NPU is absent, returns an explicitly marked 'Unavailable' report with
        analytical estimates labeled strictly as 'Estimated'.
        """
        if self.is_available():
            # Real hardware execution on Snapdragon
            self.load_model(model_path)
            output_names = [out.name for out in self.session.get_outputs()]
            feed = sample_input if sample_input is not None else {}
            if not feed:
                for inp in self.session.get_inputs():
                    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                    feed[inp.name] = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)

            mem_tracker = MemoryTracker()
            timer = LatencyTimer()
            latencies_ms: List[float] = []

            for _ in range(max(1, warmup_runs)):
                _ = self.session.run(output_names, feed)

            for _ in range(max(1, measured_runs)):
                timer.start()
                _ = self.session.run(output_names, feed)
                dur = timer.stop()
                latencies_ms.append(dur)
                mem_tracker.sample()

            stats = LatencyTimer.calculate_stats(latencies_ms, warmup_runs=warmup_runs)

            return {
                "backend": "Qualcomm Hexagon NPU",
                "device_name": "Qualcomm Hexagon HTP",
                "label_type": "Measured",
                "status": "Available",
                "stats": stats.to_dict(),
                "peak_memory_mb": mem_tracker.get_peak_mb(),
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs
            }
        else:
            # Genuinely report Unavailable per Section 1, 3, 22, 41
            return {
                "backend": "Qualcomm Hexagon NPU",
                "device_name": "Qualcomm Hexagon NPU (Unavailable on host)",
                "label_type": "Unavailable",
                "status": "Unavailable",
                "hardware_integration_point": True,
                "error": (
                    f"Qualcomm Hexagon NPU execution is unavailable on this computer ({self.hw_profile.cpu_arch} architecture). "
                    "SnapForge requires a Qualcomm Snapdragon X Elite / X Plus PC with Qualcomm QNN Execution Provider installed. "
                    "Benchmark not run to avoid fabricating numbers."
                ),
                "required_action": "Deploy on an HP Snapdragon PC (e.g., HP OmniBook X) to measure real NPU performance.",
                "warmup_runs": 0,
                "measured_runs": 0,
                "stats": LatencyTimer.calculate_stats([]).to_dict()
            }
