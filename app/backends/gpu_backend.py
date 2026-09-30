"""
GPU Execution Backend for SnapForge.
Targets Windows DirectML (Qualcomm Adreno, AMD, Intel, NVIDIA) and CUDA.
Provides real measurements when execution provider is active,
or clearly reports 'Unavailable' with installation instructions.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import onnxruntime as ort
from app.backends.base_backend import BaseBackend
from app.profiling.latency import LatencyTimer
from app.profiling.memory import MemoryTracker


class GPUBackend(BaseBackend):
    """Executes models on GPU via DirectML or CUDA Execution Provider."""

    def __init__(self):
        super().__init__(name="GPU")
        self.session: Optional[ort.InferenceSession] = None
        self.active_provider: Optional[str] = None
        self.device_name = "Unknown GPU"
        self._check_provider()

    def _check_provider(self):
        from app.core.hardware_manager import HardwareManager
        hw = HardwareManager.get_hardware_profile()
        providers = ort.get_available_providers()

        if "DmlExecutionProvider" in providers:
            self.active_provider = "DmlExecutionProvider"
            self.device_name = hw.gpu_devices[0] if hw.gpu_devices else "DirectML Graphics Device"
        elif "CUDAExecutionProvider" in providers:
            self.active_provider = "CUDAExecutionProvider"
            self.device_name = "CUDA GPU"
        else:
            self.active_provider = None
            self.device_name = hw.gpu_devices[0] if hw.gpu_devices else "GPU (No EP available)"

    def is_available(self) -> bool:
        return self.active_provider is not None

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "execution_provider": self.active_provider or "None (DmlExecutionProvider missing)",
            "is_adreno": "adreno" in self.device_name.lower(),
            "status": "Available" if self.is_available() else "Unavailable (Requires onnxruntime-directml)"
        }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supported_precisions": ["FP32", "FP16"],
            "operator_support": "High (~95% standard ONNX operators via DirectML)",
            "api": "DirectX 12 / DirectML"
        }

    def validate_model(self, model_path: str | Path) -> Tuple[bool, str]:
        if not self.is_available():
            return False, "GPU Execution Provider not installed. Install onnxruntime-directml for Qualcomm Adreno / DirectML support."
        return True, "Model ready for GPU execution."

    def load_model(self, model_path: str | Path, options: Optional[Dict[str, Any]] = None) -> ort.InferenceSession:
        if not self.is_available():
            raise RuntimeError("GPU execution provider (DmlExecutionProvider) is not installed in Python.")

        p = Path(model_path)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(p),
            sess_opts,
            providers=[self.active_provider]
        )
        self.is_initialized = True
        return self.session

    def run_inference(self, input_feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.session:
            raise RuntimeError("GPU session not loaded.")
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
        if not self.is_available():
            return {
                "backend": "GPU",
                "device_name": self.device_name,
                "label_type": "Unavailable",
                "status": "Unavailable",
                "error": "DirectML / GPU Execution Provider not installed in current Python runtime. Install onnxruntime-directml to benchmark on Adreno GPU.",
                "warmup_runs": 0,
                "measured_runs": 0,
                "stats": LatencyTimer.calculate_stats([]).to_dict()
            }

        self.load_model(model_path)
        # Input synthesis
        input_names = [inp.name for inp in self.session.get_inputs()]
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
            "backend": "GPU (DirectML)",
            "device_name": self.device_name,
            "label_type": "Measured",
            "stats": stats.to_dict(),
            "peak_memory_mb": mem_tracker.get_peak_mb(),
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs
        }
