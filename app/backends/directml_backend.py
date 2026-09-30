"""
DirectML Execution Backend for Nibble.
Targets Windows DirectX 12 / DirectML GPU acceleration (AMD Radeon, Intel Arc/Iris, NVIDIA, Qualcomm Adreno).
Genuinely tests session initialization without silently falling back to CPU.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import onnx
import onnxruntime as ort
from app.backends.base_backend import BaseBackend
from app.profiling.latency import LatencyTimer
from app.profiling.memory import MemoryTracker
from nibble.utils.hashing import calculate_file_sha256
from nibble.backends.provider_verifier import ProviderVerifier, ExecutionTrace


class DirectMLBackend(BaseBackend):
    """Executes models on GPU via ONNX Runtime DmlExecutionProvider."""

    # Status constants
    STATUS_AVAILABLE = "AVAILABLE"
    STATUS_UNAVAILABLE = "UNAVAILABLE"
    STATUS_INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"

    def __init__(self):
        super().__init__(name="DirectML GPU")
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, List[int]] = {}
        self.input_dtypes: Dict[str, str] = {}
        self.model_path: Optional[str] = None
        self.model_sha256: str = ""
        self._provider_installed = "DmlExecutionProvider" in ort.get_available_providers()
        self._init_tested = False
        self._init_success = False
        self._init_error_msg = ""
        self.execution_trace: Optional[ExecutionTrace] = None

    @classmethod
    def test_initialization(cls) -> Tuple[bool, str]:
        """Genuinely tests whether DirectML can compile and execute a micro-model."""
        if "DmlExecutionProvider" not in ort.get_available_providers():
            return False, "DmlExecutionProvider is not in ONNX Runtime available providers (requires onnxruntime-directml)."
        try:
            from onnx import helper, TensorProto
            inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
            out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
            node = helper.make_node("Relu", ["x"], ["y"], name="test_relu")
            graph = helper.make_graph([node], "dml_test", [inp], [out])
            model = helper.make_model(graph, ir_version=10)
            sess = ort.InferenceSession(model.SerializeToString(), providers=["DmlExecutionProvider"])
            res = sess.run(None, {"x": np.array([[1.0, -1.0]], dtype=np.float32)})
            if res is not None and len(res) > 0:
                return True, "DirectML execution provider initialized and verified successfully."
            return False, "DirectML test inference returned empty output."
        except Exception as e:
            return False, f"DirectML initialization test failed: {e}"

    def is_available(self) -> bool:
        if not self._provider_installed:
            return False
        if not self._init_tested:
            self._init_success, self._init_error_msg = self.test_initialization()
            self._init_tested = True
        return self._init_success

    def status(self) -> str:
        if not self._provider_installed:
            return self.STATUS_UNAVAILABLE
        if not self._init_tested:
            self._init_success, self._init_error_msg = self.test_initialization()
            self._init_tested = True
        if self._init_success:
            return self.STATUS_AVAILABLE
        return self.STATUS_INITIALIZATION_FAILED

    def reason(self) -> str:
        s = self.status()
        if s == self.STATUS_AVAILABLE:
            return "ONNX Runtime DirectML Execution Provider is installed and functional."
        if s == self.STATUS_UNAVAILABLE:
            return "onnxruntime-directml is not installed in the current Python environment."
        return f"DirectML provider failed to initialize: {self._init_error_msg}"

    @property
    def error_message(self) -> str:
        return self._init_error_msg or self.reason()

    def get_device_info(self) -> Dict[str, Any]:
        from app.core.hardware_manager import HardwareManager
        hw = HardwareManager.get_hardware_profile()
        device_name = hw.gpu_devices[0] if hw.gpu_devices else "DirectML Graphics Device"
        return {
            "device_name": device_name,
            "execution_provider": "DmlExecutionProvider" if self.is_available() else "None",
            "status": self.status(),
            "reason": self.reason(),
            "api": "DirectX 12 / DirectML"
        }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supported_precisions": ["FP32", "FP16"],
            "operator_support": "Standard ONNX operators via DirectML",
            "api": "DirectX 12 / DirectML"
        }

    def validate_model(self, model_path: str | Path) -> Tuple[bool, str]:
        if not self.is_available():
            return False, f"DirectML GPU unavailable: {self.reason()}"
        p = Path(model_path)
        if not p.exists():
            return False, f"Model file not found: {p}"
        try:
            onnx.checker.check_model(str(p))
            return True, "Model structure valid for DirectML GPU execution."
        except Exception as e:
            return False, f"Validation error: {e}"

    def load_model(self, model_path: str | Path, options: Optional[Dict[str, Any]] = None) -> ort.InferenceSession:
        if not self.is_available():
            raise RuntimeError(f"DirectML backend unavailable ({self.status()}): {self.reason()}")

        p = Path(model_path)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        device_id = 0
        if options and "device_id" in options:
            device_id = options["device_id"]

        # Note: Do not pass CPUExecutionProvider as secondary provider during explicit DirectML test
        # to prevent silent fallback to CPU execution.
        try:
            self.session = ort.InferenceSession(
                str(p),
                sess_options=sess_opts,
                providers=["DmlExecutionProvider"],
                provider_options=[{"device_id": str(device_id)}]
            )
        except Exception as e:
            raise RuntimeError(f"DirectML failed to load model without CPU fallback: {e}")

        self.model_path = str(p)
        self.model_sha256 = calculate_file_sha256(p)

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes = {inp.name: [dim if isinstance(dim, int) else 1 for dim in inp.shape] for inp in self.session.get_inputs()}
        self.input_dtypes = {inp.name: inp.type for inp in self.session.get_inputs()}

        # Verify provider actually loaded
        self.execution_trace = ProviderVerifier.verify(
            session=self.session,
            requested_provider="DmlExecutionProvider",
            model_path=p,
            device_name=self.get_device_info()["device_name"]
        )

        self.is_initialized = True
        return self.session

    def run_inference(self, input_feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.session:
            raise RuntimeError("Model session not loaded. Call load_model first.")
        raw_outputs = self.session.run(self.output_names, input_feed)
        return {name: out for name, out in zip(self.output_names, raw_outputs)}

    def benchmark(
        self,
        model_path: str | Path,
        warmup_runs: int = 10,
        measured_runs: int = 50,
        sample_input: Optional[Dict[str, np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError(f"Cannot benchmark on DirectML GPU ({self.status()}): {self.reason()}")

        self.load_model(model_path, options=options)
        if sample_input is None:
            np.random.seed(42)
            sample_input = {}
            for inp in self.session.get_inputs():
                shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
                if shape and batch_size > 1:
                    shape[0] = batch_size
                dtype = np.float32
                if "float16" in inp.type:
                    dtype = np.float16
                elif "int64" in inp.type:
                    dtype = np.int64
                sample_input[inp.name] = np.random.randn(*shape).astype(dtype)

        timer = LatencyTimer()
        mem_tracker = MemoryTracker()
        latencies_ms: List[float] = []

        # Warmup iterations
        for _ in range(max(1, warmup_runs)):
            self.session.run(self.output_names, sample_input)

        # Measured iterations
        for _ in range(max(1, measured_runs)):
            timer.start()
            self.session.run(self.output_names, sample_input)
            dur = timer.stop()
            latencies_ms.append(dur)
            mem_tracker.sample()

        stats = LatencyTimer.calculate_stats(
            latencies_ms,
            warmup_runs=warmup_runs,
            batch_size=batch_size
        )

        device_info = self.get_device_info()

        return {
            "backend": "DirectML GPU",
            "device_name": device_info["device_name"],
            "execution_provider": "DmlExecutionProvider",
            "label_type": "Measured",
            "model_path": str(Path(model_path).resolve()),
            "model_sha256": self.model_sha256,
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "stats": stats.to_dict(),
            "peak_memory_mb": mem_tracker.get_peak_mb(),
            "memory_delta_mb": mem_tracker.get_delta_mb(),
            "execution_trace": self.execution_trace.to_dict() if self.execution_trace else {}
        }
