"""
CPU Execution Backend for Nibble.
Executes models using ONNX Runtime CPUExecutionProvider with real measurements,
reproducible threading controls, and SHA-256 model provenance.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import onnx
import onnxruntime as ort
import psutil
from app.backends.base_backend import BaseBackend
from app.profiling.latency import LatencyTimer
from app.profiling.memory import MemoryTracker
from nibble.utils.hashing import calculate_file_sha256
from nibble.backends.provider_verifier import ProviderVerifier, ExecutionTrace


class CPUBackend(BaseBackend):
    """Executes models on host CPU via ONNX Runtime CPUExecutionProvider."""

    def __init__(self):
        super().__init__(name="CPU")
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, List[int]] = {}
        self.input_dtypes: Dict[str, str] = {}
        self.model_path: Optional[str] = None
        self.model_sha256: str = ""
        self.intra_op_threads: int = 0
        self.inter_op_threads: int = 0
        self.execution_trace: Optional[ExecutionTrace] = None

    def get_device_info(self) -> Dict[str, Any]:
        from app.core.hardware_manager import HardwareManager
        hw = HardwareManager.get_hardware_profile()
        return {
            "device_name": hw.cpu_model,
            "architecture": hw.cpu_arch,
            "cores_logical": psutil.cpu_count(logical=True) or 1,
            "cores_physical": psutil.cpu_count(logical=False) or 1,
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": self.intra_op_threads or "Auto",
            "inter_op_threads": self.inter_op_threads or "Auto"
        }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supported_precisions": ["FP32", "FP16", "INT8", "INT32", "INT64", "UINT8"],
            "operator_support": "Universal ONNX Operator Set (100%)",
            "threading": "Configurable intra-op and inter-op thread pools",
            "acceleration": "AVX2 / AVX-512 / ARM NEON SIMD"
        }

    def is_available(self) -> bool:
        return True

    def validate_model(self, model_path: str | Path) -> Tuple[bool, str]:
        p = Path(model_path)
        if not p.exists():
            return False, f"Model file not found: {p}"
        try:
            onnx.checker.check_model(str(p))
            return True, "Model structure valid for CPU execution."
        except Exception as e:
            return False, f"Model validation error: {e}"

    def load_model(self, model_path: str | Path, options: Optional[Dict[str, Any]] = None) -> ort.InferenceSession:
        p = Path(model_path)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Configure intra-op and inter-op threading
        physical_cores = psutil.cpu_count(logical=False) or 4
        intra = options.get("intra_op_threads") if options else None
        inter = options.get("inter_op_threads") if options else None

        if intra is None or str(intra).strip().lower() in ("auto", "0"):
            self.intra_op_threads = min(8, physical_cores)
        else:
            try:
                self.intra_op_threads = max(1, int(intra))
            except (ValueError, TypeError):
                self.intra_op_threads = physical_cores

        if inter is None or str(inter).strip().lower() in ("auto", "0"):
            self.inter_op_threads = 2
        else:
            try:
                self.inter_op_threads = max(1, int(inter))
            except (ValueError, TypeError):
                self.inter_op_threads = 2

        sess_opts.intra_op_num_threads = self.intra_op_threads
        sess_opts.inter_op_num_threads = self.inter_op_threads

        self.session = ort.InferenceSession(
            str(p),
            sess_opts,
            providers=["CPUExecutionProvider"]
        )
        self.model_path = str(p)
        self.model_sha256 = calculate_file_sha256(p)

        # Inspect inputs
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes = {}
        self.input_dtypes = {}

        for inp in self.session.get_inputs():
            shape = []
            for d in inp.shape:
                if isinstance(d, int) and d > 0:
                    shape.append(d)
                else:
                    shape.append(1)  # dynamic dimensions default to 1
            self.input_shapes[inp.name] = shape
            self.input_dtypes[inp.name] = inp.type

        # Generate execution trace
        self.execution_trace = ProviderVerifier.verify(
            session=self.session,
            requested_provider="CPUExecutionProvider",
            model_path=p,
            device_name=self.get_device_info()["device_name"]
        )

        self.is_initialized = True
        return self.session

    def create_synthetic_input(self, batch_size: int = 1, seed: Optional[int] = 42) -> Dict[str, np.ndarray]:
        """Generate deterministic input matching session signatures."""
        if seed is not None:
            np.random.seed(seed)
        feed = {}
        for name, shape in self.input_shapes.items():
            adj_shape = list(shape)
            if adj_shape and batch_size > 1:
                adj_shape[0] = batch_size

            dtype_str = self.input_dtypes.get(name, "tensor(float)")
            if "int64" in dtype_str:
                feed[name] = np.zeros(adj_shape, dtype=np.int64)
            elif "int32" in dtype_str:
                feed[name] = np.zeros(adj_shape, dtype=np.int32)
            elif "float16" in dtype_str:
                feed[name] = np.random.uniform(-1.0, 1.0, size=adj_shape).astype(np.float16)
            elif "bool" in dtype_str:
                feed[name] = np.ones(adj_shape, dtype=bool)
            else:
                feed[name] = np.random.uniform(-1.0, 1.0, size=adj_shape).astype(np.float32)
        return feed

    def run_inference(self, input_feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.session:
            raise RuntimeError("Model session not loaded. Call load_model() first.")
        outputs = self.session.run(self.output_names, input_feed)
        return dict(zip(self.output_names, outputs))

    def benchmark(
        self,
        model_path: str | Path,
        warmup_runs: int = 10,
        measured_runs: int = 50,
        sample_input: Optional[Dict[str, np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """Run real measured benchmark on CPU with consistent throughput math."""
        self.load_model(model_path, options=options)
        feed = sample_input if sample_input is not None else self.create_synthetic_input(batch_size=batch_size, seed=42)

        mem_tracker = MemoryTracker()
        timer = LatencyTimer()
        latencies_ms: List[float] = []

        # Warmup iterations (strictly discarded from measured stats)
        for _ in range(max(1, warmup_runs)):
            _ = self.session.run(self.output_names, feed)

        # Measured iterations
        for _ in range(max(1, measured_runs)):
            timer.start()
            _ = self.session.run(self.output_names, feed)
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
            "backend": "CPU",
            "device_name": device_info["device_name"],
            "execution_provider": "CPUExecutionProvider",
            "label_type": "Measured",
            "model_path": str(Path(model_path).resolve()),
            "model_sha256": self.model_sha256,
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "stats": stats.to_dict(),
            "peak_memory_mb": mem_tracker.get_peak_mb(),
            "memory_delta_mb": mem_tracker.get_delta_mb(),
            "execution_trace": self.execution_trace.to_dict() if self.execution_trace else {}
        }
