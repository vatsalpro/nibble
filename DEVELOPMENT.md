# SnapForge Developer Guide

## 1. Development Philosophy

SnapForge adheres to strict software engineering standards:
* **Authentic Measurements**: Never fabricate or simulate benchmark numbers without explicit labeling (`[Measured]`, `[Estimated]`, `[Simulated]`, `[Unavailable]`).
* **Decoupled Architecture**: All hardware-specific backends are isolated behind the `BaseBackend` interface so that developers on x86/x64 systems can develop and test without requiring Qualcomm hardware.
* **Safe Transformations**: All graph optimizations and fusions must preserve model semantics.
* **Zero UI Freezes**: All heavy computation (inspection, quantization, benchmarking) runs in background worker threads (`QThread`).

---

## 2. Setting Up the Development Environment

```powershell
# Clone or navigate to the repository
cd E:\snapdragon

# Create and activate a virtual environment (optional)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime dependencies
python -m pip install -r requirements.txt

# Verify installation with test suite
python -m unittest discover tests -v
```

---

## 3. Adding a New Operator to the Snapdragon Capability Registry

To register or update an operator's compatibility for Qualcomm Hexagon NPU or Adreno GPU:
1. Open [`app/models/compatibility.py`](file:///E:/snapdragon/app/models/compatibility.py).
2. Locate `QUALCOMM_CAPABILITY_REGISTRY`.
3. Add or update the operator mapping:
```python
"NewOperator": OpCapability(
    npu=SupportLevel.SUPPORTED,      # SUPPORTED, PARTIAL, UNSUPPORTED, or UNKNOWN
    gpu=SupportLevel.SUPPORTED,
    cpu=SupportLevel.SUPPORTED,
    npu_constraints="INT8 and FP16 accelerated on Hexagon HTP v68+.",
    gpu_constraints="Supported via DirectML."
),
```

---

## 4. Adding a New Execution Backend

To implement a new execution target (e.g. Vulkan, WebGPU, or custom hardware accelerator):
1. Create a new module in `app/backends/`.
2. Inherit from `BaseBackend` in [`app/backends/base_backend.py`](file:///E:/snapdragon/app/backends/base_backend.py).
3. Implement the required interface:
```python
from app.backends.base_backend import BaseBackend

class CustomBackend(BaseBackend):
    def __init__(self):
        super().__init__(name="Custom Accelerator")

    def get_device_info(self): ...
    def get_capabilities(self): ...
    def is_available(self): ...
    def validate_model(self, model_path): ...
    def load_model(self, model_path, options=None): ...
    def run_inference(self, input_feed): ...
    def benchmark(self, model_path, warmup_runs=10, measured_runs=100): ...
```
4. Register the new backend in `app/core/benchmark_manager.py` and `app/backends/__init__.py`.

---

## 5. Running Tests

SnapForge has a complete test suite:
```powershell
# Run all unit and integration tests
python -m unittest discover tests -v

# Run individual test modules
python -m unittest tests/test_model_inspector.py
python -m unittest tests/test_compatibility.py
python -m unittest tests/test_optimization.py
python -m unittest tests/test_benchmark.py
python -m unittest tests/test_pipeline_integration.py
```
