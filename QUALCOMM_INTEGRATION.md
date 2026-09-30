# Qualcomm Snapdragon & AI Hub Integration Guide

## 1. Overview of Qualcomm Snapdragon AI Architecture

Qualcomm Snapdragon X-series processors (Snapdragon X Elite, Snapdragon X Plus) feature a heterogeneous compute architecture tailored for on-device AI:
1. **Qualcomm Hexagon NPU**: Dedicated neural processing unit featuring the Hexagon Tensor Processor (HTP) delivering up to 45 TOPS on Snapdragon X Elite. Optimized for INT8, INT16, and FP16 operations.
2. **Qualcomm Adreno GPU**: High-throughput floating point graphics accelerator accessible via DirectX 12 DirectML.
3. **Qualcomm Oryon CPU**: High-performance multi-core ARM64 CPU with vector NEON extensions, providing low-latency fallback for dynamic graph execution.

---

## 2. Windows 11 On Snapdragon Deployment Stack

```
+-------------------------------------------------------------+
|                      User Application                       |
|               (SnapForge Desktop / CLI Engine)              |
+-------------------------------------------------------------+
                              |
       +----------------------+----------------------+
       |                                             |
       v                                             v
+-----------------------------+       +-----------------------------+
|    ONNX Runtime QNN EP      |       |      DirectML Engine        |
|  (onnxruntime-qnn for ARM)  |       |  (onnxruntime-directml)     |
+-----------------------------+       +-----------------------------+
               |                                     |
               v                                     v
+-----------------------------+       +-----------------------------+
|    Qualcomm QNN Backend     |       |    Qualcomm Adreno Driver   |
|         (QnnHtp.dll)        |       |        (D3D12 / DXGI)       |
+-----------------------------+       +-----------------------------+
               |                                     |
               v                                     v
+-----------------------------+       +-----------------------------+
|    Qualcomm Hexagon NPU     |       |    Qualcomm Adreno GPU      |
|     (45 TOPS HTP Core)      |       |    (3.8 - 4.6 TFLOPS)       |
+-----------------------------+       +-----------------------------+
```

---

## 3. Configuring the Qualcomm QNN Execution Provider

SnapForge automatically detects and configures the `QNNExecutionProvider` in ONNX Runtime when running on a Qualcomm Snapdragon device.

### Recommended Provider Options for Hexagon HTP:
```python
qnn_options = {
    "backend_path": "QnnHtp.dll",                            # Hexagon Tensor Processor backend
    "htp_performance_mode": "burst",                         # Maximum clock and voltage state
    "htp_graph_finalization_optimization_mode": "3",         # Level 3 deep graph optimization
    "vtcm_size_in_mb": "8",                                  # Allocate Vector Tightly Coupled Memory
    "precision": "INT8"                                      # Peak throughput precision
}

session = ort.InferenceSession(
    "model.onnx",
    providers=["QNNExecutionProvider"],
    provider_options=[qnn_options]
)
```

---

## 4. Diagnostics: When NPU Shows "Unavailable"

If SnapForge displays **`Qualcomm Hexagon NPU: Unavailable`**, verify the following checklist:

1. **Target Hardware**:
   - Verify that your PC is powered by a Qualcomm Snapdragon X Elite (e.g. HP OmniBook X, HP EliteBook Ultra) or Snapdragon X Plus processor.
   - Run `snapforge hardware` to view detected CPU architecture.
2. **Architecture**:
   - Python must run natively as an **ARM64** binary on Windows 11 ARM64 (`platform.machine() == 'ARM64'`).
3. **Qualcomm NPU Driver**:
   - Check Windows Device Manager under *Neural Processors* or *System Devices* for `Qualcomm(R) Hexagon(TM) NPU`.
4. **ONNX Runtime QNN Package**:
   - Install the official Qualcomm QNN-enabled ONNX Runtime package:
     ```powershell
     python -m pip install onnxruntime-qnn
     ```
5. **Development Host Fallback**:
   - When running on an x86_64/AMD64 development machine, SnapForge operates in local development mode: local benchmarks execute on the host CPU/GPU labeled strictly as `[Measured]`, while generating Snapdragon-ready INT8 optimized models for deployment.

---

## 5. Qualcomm AI Hub Cloud Integration

SnapForge connects to Qualcomm AI Hub (`https://aihub.qualcomm.com/`) for cloud-based compilation and device profiling:
* To configure, obtain your API token from your Qualcomm AI Hub account.
* Set the environment variable:
  ```powershell
  $env:QAI_HUB_API_TOKEN = "your_qualcomm_api_token"
  ```
  or configure the token in SnapForge **Settings** view.
* If credentials are not configured, SnapForge explicitly reports:
  `"Qualcomm AI Hub integration unavailable — configure credentials."`
  without fabricating responses.
