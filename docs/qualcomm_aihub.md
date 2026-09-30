# Qualcomm AI Hub Cloud Validation & Snapdragon Hardware Profiling

Nibble provides an optional, independent **Qualcomm AI Hub Cloud Validation Backend**. This enables developers working on host development machines (such as AMD64 / x86_64 PCs) to compile, profile, and benchmark models on **actual physical Qualcomm Snapdragon hardware** in the cloud (such as Snapdragon X Elite, Snapdragon X Plus, and Snapdragon 8-series devices).

---

## 1. Strict Hardware Integrity & Zero-Fabrication Rule

Nibble strictly adheres to transparent hardware reporting:

1. **Zero Fabrication**: Nibble **never** fabricates Snapdragon or Qualcomm Hexagon NPU performance numbers. If remote cloud credentials are not configured or the network is unavailable, Nibble reports `NOT CONFIGURED` or falls back to static graph inspection (`STATIC_ANALYSIS`).
2. **Clear Host Separation**: The development machine (e.g. AMD Ryzen 5 8540U w/ Radeon 740M Graphics on Windows 11 AMD64) is explicitly identified as the local host. Host CPU or GPU benchmarks are **never** labeled as NPU benchmarks.
3. **Explicit Data Classification**: Every benchmark metric in Nibble is stamped with one of three explicit labels:
   * `LOCAL_MEASUREMENT`: Real empirical measurements recorded on the local host development PC (CPUExecutionProvider or DmlExecutionProvider).
   * `ACTUAL_DEVICE_MEASUREMENT`: Real physical hardware measurements obtained from a genuine remote Snapdragon device via the Qualcomm AI Hub Python SDK (`qai-hub`).
   * `STATIC_ANALYSIS`: Theoretical compatibility scores, node support counts, and partition estimates calculated via static ONNX graph inspection.
4. **Verified NPU Execution**: A remote job running on a Snapdragon device is **not** assumed to have utilized the NPU just because the remote hardware contains a Hexagon NPU. Nibble inspects the compute unit telemetry and per-layer acceleration breakdown to verify whether operators executed on the Hexagon Tensor Processor (HTP) or fell back to the device CPU.

---

## 2. Architecture Overview

```text
                                NIBBLE
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
            LOCAL ENGINE                    CLOUD ENGINE
            (Offline MVP)                (Qualcomm AI Hub)
                  │                               │
        ┌─────────┴─────────┐                     │
        │                   │                     ▼
   Host AMD CPU         DirectML GPU       qai-hub Python SDK
(CPUExecutionProvider)(DmlExecutionProvider)      │
        │                   │                     ▼
  [LOCAL_MEASUREMENT] [LOCAL_MEASUREMENT]  Physical Snapdragon
                                             Cloud Hardware
                                            (X Elite / X Plus)
                                                  │
                                                  ▼
                                            Hexagon NPU / HTP
                                                  │
                                                  ▼
                                      [ACTUAL_DEVICE_MEASUREMENT]
                                      Telemetry & Layer Breakdown
```

---

## 3. Configuration & Authentication

Qualcomm AI Hub requires an API token from [Qualcomm AI Hub](https://app.aihub.qualcomm.com).

Nibble supports three configuration methods, evaluated in order of precedence:

### Option A: Interactive / CLI Configuration (Recommended)
```powershell
python -m nibble aihub configure --token YOUR_QUALCOMM_API_TOKEN
```
This stores your credentials in standard format at `~/.qai_hub/client.ini`.

### Option B: Environment Variable
```powershell
$env:QAI_HUB_API_TOKEN = "YOUR_QUALCOMM_API_TOKEN"
```

### Option C: Nibble Desktop GUI
Navigate to **Settings** → **Qualcomm AI Hub Cloud Integration**, paste your token, and click **Test Connection**. Nibble will validate connectivity with Qualcomm servers and save your configuration.

---

## 4. CLI Commands Reference

### Check Connectivity & Status
```powershell
python -m nibble aihub status
```
Outputs:
* Host machine architecture and genuine CPU vendor (e.g. AMD64 / AMD)
* `qai-hub` SDK installation readiness
* Masked API token display (never prints plaintext tokens)
* Remote cloud connectivity status and available device count

### Query Available Snapdragon Hardware
```powershell
python -m nibble aihub devices
```
Filter by device model:
```powershell
python -m nibble aihub devices --filter "X Elite"
```
Lists target devices, operating system (Windows 11 ARM64 / Android), chipset, and rated Hexagon NPU TOPS (e.g. 45 TOPS).

### Upload ONNX Model
```powershell
python -m nibble aihub upload models/test_cnn.onnx
```
Pre-validates the ONNX graph with `onnx.checker`, calculates SHA-256 digest and file size, and uploads to Qualcomm AI Hub.

### Profile / Benchmark on Physical Snapdragon NPU
```powershell
python -m nibble aihub profile models/test_cnn.onnx --device "Snapdragon X Elite CRD"
```
Options:
* `--device <name>`: Target device (default: `Snapdragon X Elite CRD`)
* `--options <opts>`: Compiler/runtime flags (default: `--compute_unit npu`)
* `--no-wait`: Submit asynchronously without waiting for job completion

### Compare Local Host vs Cloud Snapdragon NPU
```powershell
python -m nibble compare-local-aihub models/test_cnn.onnx
```
Executes a side-by-side benchmark comparing:
1. `LOCAL_MEASUREMENT`: Real measured latency, throughput, and memory on the host AMD PC.
2. `ACTUAL_DEVICE_MEASUREMENT`: Real measured telemetry from the physical Snapdragon device (or `STATIC_ANALYSIS` if AI Hub is not configured).

---

## 5. Execution Target Verification Engine

Nibble's `AIHubValidator` inspects the execution payload downloaded from Qualcomm AI Hub:

| Verification Classification | Criteria | Meaning |
| :--- | :--- | :--- |
| **`NPU_CONFIRMED`** | >=80% compute or all layers executed on Hexagon HTP / NPU | Verified native acceleration on Qualcomm Hexagon NPU. |
| **`QNN_CONFIRMED`** | Graph partitioned across Hexagon NPU and CPU host | Hybrid execution; supported nodes ran on NPU, remaining on CPU. |
| **`CPU_EXECUTION`** | 100% compute or layers executed on device CPU | Model executed on Snapdragon CPU; zero NPU utilization. |
| **`GPU_EXECUTION`** | Compute executed on Qualcomm Adreno GPU | Accelerated via Adreno OpenCL / DirectML. |
| **`UNKNOWN`** | No telemetry breakdown provided in job output | Telemetry inconclusive; NPU execution cannot be verified. |

---

## 6. Artifact Exporter & Reproducibility

Every completed profiling run automatically generates reproducible audit artifacts under:

```text
reports/aihub/<job_id>/
  ├── raw_result.json          # Unmodified JSON payload from Qualcomm AI Hub
  ├── normalized_result.json   # Standardized Nibble telemetry schema
  └── summary.md               # Human-readable markdown audit report
```

### Example `normalized_result.json` Schema:
```json
{
  "source": "qualcomm_ai_hub",
  "device": "Snapdragon X Elite CRD",
  "model": "test_cnn.onnx",
  "model_sha256": "4b68e2f8c5b...",
  "runtime": "QNN / HTP",
  "execution_target": "NPU_CONFIRMED",
  "latency_ms": 1.42,
  "memory_mb": 14.8,
  "throughput_fps": 704.2,
  "precision": "FP16",
  "status": "COMPLETED",
  "is_actual_hardware_measurement": true,
  "verification": "NPU Execution Confirmed: 100.0% of compute executed directly on Qualcomm Hexagon NPU/HTP.",
  "layer_breakdown": {
    "npu_layers": 7,
    "cpu_layers": 0,
    "total_layers": 7
  },
  "timestamp_utc": "2026-09-30T15:20:00Z"
}
```

---

## 7. Verified Empirical Hardware Runs

The following profiling jobs were successfully submitted, executed, and verified on real physical Qualcomm hardware via Qualcomm AI Hub:

### Benchmark 1: FP32 CNN Baseline (`test_cnn.onnx`)
* **Job ID:** `j57eqe89p`
* **Target Hardware:** `Snapdragon X Elite CRD` (`Windows 11 ARM64`, `qualcomm-snapdragon-x-elite`, `45 TOPS`)
* **Remote Model ID:** `mmrg9x1xq`
* **Execution Target:** **`NPU_CONFIRMED`** (11/11 layers executed on Hexagon HTP / NPU, 0 CPU fallback)
* **Median Latency:** `0.0460 ms` (`46.0 µs`)
* **Throughput:** `21,739.1 FPS`
* **Peak Memory:** `28.21 MB`
* **Local AMD Host Comparison:** `0.0830 ms` (`12,062.7 FPS`) on AMD Ryzen 5 8540U CPU (`LOCAL_MEASUREMENT`) vs `0.0460 ms` (`21,739.1 FPS`) on Hexagon NPU (`ACTUAL_DEVICE_MEASUREMENT`)

### Benchmark 2: FP16 Optimized CNN (`test_cnn_fp16.onnx`)
* **Job ID:** `jgzl1lqx5`
* **Target Hardware:** `Snapdragon X Elite CRD` (`Windows 11 ARM64`, `qualcomm-snapdragon-x-elite`, `45 TOPS`)
* **Remote Model ID:** `mnwvwoo3q`
* **SHA-256 Digest:** `4ed9d72650c53371c01f02be51bca1340cf729902c7cad3857a62a6f36794b69`
* **Execution Target:** **`NPU_CONFIRMED`** (11/11 layers executed on Hexagon HTP / NPU, 0 CPU fallback)
* **Median Latency:** `0.0445 ms` (`44.5 µs`)
* **Throughput:** `22,471.9 FPS`
* **Peak Memory:** `28.26 MB`
* **Local AMD Host Comparison:** `0.0980 ms` (`10,245.9 FPS`) on AMD Ryzen 5 8540U CPU (`LOCAL_MEASUREMENT`) vs `0.0445 ms` (`22,471.9 FPS`) on Hexagon NPU (`ACTUAL_DEVICE_MEASUREMENT`)

