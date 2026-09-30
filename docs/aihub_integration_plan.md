# Nibble — Qualcomm AI Hub Cloud Integration Plan

## 1. Executive Summary & Objective

This document outlines the architectural and engineering specification for integrating **Qualcomm AI Hub** cloud validation into **Nibble — AI Optimization Studio**.

Nibble currently operates as a fully validated MVP on AMD Windows hardware (`AMD Ryzen 5 8540U`, Windows 11 `AMD64`). The existing local Qualcomm backend correctly detects the AMD environment and cleanly refuses execution with `SnapdragonExecutionUnavailableError`, preventing any fabrication of local NPU results.

The objective of this integration is to introduce **Qualcomm AI Hub as an independent, cloud-device validation backend**. This empowers developers on AMD/Intel host machines to submit real models to Qualcomm-hosted physical Snapdragon hardware (e.g. Snapdragon X Elite, Snapdragon X Plus, 8 Gen 2/3), profile real on-device NPU inference, and receive authentic, verifiable hardware measurements without pretending that the local host processor is an NPU.

---

## 2. Existing Architecture Audit

### 2.1 Backend Layer
* **`BaseBackend`** (`app/backends/base_backend.py`): Abstract interface requiring `is_available()`, `validate_model()`, `load_model()`, `run_inference()`, and `benchmark()`.
* **`CPUBackend`** (`app/backends/cpu_backend.py`): Standard ONNX Runtime CPU execution provider with intra/inter-op thread controls.
* **`DirectMLBackend`** (`app/backends/directml_backend.py`): GPU execution provider via DirectML (DirectX 12).
* **`QualcommBackend`** (`app/backends/qualcomm_backend.py`): Local Qualcomm QNN/HTP backend. Strictly isolated; returns `is_available() -> False` and raises `SnapdragonExecutionUnavailableError` on x86/AMD hardware.
* **`HybridBackend`** (`app/backends/hybrid_backend.py`): Disabled experimental stub returning `is_available() -> False`.

### 2.2 Benchmarking & Accuracy
* **`BenchmarkMetrics`** (`nibble/benchmark/metrics.py`): Mathematically synchronized percentile calculations ($FPS = 1000 / \text{median\_ms}$).
* **`AccuracyValidator`** (`app/profiling/accuracy.py`): NaN/Inf detection, relative L2 error, cosine similarity, and top-1 argmax preservation.
* **`OptimizationRecommender`** (`nibble/optimizer/recommender.py`): Size vs. latency vs. fidelity trade-off scorecard engine.
* **`BenchmarkBundle`** (`nibble/benchmark/bundle.py`): Self-contained artifact exporter (`result.json` + `summary.md`).

### 2.3 Hardware & Provenance
* **`HardwareManager`** (`app/core/hardware_manager.py`): Genuine Windows hardware discovery detecting CPU vendor, RAM, OS, GPUs, and NPU availability.
* **Provenance Standards**: Explicit labeling across CLI, reports, and GUI:
  * `[MEASURED]`: Physical local or cloud hardware timer measurements.
  * `[ESTIMATED]`: Memory/latency heuristics.
  * `[STATIC ANALYSIS]`: Offline ONNX graph analysis.
  * `[NOT AVAILABLE]`: Hardware or runtime unconfigured.
  * `[NOT TESTED]`: Target unexecuted.

---

## 3. Proposed Qualcomm AI Hub Architecture

The AI Hub cloud backend operates orthogonally to local execution backends:

```text
                                NIBBLE
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
           LOCAL BACKENDS                    CLOUD BACKEND
                 │                                 │
       ┌─────────┼─────────┐                       ▼
       │         │         │               Qualcomm AI Hub
      CPU    DirectML     QNN                      │
                            │                      ▼
                            ▼              Snapdragon Cloud Devices
                     (Local Refusal         (Snapdragon X Elite,
                      on AMD Host)           8 Gen 3, etc.)
                                                   │
                                                   ▼
                                           Real NPU Execution
                                           (HTP / QNN Runtime)
                                                   │
                                                   ▼
                                         Job Result Normalization
                                         & NPU Verification
                                                   │
                                                   ▼
                                             Nibble Reports
                                          [ACTUAL DEVICE MEASUREMENT]
```

### 3.1 Architectural Principles
1. **Zero Fabrication**: If AI Hub is unreachable, unauthenticated, or fails, report `NOT AVAILABLE` or `FAILED`. Never silently substitute local CPU measurements.
2. **Dependency Isolation**: AI Hub is completely optional. If `qai-hub` is absent or the machine is offline, the local AMD MVP continues to run 100% cleanly.
3. **Execution Target Verification**: Never infer NPU execution solely because the remote target device is Snapdragon. Inspect layer execution breakdowns to verify actual `NPU`/`HTP` utilization vs `CPU` fallback.
4. **Three Strict Data Classifications**:
   * `STATIC_ANALYSIS`: Offline Nibble operator coverage prediction.
   * `LOCAL_MEASUREMENT`: Measured on AMD Ryzen 5 8540U host.
   * `ACTUAL_DEVICE_MEASUREMENT`: Measured on physical remote Qualcomm AI Hub hardware.

---

## 4. Module Design (`nibble/qualcomm/`)

We will create a clean, dedicated package `nibble/qualcomm/`:

```text
nibble/qualcomm/
    ├── __init__.py
    ├── aihub_client.py        # Authentication, client wrapper, upload, profile, polling
    ├── aihub_models.py        # Data models (AIHubDevice, AIHubModel, AIHubJob, AIHubProfileResult)
    ├── aihub_results.py       # Normalized schemas & report bundle generators
    └── aihub_validator.py     # NPU vs CPU vs GPU execution target verifier
```

### 4.1 Responsibilities

#### `aihub_models.py`
Defines typed dataclasses:
* `AIHubDevice`: `device_id`, `name`, `os`, `chipset`, `attributes`, `is_available`.
* `AIHubModel`: `model_id`, `name`, `sha256`, `size_bytes`, `uploaded_at`.
* `AIHubJob`: `job_id`, `job_type`, `model_id`, `device_id`, `status` (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`), `url`.
* `AIHubProfileResult`: `job_id`, `device_name`, `runtime`, `execution_target`, `median_latency_ms`, `peak_memory_mb`, `estimated_fps`, `npu_layers_count`, `cpu_layers_count`, `raw_data`.
* `AIHubValidationResult`: `model_sha256`, `static_score_pct`, `actual_npu_verified`, `verification_status`.

#### `aihub_validator.py`
Determines true execution target from profile breakdown data:
* Statuses: `NPU_CONFIRMED`, `QNN_CONFIRMED`, `CPU_EXECUTION`, `GPU_EXECUTION`, `UNKNOWN`.
* Parses `compute_unit_summary` and layer execution logs:
  * If $\ge 80\%$ compute or layers executed on `NPU` or `HTP`: `NPU_CONFIRMED`.
  * If partially executed on NPU: `QNN_CONFIRMED` (with CPU fallback layer count).
  * If executed entirely on CPU: `CPU_EXECUTION` (honest notification to user).

#### `aihub_client.py`
* Supports authentication through official `qai_hub`:
  1. `QAI_HUB_API_TOKEN` environment variable.
  2. `~/.qai_hub/client.ini` configuration file.
  3. Programmatic token configuration: `qai_hub.Client(config=ClientConfig(api_token=...))`.
* Pre-upload ONNX validation:
  * Checks file existence, valid extension (`.onnx`), runs `onnx.checker.check_model`, computes SHA-256.
* Job orchestration:
  * `list_devices(filter_str=None)`
  * `upload_model(model_path, model_name=None)`
  * `submit_profile(model, device, options="")`
  * `poll_job(job_id, timeout_sec=300, interval_sec=5)`
  * `download_profile_result(job_id)`

#### `aihub_results.py`
* Formats normalized JSON and Markdown artifacts saved under:
  ```text
  reports/aihub/
      └── <job_id>/
            ├── raw_result.json
            ├── normalized_result.json
            └── summary.md
  ```
* Implements side-by-side comparison format between Local AMD and Snapdragon AI Hub.

---

## 5. Files to Add & Modify

### Files to Add:
1. `nibble/qualcomm/__init__.py`
2. `nibble/qualcomm/aihub_models.py`
3. `nibble/qualcomm/aihub_validator.py`
4. `nibble/qualcomm/aihub_results.py`
5. `nibble/qualcomm/aihub_client.py`
6. `tests/test_aihub_offline.py`: Mocked unit test suite testing parsing, validation, device discovery, and error handling without network access.
7. `tests/test_aihub_live.py`: Optional live test executing only when `QAI_HUB_API_TOKEN` is present in environment (skips otherwise).
8. `docs/qualcomm_aihub.md`: Comprehensive user guide and documentation.

### Files to Modify:
1. `nibble/__main__.py`:
   * Add `python -m nibble aihub status`
   * Add `python -m nibble aihub configure`
   * Add `python -m nibble aihub devices`
   * Add `python -m nibble aihub upload <model>`
   * Add `python -m nibble aihub profile <model> --device <dev>`
   * Add `python -m nibble aihub benchmark <model> --device <dev>`
   * Add `python -m nibble compare-local-aihub <model> [--device <dev>]`
2. `app/cli.py`: Expose corresponding CLI handler functions.
3. `app/ui/dashboard.py` & `app/ui/main_window.py`: Add Qualcomm AI Hub view/tab and connection status badge.
4. `requirements-snapdragon.txt`: Document `qai-hub>=0.56.0`.
5. `README.md`: Update with Snapdragon validation notes.

---

## 6. Authentication Requirements & Security
* API tokens will **never** be hardcoded or written to log files, version control, or generated reports.
* Configuration priority:
  1. `QAI_HUB_API_TOKEN` environment variable.
  2. `~/.qai_hub/client.ini` (standard Qualcomm AI Hub SDK config).
* Token masking: diagnostic outputs will only display masked representations (e.g. `qai_********************3a9f`).

---

## 7. Error & Network Failure Handling
* **Missing SDK (`ImportError`)**: Gracefully reported as "Qualcomm AI Hub SDK (`qai-hub`) is not installed in the current environment. Local AMD features remain fully active."
* **Missing Credentials (`qai_hub.client.UserError`)**: Clear prompt directing user to set `QAI_HUB_API_TOKEN` or run `qai-hub configure`.
* **Network Timeout / Connection Error**: Reports actionable message without crashing; falls back to offline guidance.
* **Device Unavailable**: Reports current device list instead of arbitrary failure.
* **Remote Job Failure**: Captures and persists remote job failure logs without fabricating metrics.

---

## 8. Testing Strategy
1. **Mocked Offline Tests (`tests/test_aihub_offline.py`)**:
   * Test device list formatting with simulated `qai_hub.Device` objects.
   * Test model pre-validation and SHA-256 calculation.
   * Test NPU verification engine with various layer compute profiles (`HTP`, `CPU`, `GPU`).
   * Test normalized result serialization and bundle generation.
   * Verify all 29 existing unit tests continue to pass with 0 regressions.
2. **Live Optional Tests (`tests/test_aihub_live.py`)**:
   * Guarded by `unittest.skipUnless(os.environ.get("QAI_HUB_API_TOKEN"), "QAI_HUB_API_TOKEN not configured")`.
3. **End-to-End Regression Validation**:
   * Run `python -m nibble test-mvp` to guarantee that the AMD pipeline remains intact.
