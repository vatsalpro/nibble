# Nibble — Post-MVP Comprehensive Technical Audit

**Date:** 2026-09-30  
**Host Hardware:** AMD Ryzen 5 8540U w/ Radeon 740M Graphics (Architecture: AMD64 / x86_64)  
**Host Operating System:** Microsoft Windows 11 Home (Build 26100)  
**Python Runtime:** Python 3.14.3 / ONNX 1.23.1 / ONNX Runtime 1.30.0  
**Scope:** Engineering review of Nibble codebase before post-MVP hardening.

---

## 1. Executive Summary & Current Architecture

Nibble has successfully validated its initial Minimum Viable Product (MVP) on this AMD development host through an automated 15-stage pipeline (`scripts/test_mvp.py`). The application adheres strictly to the **Iron Rule of Hardware Integrity**: it does not claim AMD is Qualcomm Snapdragon, does not simulate or fabricate NPU speedups, and clearly marks the Qualcomm Hexagon NPU as **`[Not detected / Unavailable on this hardware]`**.

### High-Level Architecture Overview

```text
nibble/ & app/
├── backends/             # Execution providers (CPU, DirectML, Qualcomm, Hybrid, NPU)
├── core/                 # Hardware detection, model management, benchmark & optimization orchestration
├── database/             # SQLite / SQLAlchemy persistent records (Projects, Runs, Benchmarks, Reports)
├── models/               # Model inspection, ONNX shape/type extraction, compatibility analysis
├── optimization/         # Fusion, graph simplifier, precision converters (FP16, INT8)
├── profiling/            # Accuracy validator, latency timers, memory trackers, system profiler
├── reports/              # PDF, Markdown, JSON, and CSV technical report generation
└── ui/                   # PySide6 Desktop GUI (Fusion of Win11 & macOS Settings style)
```

While the baseline pipeline executes end-to-end without crashing, this audit identifies critical mathematical inconsistencies, architectural ambiguities, unvalidated fallback behaviors, and technical debt that must be resolved to make Nibble scientifically credible, reproducible, and ready for future HP Snapdragon Copilot+ PC deployment.

---

## 2. In-Depth Component Audit

### Component 1: Benchmark Implementation & Mathematical Consistency

* **Current File:** `app/profiling/latency.py` (`LatencyTimer`, `LatencyStats`), `app/core/benchmark_manager.py`
* **Critical Mathematical Bug Discovered:**
  In `LatencyTimer.calculate_stats()` (lines 112–113):
  ```python
  # Throughput (FPS) = 1000.0 / mean_latency_ms
  throughput = (1000.0 / mean_val) if mean_val > 0 else 0.0
  ```
  The throughput metric is computed from the **arithmetic mean** (`mean_val`), while the primary latency metric presented to users and reports is the **median** (`median_ms`).
  In a real OS environment with occasional OS interrupts or cold thread scheduling:
  - Original Model: `Median = 0.082 ms`, but outlier runs pulled `Mean = 0.155 ms` $\rightarrow$ `Throughput = 6,445 FPS`.
  - Optimized Model: `Median = 0.091 ms` (11% slower median), but lower variance kept `Mean = 0.134 ms` $\rightarrow$ `Throughput = 7,427 FPS` (15% higher throughput!).
  - **Verdict:** An observer or academic evaluator sees higher median latency accompanied by higher reported throughput for single-sample inference, which appears mathematically contradictory.
* **Missing Features:**
  - Throughput does not take `batch_size` into account: $\text{FPS} = \frac{\text{batch\_size} \times 1000.0}{\text{latency\_ms}}$.
  - No dedicated calculation module with rigorous unit tests for percent change, speedup ratio, latency reduction, and P95.
  - Raw sample latencies are saved in memory but not persisted in the export JSON for reproducible verification.

### Component 2: Execution Backends & Provider Verification

* **Files:** `app/backends/base_backend.py`, `cpu_backend.py`, `directml_backend.py`, `gpu_backend.py`, `npu_backend.py`, `hybrid_backend.py`, `qualcomm_backend.py`
* **DirectML Backend (`directml_backend.py`):**
  - **Bug:** `DirectMLBackend.benchmark()` calls `with timer.time_scope():` and `timer.get_summary_stats()`, which do not exist on `LatencyTimer` (would cause runtime `AttributeError` if executed).
  - **Defect:** `is_available()` merely checks `"DmlExecutionProvider" in ort.get_available_providers()`. It never actually attempts a test session initialization or small dummy inference to confirm DirectX 12 hardware driver compatibility.
  - **Silent Fallback Hazard:** When instantiating `InferenceSession`, passing `providers=["DmlExecutionProvider", "CPUExecutionProvider"]` allows ONNX Runtime to silently drop DirectML and run on CPU without notifying the caller.
* **Qualcomm Backend (`qualcomm_backend.py`):**
  - Correctly reports `is_available() -> False` on AMD and raises `SnapdragonExecutionUnavailableError`.
  - Needs verification that any explicit `--provider qnn` CLI or GUI request refuses execution unconditionally without falling back.
* **Hybrid Backend (`hybrid_backend.py`) — Technical Debt / Fake Mode:**
  - **Major Finding:** `HybridBackend` claims to be `"Hybrid (Snapdragon NPU + CPU)"`, but its `run_inference()` and `benchmark()` methods simply route 100% of the execution to `CPUBackend`.
  - Calling this "Hybrid" is misleading. Per Section 22 of requirements, this must be either removed or clearly designated as `[Experimental / Not Implemented]` with real graph partitioning reserved for future work.

### Component 3: CPU Threading Control

* **Current File:** `app/backends/cpu_backend.py`
* **Defect:**
  ```python
  num_threads = min(8, psutil.cpu_count(logical=False) or 4)
  sess_opts.intra_op_num_threads = num_threads
  sess_opts.inter_op_num_threads = 2
  ```
  - Thread counts are hardcoded.
  - The CLI and GUI do not expose `--intra-op-threads` or `--inter-op-threads`.
  - Benchmarks do not record the active thread configuration, preventing fair cross-machine comparisons.

### Component 4: Compatibility Scoring & Terminology

* **Current File:** `app/models/compatibility.py`
* **Issues:**
  - Displays `"Snapdragon NPU Compatibility: 100.0%"`. Although tagged static analysis, the phrasing can easily be misread as actual NPU execution readiness.
  - Must be explicitly renamed to **`Static Snapdragon NPU Compatibility Estimate`** with the mandatory caveat:
    *"Static analysis only. Actual Snapdragon NPU execution has not been tested."*
  - Dual metrics must be made explicit:
    1. **Operator Coverage:** $\frac{\text{supported\_ops}}{\text{total\_ops}} \times 100\%$
    2. **Estimated Compute Coverage:** $\frac{\text{supported\_flops}}{\text{total\_flops}} \times 100\%$
  - In `ModelInspector._estimate_node_flops`: magic fallbacks (e.g. `return 1000`, `count * 9`) must be replaced with strict compute modeling or marked `"Unknown compute cost"`.

### Component 5: Optimization Evaluation & "Should Optimize?" Engine

* **Current Files:** `app/core/optimization_manager.py`, `app/optimization/quantization.py`, `app/optimization/graph_optimizer.py`
* **Defects:**
  - Optimization is currently treated as an unconditional success: converting FP32 to FP16 is celebrated as a "47% size reduction" even when CPU execution latency slows down by 11%.
  - No mechanism evaluates whether an optimization was actually beneficial for a given execution target.
  - Missing an analytical scorecard comparing size, latency, fidelity, and hardware fit.
  - Missing the `nibble/optimizer/recommender.py` engine that evaluates expected benefits vs risks and provides conditional advice (e.g., *"FP16 reduced model size by 47% but is 11% slower on CPU due to lack of native FP16 vector compute on this x86_64 host; recommended for memory-constrained targets or Snapdragon NPU deployment"*).

### Component 6: Reproducibility & Model Provenance

* **Defects:**
  - Models do not have their SHA-256 cryptographic hashes calculated or stored. If a model file is modified or overwritten on disk, baseline comparisons become invalid.
  - Test inputs in `AccuracyValidator` and benchmarking use pseudo-random generators without a shared, fixed seed, introducing run-to-run jitter.
  - Benchmark outputs are saved into database tables, but lack self-contained, reproducible artifact folders containing `result.json` and `summary.md`.

### Component 7: Accuracy & Fidelity Edge Cases

* **Current File:** `app/profiling/accuracy.py`
* **Defects:**
  - Does not check for `NaN` or `Inf` in output tensors before computing norms or dot products (can produce silent `nan` cosine similarity).
  - Zero-magnitude output vectors are not safely handled.
  - No relative error metric ($\frac{\|y - \hat{y}\|}{\|y\|}$) is calculated.
  - No top-1 class agreement preservation check ($\text{argmax}(y) == \text{argmax}(\hat{y})$) for classification outputs.

### Component 8: Security & File Integrity

* **Current File:** `app/models/model_inspector.py`
* **Defects:**
  - `torch.load(..., weights_only=False)` is used when inspecting PyTorch models, which is an arbitrary code execution risk.
  - ONNX file ingestion lacks strict extension whitelisting, file path traversal sanitization, and corrupted payload trapping.

### Component 9: Logging & Diagnostics

* **Current State:**
  - `print()` statements scattered throughout scripts.
  - No structured logging framework with `DEBUG`, `INFO`, `WARNING`, and `ERROR` levels.
  - No dedicated persistent log file (`logs/nibble.log`).
  - Python 3.12+ `datetime.utcnow()` deprecation warnings emitted in SQLAlchemy schemas and report timestamps.

### Component 10: CLI & Developer Workflow

* **Current State:**
  - Unified entrypoint `python -m nibble` exists and supports `hardware`, `analyze`, `optimize`, `benchmark`, and `test-mvp`.
  - Missing `python -m nibble compare <orig.onnx> <opt.onnx>`.
  - Missing `python -m nibble providers` (displaying active vs unavailable providers).
  - Missing `--intra-op-threads`, `--inter-op-threads`, and `--provider directml` options in benchmark command.

### Component 11: GUI Dashboard Provenance Badging

* **Current File:** `app/ui/dashboard.py`, `hardware.py`, `main_window.py`
* **Requirement:**
  - Must visually distinguish between:
    - `[MEASURED]` (Real CPU/GPU execution)
    - `[ESTIMATED]` (Analytical compute/FLOPs models)
    - `[STATIC ANALYSIS]` (Snapdragon compatibility predictions)
    - `[NOT AVAILABLE]` (Hexagon NPU on AMD host)
    - `[NOT TESTED]` (Unrun benchmarks)
  - Must remove fake Hybrid execution from selection until real subgraph partitioning is implemented.

---

## 3. Prioritized Engineering Action Plan

| Phase | Milestone | Files Involved |
| :--- | :--- | :--- |
| **Phase 1** | **Benchmark Math & Calculation Module** | Create `nibble/benchmark/metrics.py`, update `app/profiling/latency.py`, add unit tests |
| **Phase 2** | **Reproducibility, Model Hashing & CPU Threading** | Add SHA-256 hashing, expose `--intra-op-threads` / `--inter-op-threads`, store metadata JSON |
| **Phase 3** | **DirectML Hardening & Provider Verification** | Fix `DirectMLBackend`, add initialization test, provider verification, and fallback detection |
| **Phase 4** | **Static Compatibility Scoring & Terminology** | Rename to `Static Snapdragon NPU Compatibility Estimate`, split Operator vs Compute coverage |
| **Phase 5** | **Optimization Scorecard & Recommender Engine** | Create `nibble/optimizer/recommender.py`, conditional optimization evaluations |
| **Phase 6** | **Accuracy Validation Hardening** | Add NaN/Inf checks, relative error, and top-1 argmax preservation in `app/profiling/accuracy.py` |
| **Phase 7** | **Disable Fake Hybrid Mode & Prep Partitioning** | Mark Hybrid as `Experimental / Not Implemented`, architect clean `GraphPartitioner` interface |
| **Phase 8** | **CLI Expansion & Structured Logging** | Implement `compare`, `providers`, setup `logs/nibble.log`, eliminate `utcnow()` deprecations |
| **Phase 9** | **GUI Badging & Report Enhancements** | Add `[MEASURED]` / `[STATIC ANALYSIS]` badges, self-contained benchmark report folders |
| **Phase 10**| **Snapdragon Validation Checklist & Docs** | Create `docs/snapdragon_validation.md`, update `README.md`, run full test suite |

---

## 4. Verification & Defense Statement

Upon completion of these hardening steps, Nibble will be technically defensible before any engineering committee or evaluator:
- It **never claims** AMD is Snapdragon.
- It **never claims** NPU execution occurred when it did not.
- Its throughput and latency numbers are **mathematically harmonious and verifiable**.
- Its optimization advice is **empirically substantiated** by honest trade-off scorecards.
- Its Snapdragon QNN path is **cleanly isolated and fully documented** with an actionable validation procedure for real Qualcomm Copilot+ PC hardware.
