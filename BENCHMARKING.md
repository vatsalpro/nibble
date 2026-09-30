# SnapForge Benchmarking & Validation Methodology

## 1. Principles of Empirical Measurement

SnapForge adheres to rigorous scientific and engineering principles for model performance evaluation:
1. **Zero Fabrication**: No benchmark numbers are ever hardcoded or fabricated.
2. **Provenance Labels**: Every result in the UI, CLI, and exported PDF reports is tagged with its provenance:
   - **`[Measured]`**: Directly timed using high-resolution hardware monotonic timers during execution.
   - **`[Estimated]`**: Computed from analytical hardware cost models (FLOPs, memory bandwidth).
   - **`[Simulated]`**: Produced via software emulation or cycle simulator.
   - **`[Unavailable]`**: Unexposed or inaccessible on current hardware.

---

## 2. Timing Protocol

### 2.1 Warmup Phase
Neural network runtimes incur one-time overheads during initial inference:
- Kernel compilation and JIT compilation.
- Weight buffer staging into GPU/NPU memory.
- Dynamic frequency scaling (DVFS) transition to high-performance governor states.

SnapForge executes a configurable warmup phase (default: 10 iterations) prior to taking measurements.

### 2.2 Measurement Phase
- Timing is captured per iteration using `time.perf_counter_ns()`.
- Latencies are converted to fractional milliseconds ($10^{-3}\text{ s}$).
- Both warm-up and measured iterations are configurable in the UI and CLI.

---

## 3. Statistical Metrics

Given a collection of $N$ measured latencies $L = \{l_1, l_2, \dots, l_N\}$:

### 3.1 Median Latency ($L_{50}$)
The median is the primary performance KPI because it is robust against operating system background context switching spikes:
$$\text{Median} = \begin{cases} l_{(N+1)/2} & \text{if } N \text{ is odd} \\ \frac{1}{2}(l_{N/2} + l_{N/2 + 1}) & \text{if } N \text{ is even} \end{cases}$$

### 3.2 95th Percentile Latency ($L_{95}$)
Measures tail latency under load, representing 95% worst-case responsiveness for real-time applications.

### 3.3 Throughput (FPS)
Throughput measures inferences per second:
$$\text{Throughput} = \frac{1000}{\text{Mean Latency (ms)}}$$

### 3.4 Speedup Factor
$$\text{Speedup} = \frac{\text{Baseline Median Latency (ms)}}{\text{Optimized Median Latency (ms)}}$$

---

## 4. Numerical Accuracy & Fidelity Validation

Optimization must not silently degrade model accuracy. SnapForge evaluates numerical drift between the original FP32 baseline and the optimized INT8/FP16 model:

### 4.1 Cosine Similarity
Measures angular alignment of output tensors across all classes and spatial predictions:
$$\text{Cosine Similarity} = \frac{\mathbf{y}_{\text{orig}} \cdot \mathbf{y}_{\text{opt}}}{\|\mathbf{y}_{\text{orig}}\| \|\mathbf{y}_{\text{opt}}\|}$$
- $1.0000$: Perfectly identical numerical outputs.
- $\ge 0.9990$: High-fidelity quantization.
- $< 0.9500$: Significant drift requiring static calibration refinement.

### 4.2 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{M} \sum_{i=1}^M |y_{\text{orig}, i} - y_{\text{opt}, i}|$$

### 4.3 Maximum Absolute Difference
$$\text{Max Diff} = \max_i |y_{\text{orig}, i} - y_{\text{opt}, i}|$$

---

## 5. Live Telemetry Monitoring

During benchmark runs, the `SystemProfiler` samples hardware state at high frequency:
* **CPU Load**: Process CPU utilization and system-wide per-core load via `psutil`.
* **Process Memory**: Resident Set Size (RSS) and peak allocated memory.
* **Battery & Power**: State of charge (%) and AC power connectivity.
* **NPU Telemetry**: Real session status when running under Qualcomm QNN EP; clearly reported as `Unavailable` on non-Snapdragon host processors.
