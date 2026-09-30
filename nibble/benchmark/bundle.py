"""
Benchmark Bundle Exporter for Nibble.
Generates self-contained, reproducible benchmark artifacts:
- result.json: Full machine-readable environment, hardware, hashes, and distribution statistics.
- summary.md: Clear Markdown summary with provenance, before vs after tables, and deployment recommendations.
Artifacts are saved in reports/benchmark_<timestamp>_<model_name>/
"""

import os
import sys
import json
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from nibble.utils.hashing import calculate_file_sha256
from nibble.optimizer.recommender import OptimizationRecommender, OptimizationScorecard


def create_benchmark_bundle(
    original_model_path: str | Path,
    optimized_model_path: str | Path,
    comparison_data: Dict[str, Any],
    provider: str = "CPUExecutionProvider",
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
    warmup_runs: int = 10,
    measured_runs: int = 50,
    batch_size: int = 1,
    reports_dir: str | Path = "reports",
) -> Path:
    """Generate self-contained benchmark artifacts (result.json and summary.md)."""
    from app.core.hardware_manager import HardwareManager

    orig_path = Path(original_model_path)
    opt_path = Path(optimized_model_path)
    base_reports_dir = Path(reports_dir)
    base_reports_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    ts_folder = now_utc.strftime("%Y-%m-%d_%H%M%S")
    model_slug = orig_path.stem
    bundle_dir = base_reports_dir / f"benchmark_{ts_folder}_{model_slug}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hardware and Environment metadata
    hw = HardwareManager.get_hardware_profile()
    orig_sha256 = calculate_file_sha256(orig_path)
    opt_sha256 = calculate_file_sha256(opt_path)

    orig_stats = comparison_data.get("original_stats", {})
    opt_stats = comparison_data.get("optimized_stats", {})
    accuracy_info = comparison_data.get("accuracy", {})

    orig_med = float(orig_stats.get("median_ms", comparison_data.get("original_median_ms", 0.0)))
    opt_med = float(opt_stats.get("median_ms", comparison_data.get("optimized_median_ms", 0.0)))
    orig_fps = float(orig_stats.get("throughput_fps", comparison_data.get("original_throughput_fps", 0.0)))
    opt_fps = float(opt_stats.get("throughput_fps", comparison_data.get("optimized_throughput_fps", 0.0)))

    orig_size_bytes = orig_path.stat().st_size if orig_path.exists() else 0
    opt_size_bytes = opt_path.stat().st_size if opt_path.exists() else 0

    cosine_sim = float(accuracy_info.get("overall_cosine_similarity", 1.0))
    top_pred_pres = str(accuracy_info.get("top_prediction_preserved", "YES"))
    contains_nan_inf = bool(accuracy_info.get("contains_nan_or_inf", False))

    # Evaluate honest scorecard
    scorecard = OptimizationRecommender.evaluate(
        model_name=orig_path.name,
        optimization_type=opt_path.stem.split("_")[-1].upper() if "_" in opt_path.stem else "Optimized",
        original_size_bytes=orig_size_bytes,
        optimized_size_bytes=opt_size_bytes,
        original_latency_ms=orig_med,
        optimized_latency_ms=opt_med,
        original_throughput_fps=orig_fps,
        optimized_throughput_fps=opt_fps,
        cosine_similarity=cosine_sim,
        top_prediction_preserved=top_pred_pres,
        contains_nan_or_inf=contains_nan_inf,
        execution_provider=provider,
    )

    # 2. Build JSON bundle
    result_dict = {
        "benchmark_id": f"benchmark_{ts_folder}_{model_slug}",
        "timestamp_utc": now_utc.isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "os_name": hw.os_name,
            "os_release": hw.os_release,
            "os_build": hw.os_build,
            "platform": platform.platform(),
            "cpu_vendor": hw.cpu_vendor,
            "cpu_model": hw.cpu_model,
            "cpu_arch": hw.cpu_arch,
            "cpu_cores_physical": hw.cpu_cores_physical,
            "cpu_cores_logical": hw.cpu_cores_logical,
            "gpu_devices": hw.gpu_devices,
            "snapdragon_npu_detected": hw.npu_present,
            "snapdragon_npu_status": hw.npu_status,
            "total_ram_gb": hw.total_ram_gb,
        },
        "execution_configuration": {
            "execution_provider": provider,
            "intra_op_threads": intra_op_threads if intra_op_threads > 0 else "Auto",
            "inter_op_threads": inter_op_threads if inter_op_threads > 0 else "Auto",
            "batch_size": batch_size,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
        },
        "models": {
            "original": {
                "file_name": orig_path.name,
                "file_path": str(orig_path.resolve()),
                "file_size_bytes": orig_size_bytes,
                "file_size_mb": round(orig_size_bytes / (1024 * 1024), 3),
                "sha256": orig_sha256,
            },
            "optimized": {
                "file_name": opt_path.name,
                "file_path": str(opt_path.resolve()),
                "file_size_bytes": opt_size_bytes,
                "file_size_mb": round(opt_size_bytes / (1024 * 1024), 3),
                "sha256": opt_sha256,
            },
        },
        "metrics": {
            "original": orig_stats,
            "optimized": opt_stats,
            "comparison": {
                "latency_reduction_pct": comparison_data.get("latency_reduction_pct", 0.0),
                "speedup_factor": comparison_data.get("speedup_factor", 1.0),
                "size_reduction_pct": comparison_data.get("size_reduction_pct", 0.0),
            },
            "accuracy": accuracy_info,
        },
        "scorecard": scorecard.to_dict(),
    }

    # Write result.json
    result_json_path = bundle_dir / "result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    # 3. Build summary.md
    summary_md_content = f"""# Nibble Benchmark Summary: {orig_path.name}

**Run Timestamp (UTC):** {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Execution Target:** `{provider}` (Threads: Intra={intra_op_threads or 'Auto'}, Inter={inter_op_threads or 'Auto'})  
**Recommendation Grade:** **{scorecard.recommendation_grade}**

---

## 1. Reproducibility & Integrity Hashes
| Asset | File Name | Size (MB) | SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **Original Model** | `{orig_path.name}` | {round(orig_size_bytes / (1024*1024), 2)} MB | `{orig_sha256}` |
| **Optimized Model** | `{opt_path.name}` | {round(opt_size_bytes / (1024*1024), 2)} MB | `{opt_sha256}` |

---

## 2. Hardware Environment
* **CPU:** {hw.cpu_model} ({hw.cpu_arch}, {hw.cpu_cores_logical} logical cores)
* **GPU Devices:** {', '.join(hw.gpu_devices) if hw.gpu_devices else 'None detected'}
* **Snapdragon NPU:** {hw.npu_status}
* **Total System RAM:** {hw.total_ram_gb:.1f} GB
* **Platform:** {platform.platform()} (Python {sys.version.split()[0]})

---

## 3. Empirical Latency & Throughput Comparison
| Metric | Original | Optimized | Delta / Impact | Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **Median Latency** | `{orig_med:.3f} ms` | `{opt_med:.3f} ms` | **{comparison_data.get('latency_reduction_pct', 0.0):+.1f}%** | `[MEASURED]` |
| **P95 Latency** | `{orig_stats.get('p95_ms', 0.0):.3f} ms` | `{opt_stats.get('p95_ms', 0.0):.3f} ms` | — | `[MEASURED]` |
| **Min / Max Latency** | `{orig_stats.get('min_ms', 0.0):.3f} / {orig_stats.get('max_ms', 0.0):.3f} ms` | `{opt_stats.get('min_ms', 0.0):.3f} / {opt_stats.get('max_ms', 0.0):.3f} ms` | — | `[MEASURED]` |
| **Throughput (FPS)** | `{orig_fps:.1f} FPS` | `{opt_fps:.1f} FPS` | **{comparison_data.get('speedup_factor', 1.0):.2f}x** | `[MEASURED]` |
| **Storage Footprint** | `{round(orig_size_bytes / (1024*1024), 2)} MB` | `{round(opt_size_bytes / (1024*1024), 2)} MB` | **{comparison_data.get('size_reduction_pct', 0.0):+.1f}%** | `[MEASURED]` |

---

## 4. Numerical Accuracy & Fidelity
* **Cosine Output Similarity:** `{cosine_sim:.5f}` ({accuracy_info.get('fidelity_grade', 'N/A')})
* **Relative Error:** `{accuracy_info.get('overall_relative_error', 0.0):.6f}`
* **Top Prediction Preserved:** `{top_pred_pres}`
* **Numerical Corruption:** `{'YES (NaN/Inf Detected)' if contains_nan_inf else 'None'}`

---

## 5. Objective Scorecard & Target Deployment
**Status:** `{'RECOMMENDED' if scorecard.is_recommended else 'QUALIFIED / NOT RECOMMENDED'}`

**Analysis:**
{scorecard.summary_narrative}

### Target Platform Readiness:
* **Host CPU ({hw.cpu_vendor}):** {'✅ ' if scorecard.target_recommendations.get('host_cpu', {}).get('recommended') else '❌ '}{scorecard.target_recommendations.get('host_cpu', {}).get('reason', '')}
* **Snapdragon Hexagon NPU:** {'✅ ' if scorecard.target_recommendations.get('snapdragon_npu', {}).get('recommended') else '❌ '}{scorecard.target_recommendations.get('snapdragon_npu', {}).get('reason', '')}
* **DirectML GPU:** {'✅ ' if scorecard.target_recommendations.get('directml_gpu', {}).get('recommended') else '❌ '}{scorecard.target_recommendations.get('directml_gpu', {}).get('reason', '')}

---
*Report generated automatically by Nibble AI Optimization Platform.*
"""

    summary_md_path = bundle_dir / "summary.md"
    summary_md_path.write_text(summary_md_content, encoding="utf-8")

    return bundle_dir
