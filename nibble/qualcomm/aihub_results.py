"""
Qualcomm AI Hub Result Normalization and Artifact Exporter for Nibble.
Standardizes raw Qualcomm responses into Nibble internal schemas and exports:
- raw_result.json
- normalized_result.json
- summary.md
under reports/aihub/<job_id>/
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from nibble.qualcomm.aihub_models import AIHubProfileResult
from nibble.qualcomm.aihub_validator import AIHubValidator


class AIHubResultManager:
    """Manages normalization, storage, and presentation of Qualcomm AI Hub benchmark results."""

    @classmethod
    def normalize_profile_result(
        cls,
        raw_result: Dict[str, Any],
        job_id: str,
        device_name: str,
        model_name: str,
        model_sha256: str,
        precision: str = "Unknown",
        runtime: str = "QNN / HTP"
    ) -> Dict[str, Any]:
        """Normalize raw Qualcomm AI Hub profile payload into Nibble's strict schema."""
        target, notes, npu_l, cpu_l, tot_l = AIHubValidator.verify_execution_target(raw_result)

        # Extract timing (handling seconds, milliseconds, or microseconds from various hub schemas)
        import statistics
        timing = raw_result.get("execution_summary", {}) if isinstance(raw_result.get("execution_summary"), dict) else (raw_result.get("timing", {}) or raw_result)
        latency_ms = None
        if "all_inference_times" in timing and timing["all_inference_times"]:
            median_us = float(statistics.median(timing["all_inference_times"]))
            latency_ms = round(median_us / 1000.0, 4)
        elif "inference_time" in timing and timing["inference_time"] is not None:
            val = float(timing["inference_time"])
            if "us" in str(timing.get("time_unit", "")).lower() or val > 500:
                latency_ms = round(val / 1000.0, 4)
            else:
                latency_ms = round(val, 4)
        elif "estimated_inference_time" in timing and timing["estimated_inference_time"] is not None:
            val = float(timing["estimated_inference_time"])
            # Qualcomm AI Hub estimated_inference_time is reported in microseconds
            latency_ms = round(val / 1000.0, 4) if val > 20 else round(val, 4)
        elif "median_latency_ms" in raw_result and raw_result["median_latency_ms"] is not None:
            latency_ms = round(float(raw_result["median_latency_ms"]), 4)

        # Extract memory
        mem_info = raw_result.get("memory", {}) if isinstance(raw_result.get("memory"), dict) else (raw_result.get("peak_memory", {}) or raw_result)
        memory_mb = None
        if "estimated_inference_peak_memory" in timing and timing["estimated_inference_peak_memory"] is not None:
            memory_mb = round(float(timing["estimated_inference_peak_memory"]) / (1024.0 * 1024.0), 2)
        elif "peak_memory_bytes" in mem_info and mem_info["peak_memory_bytes"] is not None:
            memory_mb = round(float(mem_info["peak_memory_bytes"]) / (1024.0 * 1024.0), 2)
        elif "peak_memory_mb" in raw_result and raw_result["peak_memory_mb"] is not None:
            memory_mb = round(float(raw_result["peak_memory_mb"]), 2)

        fps = round(1000.0 / latency_ms, 1) if latency_ms and latency_ms > 0 else None

        normalized = {
            "source": "qualcomm_ai_hub",
            "device": device_name,
            "model": model_name,
            "model_sha256": model_sha256,
            "runtime": runtime,
            "execution_target": target,
            "latency_ms": latency_ms,
            "memory_mb": memory_mb,
            "throughput_fps": fps,
            "precision": precision,
            "status": raw_result.get("status", "COMPLETED"),
            "is_actual_hardware_measurement": True,
            "verification": notes,
            "layer_breakdown": {
                "npu_layers": npu_l,
                "cpu_layers": cpu_l,
                "total_layers": tot_l
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }
        return normalized

    @classmethod
    def save_job_artifacts(
        cls,
        job_id: str,
        raw_result: Dict[str, Any],
        normalized_result: Dict[str, Any],
        reports_dir: str | Path = "reports"
    ) -> Path:
        """Store raw, normalized, and summary artifacts in reports/aihub/<job_id>/."""
        target_dir = Path(reports_dir) / "aihub" / str(job_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. raw_result.json
        raw_file = target_dir / "raw_result.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(raw_result, f, indent=2, default=str)

        # 2. normalized_result.json
        norm_file = target_dir / "normalized_result.json"
        with open(norm_file, "w", encoding="utf-8") as f:
            json.dump(normalized_result, f, indent=2, default=str)

        # 3. summary.md
        summary_md = f"""# Qualcomm AI Hub Physical Device Benchmark

**Job ID:** `{job_id}`  
**Measurement Source:** `QUALCOMM_AI_HUB` (Actual Hardware Measurement)  
**Target Device:** `{normalized_result.get('device', 'Snapdragon Device')}`  
**Model:** `{normalized_result.get('model', 'Model')}`  
**SHA-256 Digest:** `{normalized_result.get('model_sha256', 'N/A')}`  
**Timestamp (UTC):** `{normalized_result.get('timestamp_utc', 'N/A')}`  

---

## 1. Physical Device Telemetry
* **Execution Target Verification:** **`{normalized_result.get('execution_target')}`**
* **Verification Detail:** {normalized_result.get('verification')}
* **Runtime / Engine:** `{normalized_result.get('runtime')}`
* **Precision Mode:** `{normalized_result.get('precision')}`
* **Median Latency:** `{normalized_result.get('latency_ms')} ms`
* **Throughput (FPS):** `{normalized_result.get('throughput_fps')} FPS`
* **Peak Memory:** `{normalized_result.get('memory_mb')} MB`

---

## 2. Layer Execution Distribution
* **Hexagon NPU/HTP Layers:** `{normalized_result.get('layer_breakdown', {}).get('npu_layers', 0)}`
* **Device CPU Fallback Layers:** `{normalized_result.get('layer_breakdown', {}).get('cpu_layers', 0)}`
* **Total Layers:** `{normalized_result.get('layer_breakdown', {}).get('total_layers', 0)}`

---
*Generated by Nibble — AI Optimization Studio.*
"""
        summary_file = target_dir / "summary.md"
        summary_file.write_text(summary_md, encoding="utf-8")

        return target_dir
