"""
Optimization Manager for SnapForge.
Orchestrates the complete optimization pipeline:
1. Operator Fusion (Conv + BatchNorm)
2. Graph Optimization (Constant folding, dead node removal)
3. Precision Quantization (FP16 or INT8 Dynamic/Static)
4. Audit logging and SQLite persistence.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from app.database.database import Database
from app.database.schema import OptimizationRun, ModelRecord, Project
from app.optimization.operator_fusion import OperatorFusionEngine
from app.optimization.graph_optimizer import GraphOptimizer
from app.optimization.quantization import QuantizationEngine, SyntheticCalibrationReader

OUTPUT_MODELS_DIR = Path("optimized_models").resolve()


class OptimizationManager:
    """Executes the multi-stage optimization pipeline."""

    @classmethod
    def run_optimization(
        cls,
        input_model_path: str | Path,
        project_id: int,
        model_id: int,
        target_backend: str = "Snapdragon NPU",
        strategy: str = "Balanced",
        precision: str = "INT8",
        enable_fusion: bool = True,
        enable_graph_opt: bool = True
    ) -> Dict[str, Any]:
        OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        in_path = Path(input_model_path)
        orig_size_bytes = in_path.stat().st_size

        log_lines: List[str] = [f"=== Optimization Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ==="]
        log_lines.append(f"Model: {in_path.name}")
        log_lines.append(f"Target Backend: {target_backend}")
        log_lines.append(f"Strategy: {strategy} | Target Precision: {precision}")

        current_path = in_path

        # Step 1: Operator Fusion
        if enable_fusion:
            fused_path = OUTPUT_MODELS_DIR / f"{in_path.stem}_fused.onnx"
            fusion_res = OperatorFusionEngine.fuse_conv_batchnorm(current_path, fused_path)
            if fusion_res["success"] and fusion_res["fused_count"] > 0:
                current_path = fused_path
                log_lines.append(f"[OK] Operator Fusion: Folded {fusion_res['fused_count']} BatchNorm node(s) into Conv layers.")
                for detail in fusion_res.get("fused_details", []):
                    log_lines.append(f"    - {detail}")
            else:
                log_lines.append("* Operator Fusion: No un-fused BatchNorm nodes found.")

        # Step 2: Graph Optimization
        if enable_graph_opt:
            opt_path = OUTPUT_MODELS_DIR / f"{in_path.stem}_graph_opt.onnx"
            graph_res = GraphOptimizer.optimize(current_path, opt_path, opt_level="BASIC")
            if graph_res["success"]:
                current_path = opt_path
                for entry in graph_res.get("log", []):
                    log_lines.append(f"    {entry}")

        # Step 3: Quantization
        final_optimized_path = OUTPUT_MODELS_DIR / f"{in_path.stem}_optimized_{precision.lower()}.onnx"
        quant_res = {}

        if precision.upper() == "FP16":
            quant_res = QuantizationEngine.quantize_fp16(current_path, final_optimized_path)
            log_lines.append(f"[OK] Precision Conversion: Converted model to FP16 ({quant_res['size_reduction_pct']}% size reduction).")
        elif precision.upper() == "INT8":
            # Attempt static QDQ if calibration is requested, or dynamic INT8
            try:
                cal_reader = SyntheticCalibrationReader(str(current_path), num_samples=10)
                quant_res = QuantizationEngine.quantize_int8_static(current_path, final_optimized_path, cal_reader)
                log_lines.append(f"[OK] Quantization: Static INT8 (QDQ) applied with calibration data ({quant_res['size_reduction_pct']}% size reduction).")
            except Exception as e:
                log_lines.append(f"* Static INT8 fallback to Dynamic INT8 ({e})")
                quant_res = QuantizationEngine.quantize_int8_dynamic(current_path, final_optimized_path)
                log_lines.append(f"[OK] Quantization: Dynamic INT8 applied ({quant_res['size_reduction_pct']}% size reduction).")
        else:  # FP32
            import shutil
            shutil.copy2(current_path, final_optimized_path)
            quant_res = {
                "size_reduction_pct": 0.0,
                "optimized_size_bytes": final_optimized_path.stat().st_size
            }
            log_lines.append("* Precision: Retained FP32 precision.")

        final_size_bytes = final_optimized_path.stat().st_size
        pct_reduction = round(((orig_size_bytes - final_size_bytes) / max(1, orig_size_bytes)) * 100.0, 1)

        # Validate optimized ONNX model structure
        try:
            import onnx
            onnx.checker.check_model(str(final_optimized_path))
            log_lines.append(f"[OK] Model Validation: onnx.checker.check_model passed for {final_optimized_path.name}")
        except Exception as e:
            log_lines.append(f"[WARNING] Validation warning for {final_optimized_path.name}: {e}")

        log_lines.append(f"=== Optimization Completed Successfully ===")
        log_lines.append(f"Original Size: {round(orig_size_bytes / (1024*1024), 2)} MB → Optimized: {round(final_size_bytes / (1024*1024), 2)} MB")
        log_lines.append(f"Net File Size Reduction: {pct_reduction}%")
        log_text = "\n".join(log_lines)

        # Store run in database
        db = Database.get_instance()
        with db.session_scope() as session:
            opt_run = OptimizationRun(
                project_id=project_id,
                model_id=model_id,
                target_backend=target_backend,
                strategy=strategy,
                precision=precision,
                optimizations_applied_json=json.dumps([
                    "Conv+BatchNorm Fusion" if enable_fusion else None,
                    "Graph Optimization (Constant Folding & Dead Code)" if enable_graph_opt else None,
                    f"{precision} Quantization"
                ]),
                optimized_model_path=str(final_optimized_path.resolve()),
                original_size_bytes=orig_size_bytes,
                optimized_size_bytes=final_size_bytes,
                size_reduction_pct=pct_reduction,
                status="Completed",
                log_text=log_text,
                created_at=datetime.now(timezone.utc)
            )
            session.add(opt_run)
            session.flush()
            run_id = opt_run.id

            # Update project status
            proj = session.query(Project).filter_by(id=project_id).first()
            if proj:
                proj.status = "Optimized"

        return {
            "run_id": run_id,
            "optimized_model_path": str(final_optimized_path),
            "original_size_mb": round(orig_size_bytes / (1024 * 1024), 2),
            "optimized_size_mb": round(final_size_bytes / (1024 * 1024), 2),
            "size_reduction_pct": pct_reduction,
            "log_text": log_text
        }
