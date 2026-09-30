"""
Nibble MVP Automated End-to-End Validation Script.
Validates the complete AI model optimization pipeline on an AMD Windows development laptop:
1. Environment Check
2. Hardware Detection Check (confirms AMD detected, confirms Snapdragon false)
3. Model Generation (test_cnn.onnx)
4. Model Validation (onnx.checker.check_model)
5. Model Inspection (params, layers, inputs/outputs)
6. Graph Analysis (nodes, edges, partitions)
7. Snapdragon Compatibility Analysis [Static Analysis]
8. Model Optimization (Graph optimization & FP16)
9. Output Model Validation (onnx.checker.check_model)
10. Original Model CPU Inference
11. Optimized Model CPU Inference
12. Benchmark Execution (10 warmup, 50 measured runs on host CPU)
13. Accuracy Validation (MAE, RMSE, Cosine Similarity)
14. Qualcomm Backend Isolation Test (confirms clean refusal on AMD)
15. Report Generation (Markdown, JSON, CSV with AMD notice & disclaimer)
16. Final Verification Summary
"""

import sys
import os
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_stage(step_num: int, title: str, func):
    print(f"\n[{step_num}/15] {title}...")
    try:
        res = func()
        print(f"      [PASS] {title}")
        return res
    except Exception as e:
        print(f"\n[FAILED] Stage {step_num} failed: {title}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def stage_1_environment():
    import onnx
    import onnxruntime as ort
    import psutil
    py_ver = sys.version.split()[0]
    print(f"      Python: {py_ver} | ONNX: {onnx.__version__} | ONNX Runtime: {ort.__version__}")
    print(f"      ORT Available Providers: {ort.get_available_providers()}")
    return True


def stage_2_hardware():
    from app.core.hardware_manager import HardwareManager
    hw = HardwareManager.get_hardware_profile(force_refresh=True)
    print(f"      Vendor: {hw.cpu_vendor}")
    print(f"      Model: {hw.cpu_model}")
    print(f"      Architecture: {hw.cpu_arch}")
    print(f"      Snapdragon: {hw.is_snapdragon}")
    print(f"      Qualcomm NPU Status: {hw.npu_status}")
    print(f"      Qualcomm QNN Status: {hw.qnn_status}")
    print(f"      Hardware Badge: {hw.hardware_badge_text}")

    assert hw.cpu_vendor == "AMD", f"Expected AMD vendor, got {hw.cpu_vendor}"
    assert hw.is_snapdragon is False, f"Expected is_snapdragon=False, got {hw.is_snapdragon}"
    assert hw.npu_status in ("Not detected", "Unavailable", "Unavailable on this hardware"), f"Unexpected NPU status: {hw.npu_status}"
    assert hw.qnn_status == "Not active", f"Expected QNN Not active, got {hw.qnn_status}"
    return hw


def stage_3_generate_model():
    from scripts.create_test_model import generate_test_cnn_model
    model_path = Path("models/test_cnn.onnx")
    if not model_path.exists():
        model_path = generate_test_cnn_model(model_path)
    print(f"      Model path: {model_path} ({model_path.stat().st_size:,} bytes)")
    return model_path


def stage_4_validate_model(model_path):
    import onnx
    m = onnx.load(str(model_path))
    onnx.checker.check_model(m)
    print(f"      onnx.checker passed! IR Version: {m.ir_version}, Opset: {m.opset_import[0].version}")
    return m


def stage_5_inspect_model(model_path):
    from app.models.model_inspector import ModelInspector
    meta = ModelInspector.inspect(str(model_path))
    print(f"      Name: {meta.name} | Parameters: {meta.total_params:,} | Size: {meta.file_size_mb} MB")
    print(f"      Operators: {meta.op_counts}")
    print(f"      Inputs: {[(i.name, i.shape, i.dtype) for i in meta.inputs]}")
    print(f"      Outputs: {[(o.name, o.shape, o.dtype) for o in meta.outputs]}")
    assert meta.total_params > 0, "Model params must be > 0"
    assert len(meta.inputs) > 0, "Model must have inputs"
    assert len(meta.outputs) > 0, "Model must have outputs"
    return meta


def stage_6_graph_analysis(meta):
    from app.models.graph_analyzer import GraphAnalyzer
    graph = GraphAnalyzer.analyze_graph(meta)
    print(f"      Graph Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)} | Partitions: {len(graph.partitions)}")
    print(f"      Hexagon NPU Candidate Compute: {graph.npu_compute_pct}%")
    assert len(graph.nodes) > 0, "Graph must contain nodes"
    assert len(graph.edges) > 0, "Graph must contain edges"
    return graph


def stage_7_compatibility(meta):
    from app.models.compatibility import SnapdragonCompatibilityEngine
    compat = SnapdragonCompatibilityEngine.analyze(meta)
    print(f"      [Static Analysis] Weighted Hexagon NPU Score: {compat.weighted_npu_score}%")
    print(f"      [Static Analysis] Operator Count NPU Score: {compat.npu_score}%")
    print(f"      [Static Analysis] Supported Ops: {compat.supported_nodes_count}, Partial: {compat.partial_nodes_count}, Unsupported: {compat.unsupported_nodes_count}")
    print(f"      [Static Analysis] Recommendations: {compat.recommendations}")
    assert compat.npu_score >= 80.0, "Test CNN should have high NPU compatibility score"
    return compat


def stage_8_optimization(model_path):
    from app.optimization.graph_optimizer import GraphOptimizer
    from app.optimization.quantization import QuantizationEngine
    out_dir = Path("optimized_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Graph optimization
    graph_opt_path = out_dir / "test_cnn_graph_opt.onnx"
    g_res = GraphOptimizer.optimize(model_path, graph_opt_path, opt_level="BASIC")
    print(f"      Graph Optimization: {g_res.get('reduction_pct', 0)}% node reduction")

    # 2. FP16 Conversion
    fp16_path = out_dir / "test_cnn_fp16.onnx"
    q_res = QuantizationEngine.quantize_fp16(graph_opt_path if graph_opt_path.exists() else model_path, fp16_path)
    print(f"      FP16 Conversion: {q_res.get('size_reduction_pct', 0)}% size reduction ({q_res.get('optimized_size_bytes', 0):,} bytes)")

    assert fp16_path.exists(), "Optimized FP16 model must exist"
    return fp16_path


def stage_9_validate_optimized(opt_path):
    import onnx
    m = onnx.load(str(opt_path))
    onnx.checker.check_model(m)
    print(f"      Optimized model valid: IR Version: {m.ir_version}, Nodes: {len(m.graph.node)}")
    return m


def stage_10_original_inference(model_path):
    from app.backends import CPUBackend
    cpu = CPUBackend()
    cpu.load_model(model_path)
    np.random.seed(123)
    sample_input = {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)}
    out = cpu.run_inference(sample_input)
    print(f"      Original output shape: {out['output'].shape}, sample values: {out['output'][0, :3]}")
    return cpu, sample_input, out


def stage_11_optimized_inference(opt_path, sample_input):
    from app.backends import CPUBackend
    cpu_opt = CPUBackend()
    cpu_opt.load_model(opt_path)
    sample_fp16 = {"input": sample_input["input"].astype(np.float16)}
    out_opt = cpu_opt.run_inference(sample_fp16)
    print(f"      Optimized output shape: {out_opt['output'].shape}, sample values: {out_opt['output'][0, :3]}")
    return cpu_opt, out_opt


def stage_12_benchmarking(cpu_orig, cpu_opt, orig_path, opt_path):
    warmup = 10
    measured = 50
    print(f"      Running real measurements on host CPU ({warmup} warmup, {measured} measured)...")
    bench_orig = cpu_orig.benchmark(orig_path, warmup_runs=warmup, measured_runs=measured)
    bench_opt = cpu_opt.benchmark(opt_path, warmup_runs=warmup, measured_runs=measured)

    s1 = bench_orig["stats"]
    s2 = bench_opt["stats"]
    speedup = round(s1["median_ms"] / max(0.001, s2["median_ms"]), 2)
    pct_diff = round(((s1["median_ms"] - s2["median_ms"]) / max(0.001, s1["median_ms"])) * 100.0, 1)

    print(f"      [Measured] Original Median: {s1['median_ms']:.3f} ms | Throughput: {s1['throughput_fps']:.1f} FPS")
    print(f"      [Measured] Optimized Median: {s2['median_ms']:.3f} ms | Throughput: {s2['throughput_fps']:.1f} FPS")
    print(f"      [Measured] Delta: {pct_diff}% (Speedup Factor: {speedup}x)")

    return {
        "original_median_ms": s1["median_ms"],
        "optimized_median_ms": s2["median_ms"],
        "original_mean_ms": s1["mean_ms"],
        "optimized_mean_ms": s2["mean_ms"],
        "original_p95_ms": s1["p95_ms"],
        "optimized_p95_ms": s2["p95_ms"],
        "original_throughput_fps": s1["throughput_fps"],
        "optimized_throughput_fps": s2["throughput_fps"],
        "latency_reduction_pct": pct_diff,
        "speedup_factor": speedup,
        "backend_name": "CPU",
        "original_backend": "CPU",
        "optimized_backend": "CPU",
        "label_type": "Measured"
    }


def stage_13_accuracy(orig_out, opt_out):
    from app.profiling.accuracy import AccuracyValidator
    acc = AccuracyValidator.compare_outputs(orig_out, opt_out)
    print(f"      Cosine Similarity: {acc['overall_cosine_similarity']:.5f}")
    print(f"      Mean Absolute Error (MAE): {acc['overall_mae']:.6f}")
    print(f"      Max Absolute Difference: {acc['max_absolute_diff']:.6f}")
    print(f"      Fidelity Grade: {acc['fidelity_grade']}")
    assert acc["overall_cosine_similarity"] >= 0.999, f"Cosine similarity below threshold: {acc['overall_cosine_similarity']}"
    return acc


def stage_14_qualcomm_refusal():
    from app.backends import QualcommBackend, SnapdragonExecutionUnavailableError
    qcom = QualcommBackend()
    avail = qcom.is_available()
    stat = qcom.status()
    reason = qcom.reason()

    print(f"      QualcommBackend.is_available(): {avail}")
    print(f"      QualcommBackend.status(): {stat}")
    print(f"      QualcommBackend.reason(): {reason}")

    assert avail is False, "Qualcomm backend MUST report is_available() == False on AMD"
    assert stat == "NOT AVAILABLE", f"Expected 'NOT AVAILABLE', got '{stat}'"
    assert "Snapdragon hardware was not detected" in reason, f"Unexpected reason: {reason}"

    # Verify that trying to execute raises SnapdragonExecutionUnavailableError
    raised = False
    try:
        qcom.run_inference({"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})
    except SnapdragonExecutionUnavailableError as e:
        raised = True
        print(f"      Correctly raised SnapdragonExecutionUnavailableError: {e}")
    except Exception as e:
        print(f"      Wrong exception type raised: {type(e)}: {e}")

    assert raised, "QualcommBackend MUST raise SnapdragonExecutionUnavailableError when run_inference is invoked on AMD"
    return True


def stage_15_report(hw, meta, compat, opt_path, bench_summary, acc):
    from app.reports.report_generator import ReportGenerator
    orig_path = Path("models/test_cnn.onnx")

    bench_summary["accuracy"] = acc
    bench_summary["original_size_mb"] = meta.file_size_mb
    bench_summary["optimized_size_mb"] = round(opt_path.stat().st_size / (1024 * 1024), 3)
    bench_summary["size_reduction_pct"] = round(
        ((meta.file_size_bytes - opt_path.stat().st_size) / max(1, meta.file_size_bytes)) * 100.0, 1
    )

    proj_data = {"name": "Nibble-MVP-Validation", "status": "Optimized & Benchmarked"}
    opt_data = {
        "strategy": "Balanced",
        "precision": "FP16",
        "original_size_mb": bench_summary["original_size_mb"],
        "optimized_size_mb": bench_summary["optimized_size_mb"],
        "size_reduction_pct": bench_summary["size_reduction_pct"]
    }

    files = ReportGenerator.generate_full_report(
        project_data=proj_data,
        hardware_data=hw.to_dict(),
        model_data=meta.to_dict(),
        compat_data=compat.to_dict(),
        opt_data=opt_data,
        bench_data=bench_summary,
        output_format="ALL"
    )

    print(f"      Generated report files:")
    for fmt, p in files.items():
        print(f"        - {fmt}: {p}")
        assert Path(p).exists(), f"Report file missing: {p}"

    # Verify AMD disclaimer in JSON
    import json
    with open(files["JSON"], "r", encoding="utf-8") as f:
        j = json.load(f)
        assert "AMD" in j["platform"], f"Expected AMD platform in JSON, got: {j.get('platform')}"
        assert "Snapdragon NPU was not benchmarked because this development machine is AMD." in j["disclaimer"], "Disclaimer missing in JSON report"

    # Verify AMD disclaimer in Markdown
    with open(files["MD"], "r", encoding="utf-8") as f:
        md = f.read()
        assert "Snapdragon NPU was not benchmarked because this development machine is AMD." in md, "Disclaimer missing in Markdown report"

    print("      Verified AMD notice and disclaimer in generated reports.")
    return files


def main():
    print("=" * 60)
    print("    NIBBLE — AMD MVP END-TO-END VALIDATION SUITE")
    print("=" * 60)

    run_stage(1, "Python Environment Check", stage_1_environment)
    hw = run_stage(2, "Hardware Detection Check (AMD Isolation)", stage_2_hardware)
    model_path = run_stage(3, "Test Model Generation (CNN)", stage_3_generate_model)
    run_stage(4, "Model Structural Validation", lambda: stage_4_validate_model(model_path))
    meta = run_stage(5, "Model Architecture Inspection", lambda: stage_5_inspect_model(model_path))
    run_stage(6, "Computational Graph Analysis", lambda: stage_6_graph_analysis(meta))
    compat = run_stage(7, "Snapdragon Compatibility Analysis [Static Analysis]", lambda: stage_7_compatibility(meta))
    opt_path = run_stage(8, "Model Optimization (Fusion & FP16 Conversion)", lambda: stage_8_optimization(model_path))
    run_stage(9, "Optimized Model Validation", lambda: stage_9_validate_optimized(opt_path))
    cpu_orig, sample_input, orig_out = run_stage(10, "Original Model CPU Inference", lambda: stage_10_original_inference(model_path))
    cpu_opt, opt_out = run_stage(11, "Optimized Model CPU Inference", lambda: stage_11_optimized_inference(opt_path, sample_input))
    bench_summary = run_stage(12, "Empirical Benchmarking (Warmup & Measured Runs)", lambda: stage_12_benchmarking(cpu_orig, cpu_opt, model_path, opt_path))
    acc = run_stage(13, "Accuracy Validation (MAE & Cosine Similarity)", lambda: stage_13_accuracy(orig_out, opt_out))
    run_stage(14, "Qualcomm Backend Refusal & Isolation Test", stage_14_qualcomm_refusal)
    run_stage(15, "Report Generation (MD, JSON, CSV, PDF)", lambda: stage_15_report(hw, meta, compat, opt_path, bench_summary, acc))

    print("\n" + "=" * 60)
    print("    NIBBLE MVP VALIDATION: PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
