"""
Nibble CLI Entrypoint.
Enables running:
  python -m nibble hardware
  python -m nibble providers
  python -m nibble analyze <model.onnx>
  python -m nibble optimize <model.onnx>
  python -m nibble benchmark <model.onnx> [--intra-op-threads 4]
  python -m nibble compare <orig.onnx> <opt.onnx>
  python -m nibble test-mvp
  python -m nibble report <project_id>
  python -m nibble gui
"""

import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import argparse
from app.cli import (
    cmd_hardware,
    cmd_providers,
    cmd_analyze,
    cmd_optimize,
    cmd_benchmark,
    cmd_compare,
    cmd_report,
    cmd_gui,
)


def cmd_test_mvp(args):
    """Run automated MVP validation suite."""
    from scripts.test_mvp import main as run_test_mvp
    run_test_mvp()


def main():
    parser = argparse.ArgumentParser(
        prog="python -m nibble",
        description="Nibble — AI Model Optimization & Deployment Studio"
    )
    subparsers = parser.add_subparsers(dest="command", help="Nibble command to execute")

    # hardware
    p_hw = subparsers.add_parser("hardware", help="Display genuine hardware detection & Qualcomm NPU status")
    p_hw.set_defaults(func=cmd_hardware)

    # providers
    p_prov = subparsers.add_parser("providers", help="Display ONNX Runtime execution provider readiness (CPU, DirectML, QNN)")
    p_prov.set_defaults(func=cmd_providers)

    # analyze
    p_an = subparsers.add_parser("analyze", help="Inspect model and analyze Snapdragon compatibility")
    p_an.add_argument("model_path", help="Path to ONNX or PyTorch model file")
    p_an.set_defaults(func=cmd_analyze)

    # optimize
    p_opt = subparsers.add_parser("optimize", help="Optimize model with fusion and quantization")
    p_opt.add_argument("model_path", help="Path to ONNX model file")
    p_opt.add_argument("--target", default="CPU", help="Target backend (npu, cpu, gpu, hybrid)")
    p_opt.add_argument("--precision", default="FP16", choices=["INT8", "FP16", "FP32", "int8", "fp16", "fp32"], help="Target precision")
    p_opt.set_defaults(func=cmd_optimize)

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Benchmark model with latency percentiles and throughput")
    p_bm.add_argument("model_path", help="Path to model file")
    p_bm.add_argument("--backend", default="CPU", choices=["CPU", "GPU", "Snapdragon NPU", "Hybrid", "cpu", "gpu"], help="Execution backend")
    p_bm.add_argument("--provider", default=None, help="Explicit ONNX Runtime Execution Provider (e.g. CPUExecutionProvider, DmlExecutionProvider)")
    p_bm.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p_bm.add_argument("--runs", type=int, default=50, help="Measured iterations")
    p_bm.add_argument("--intra-op-threads", type=int, default=0, help="Intra-op thread count (0 = auto)")
    p_bm.add_argument("--inter-op-threads", type=int, default=0, help="Inter-op thread count (0 = auto)")
    p_bm.set_defaults(func=cmd_benchmark)

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare original vs optimized models and export reproducible benchmark bundle")
    p_cmp.add_argument("original_model", help="Path to original ONNX model file")
    p_cmp.add_argument("optimized_model", help="Path to optimized ONNX model file")
    p_cmp.add_argument("--backend", default="CPU", choices=["CPU", "GPU", "Snapdragon NPU", "cpu", "gpu"], help="Execution backend")
    p_cmp.add_argument("--provider", default=None, help="Explicit ONNX Runtime Execution Provider (e.g. CPUExecutionProvider, DmlExecutionProvider)")
    p_cmp.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p_cmp.add_argument("--runs", type=int, default=50, help="Measured iterations")
    p_cmp.add_argument("--intra-op-threads", type=int, default=0, help="Intra-op thread count (0 = auto)")
    p_cmp.add_argument("--inter-op-threads", type=int, default=0, help="Inter-op thread count (0 = auto)")
    p_cmp.set_defaults(func=cmd_compare)

    # test-mvp
    p_tm = subparsers.add_parser("test-mvp", help="Execute complete automated end-to-end MVP validation pipeline")
    p_tm.set_defaults(func=cmd_test_mvp)

    # report
    p_rep = subparsers.add_parser("report", help="Generate PDF/Markdown/JSON/CSV optimization report")
    p_rep.add_argument("project_id", type=int, help="ID of the project")
    p_rep.add_argument("--format", default="ALL", choices=["PDF", "MD", "JSON", "CSV", "ALL"], help="Output format")
    p_rep.set_defaults(func=cmd_report)

    # gui
    p_gui = subparsers.add_parser("gui", help="Launch Nibble PySide6 desktop interface")
    p_gui.set_defaults(func=cmd_gui)

    if len(sys.argv) == 1:
        cmd_gui(None)
        return

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
