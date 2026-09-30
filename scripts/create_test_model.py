"""
Test Model Generator for Nibble MVP.
Generates a valid, lightweight, reproducible ONNX test CNN model
containing standard layers:
Input -> Conv -> Relu -> Conv -> Relu -> GlobalAveragePool -> Flatten -> Gemm -> Output
Validated with onnx.checker.check_model.
Zero heavyweight framework dependencies (pure ONNX helper).
"""

import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto


def generate_test_cnn_model(output_path: str | Path = "models/test_cnn.onnx") -> Path:
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # 1. Weights and biases (reproducible numpy arrays)
    conv1_w = (np.random.randn(16, 3, 3, 3) * 0.05).astype(np.float32)
    conv1_b = np.zeros(16, dtype=np.float32)
    conv2_w = (np.random.randn(32, 16, 3, 3) * 0.05).astype(np.float32)
    conv2_b = np.zeros(32, dtype=np.float32)
    gemm_w = (np.random.randn(32, 10) * 0.05).astype(np.float32)
    gemm_b = np.zeros(10, dtype=np.float32)

    # Convert initializers to ONNX TensorProto
    initializers = [
        helper.make_tensor("conv1_w", TensorProto.FLOAT, conv1_w.shape, conv1_w.flatten().tolist()),
        helper.make_tensor("conv1_b", TensorProto.FLOAT, conv1_b.shape, conv1_b.flatten().tolist()),
        helper.make_tensor("conv2_w", TensorProto.FLOAT, conv2_w.shape, conv2_w.flatten().tolist()),
        helper.make_tensor("conv2_b", TensorProto.FLOAT, conv2_b.shape, conv2_b.flatten().tolist()),
        helper.make_tensor("gemm_w", TensorProto.FLOAT, gemm_w.shape, gemm_w.flatten().tolist()),
        helper.make_tensor("gemm_b", TensorProto.FLOAT, gemm_b.shape, gemm_b.flatten().tolist()),
    ]

    # 2. Computational Graph Nodes
    # Conv1: (1, 3, 32, 32) -> (1, 16, 32, 32)
    conv1_node = helper.make_node(
        "Conv",
        inputs=["input", "conv1_w", "conv1_b"],
        outputs=["conv1_out"],
        name="conv1",
        pads=[1, 1, 1, 1],
        strides=[1, 1]
    )

    # Relu1: (1, 16, 32, 32) -> (1, 16, 32, 32)
    relu1_node = helper.make_node(
        "Relu",
        inputs=["conv1_out"],
        outputs=["relu1_out"],
        name="relu1"
    )

    # Conv2: (1, 16, 32, 32) -> (1, 32, 32, 32)
    conv2_node = helper.make_node(
        "Conv",
        inputs=["relu1_out", "conv2_w", "conv2_b"],
        outputs=["conv2_out"],
        name="conv2",
        pads=[1, 1, 1, 1],
        strides=[1, 1]
    )

    # Relu2: (1, 32, 32, 32) -> (1, 32, 32, 32)
    relu2_node = helper.make_node(
        "Relu",
        inputs=["conv2_out"],
        outputs=["relu2_out"],
        name="relu2"
    )

    # GlobalAveragePool: (1, 32, 32, 32) -> (1, 32, 1, 1)
    gap_node = helper.make_node(
        "GlobalAveragePool",
        inputs=["relu2_out"],
        outputs=["gap_out"],
        name="global_average_pool"
    )

    # Flatten: (1, 32, 1, 1) -> (1, 32)
    flatten_node = helper.make_node(
        "Flatten",
        inputs=["gap_out"],
        outputs=["flatten_out"],
        name="flatten",
        axis=1
    )

    # Gemm: (1, 32) x (32, 10) + (10,) -> (1, 10)
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["flatten_out", "gemm_w", "gemm_b"],
        outputs=["output"],
        name="gemm",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0
    )

    nodes = [
        conv1_node,
        relu1_node,
        conv2_node,
        relu2_node,
        gap_node,
        flatten_node,
        gemm_node
    ]

    # 3. Graph Inputs and Outputs
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        nodes=nodes,
        name="NibbleTestCNN",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers
    )

    opset_import = helper.make_opsetid("", 17)
    model = helper.make_model(
        graph,
        opset_imports=[opset_import],
        ir_version=10,
        producer_name="nibble-test-generator",
        doc_string="Standard Test CNN model for Nibble AMD MVP validation"
    )

    # 4. Strict Validation
    onnx.checker.check_model(model)

    # 5. Save model
    onnx.save(model, str(out_file))

    # Re-verify saved model on disk
    loaded = onnx.load(str(out_file))
    onnx.checker.check_model(loaded)

    print(f"[OK] Generated and verified test model: {out_file} ({out_file.stat().st_size:,} bytes)")
    return out_file


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "models/test_cnn.onnx"
    generate_test_cnn_model(target)
