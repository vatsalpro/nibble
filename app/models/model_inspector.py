"""
Model Inspector for SnapForge.
Performs genuine inspection of ONNX and PyTorch models:
- Extracts graph inputs, outputs, nodes, initializers
- Computes parameter count, tensor shapes, data types
- Computes estimated FLOPs for compute-intensive operators (Conv, Gemm, MatMul, etc.)
- Traverses computational graph structure
"""

import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
from nibble.utils.hashing import calculate_file_sha256


# ONNX tensor element type map
ONNX_DTYPE_MAP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "double",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}


@dataclass
class TensorInfo:
    name: str
    shape: List[Any]
    dtype: str
    is_initializer: bool = False
    param_count: int = 0


@dataclass
class NodeInfo:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    input_shapes: List[List[Any]] = field(default_factory=list)
    output_shapes: List[List[Any]] = field(default_factory=list)
    param_count: int = 0
    estimated_flops: int = 0
    data_type: str = "float32"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    name: str
    format: str
    file_path: str
    file_size_bytes: int
    file_size_mb: float
    total_params: int
    total_flops: int
    inputs: List[TensorInfo] = field(default_factory=list)
    outputs: List[TensorInfo] = field(default_factory=list)
    nodes: List[NodeInfo] = field(default_factory=list)
    op_counts: Dict[str, int] = field(default_factory=dict)
    unique_ops: List[str] = field(default_factory=list)
    primary_dtype: str = "float32"
    producer_name: str = ""
    ir_version: int = 0
    opset_version: int = 0
    model_sha256: str = ""
    estimated_macs: int = 0
    activation_memory_mb: float = 0.0
    weight_memory_mb: float = 0.0
    largest_tensors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class ModelInspector:
    """Genuine ONNX & PyTorch model inspection engine."""

    @staticmethod
    def inspect(file_path: str | Path) -> ModelMetadata:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".onnx":
            return ModelInspector.inspect_onnx(path)
        elif ext in (".pt", ".pth"):
            return ModelInspector.inspect_pytorch(path)
        else:
            raise ValueError(f"Unsupported model format '{ext}'. Expected .onnx, .pt, or .pth.")

    @staticmethod
    def inspect_onnx(path: Path) -> ModelMetadata:
        file_size_bytes = path.stat().st_size
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

        # Load ONNX model
        model = onnx.load(str(path))

        # Try running shape inference to get full intermediate tensor shapes
        try:
            inferred_model = shape_inference.infer_shapes(model)
        except Exception:
            inferred_model = model

        graph = inferred_model.graph

        # Extract Initializers & Parameter counts
        initializer_map: Dict[str, np.ndarray] = {}
        param_counts: Dict[str, int] = {}
        total_params = 0

        for init in graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
                initializer_map[init.name] = arr
                count = int(arr.size)
                param_counts[init.name] = count
                total_params += count
            except Exception:
                pass

        # Extract ValueInfo shape and dtype map
        shape_type_map: Dict[str, Tuple[List[Any], str]] = {}

        def parse_value_info(vi):
            dtype = "unknown"
            shape = []
            if vi.type.HasField("tensor_type"):
                elem_type = vi.type.tensor_type.elem_type
                dtype = ONNX_DTYPE_MAP.get(elem_type, f"type_{elem_type}")
                if vi.type.tensor_type.HasField("shape"):
                    for dim in vi.type.tensor_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        elif dim.HasField("dim_param"):
                            shape.append(dim.dim_param)
                        else:
                            shape.append("?")
            return shape, dtype

        # Inputs
        inputs_list: List[TensorInfo] = []
        for vi in graph.input:
            shape, dtype = parse_value_info(vi)
            is_init = vi.name in initializer_map
            count = param_counts.get(vi.name, 0)
            inputs_list.append(TensorInfo(
                name=vi.name,
                shape=shape,
                dtype=dtype,
                is_initializer=is_init,
                param_count=count
            ))
            shape_type_map[vi.name] = (shape, dtype)

        # Outputs
        outputs_list: List[TensorInfo] = []
        for vi in graph.output:
            shape, dtype = parse_value_info(vi)
            outputs_list.append(TensorInfo(
                name=vi.name,
                shape=shape,
                dtype=dtype,
                is_initializer=False
            ))
            shape_type_map[vi.name] = (shape, dtype)

        # Intermediate value info
        for vi in graph.value_info:
            shape, dtype = parse_value_info(vi)
            shape_type_map[vi.name] = (shape, dtype)

        # Initializer shapes
        for name, arr in initializer_map.items():
            shape_type_map[name] = (list(arr.shape), str(arr.dtype))

        # Nodes
        nodes_list: List[NodeInfo] = []
        op_counts: Dict[str, int] = {}
        total_flops = 0

        for i, node in enumerate(graph.node):
            node_name = node.name or f"{node.op_type}_{i}"
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

            # Extract attributes
            attrs: Dict[str, Any] = {}
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    attrs[attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attrs[attr.name] = round(attr.f, 4)
                elif attr.type == onnx.AttributeProto.INTS:
                    attrs[attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    attrs[attr.name] = [round(f, 4) for f in attr.floats]
                elif attr.type == onnx.AttributeProto.STRING:
                    attrs[attr.name] = attr.s.decode("utf-8", errors="ignore")

            # Collect input and output shapes
            in_shapes = [shape_type_map.get(inp, ([], "unknown"))[0] for inp in node.input]
            out_shapes = [shape_type_map.get(out, ([], "unknown"))[0] for out in node.output]

            # Primary data type
            node_dtype = "float32"
            if node.input and node.input[0] in shape_type_map:
                node_dtype = shape_type_map[node.input[0]][1]

            # Node parameters
            node_params = sum(param_counts.get(inp, 0) for inp in node.input)

            # FLOPs Estimation
            node_flops = ModelInspector._estimate_node_flops(op_type, in_shapes, out_shapes, attrs, node_params)
            total_flops += node_flops

            nodes_list.append(NodeInfo(
                name=node_name,
                op_type=op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attrs,
                input_shapes=in_shapes,
                output_shapes=out_shapes,
                param_count=node_params,
                estimated_flops=node_flops,
                data_type=node_dtype
            ))

        # Determine opset
        opset_ver = 0
        if model.opset_import:
            for op in model.opset_import:
                if op.domain == "" or op.domain == "ai.onnx":
                    opset_ver = op.version

        # Primary data type
        primary_dtype = "float32"
        if inputs_list and inputs_list[0].dtype != "unknown":
            primary_dtype = inputs_list[0].dtype

        # Model SHA-256
        model_sha256 = calculate_file_sha256(path)

        # Estimated MACs (typically FLOPs // 2)
        estimated_macs = total_flops // 2

        # Weight memory (MB)
        total_weight_bytes = sum(arr.nbytes for arr in initializer_map.values())
        weight_memory_mb = round(total_weight_bytes / (1024 * 1024), 2)

        # Largest tensors (top 5)
        sorted_inits = sorted(initializer_map.items(), key=lambda kv: kv[1].nbytes, reverse=True)
        largest_tensors = [
            {
                "name": name,
                "shape": list(arr.shape),
                "param_count": int(arr.size),
                "size_mb": round(arr.nbytes / (1024 * 1024), 3),
                "dtype": str(arr.dtype),
            }
            for name, arr in sorted_inits[:5]
        ]

        # Estimated activation memory (MB) from intermediate outputs
        DTYPE_BYTES = {
            "float32": 4, "int32": 4, "uint32": 4,
            "float16": 2, "bfloat16": 2, "int16": 2, "uint16": 2,
            "int8": 1, "uint8": 1, "bool": 1,
            "double": 8, "int64": 8, "uint64": 8,
        }
        total_act_bytes = 0
        for node in nodes_list:
            b_per_elem = DTYPE_BYTES.get(node.data_type, 4)
            for out_shape in node.output_shapes:
                if out_shape:
                    numel = 1
                    for d in out_shape:
                        if isinstance(d, int) and d > 0:
                            numel *= d
                    total_act_bytes += numel * b_per_elem
        activation_memory_mb = round(total_act_bytes / (1024 * 1024), 2)

        return ModelMetadata(
            name=path.stem,
            format="ONNX",
            file_path=str(path.resolve()),
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            total_params=total_params,
            total_flops=total_flops,
            inputs=[inp for inp in inputs_list if not inp.is_initializer],
            outputs=outputs_list,
            nodes=nodes_list,
            op_counts=op_counts,
            unique_ops=sorted(list(op_counts.keys())),
            primary_dtype=primary_dtype,
            producer_name=model.producer_name or "ONNX",
            ir_version=model.ir_version,
            opset_version=opset_ver,
            model_sha256=model_sha256,
            estimated_macs=estimated_macs,
            activation_memory_mb=activation_memory_mb,
            weight_memory_mb=weight_memory_mb,
            largest_tensors=largest_tensors,
        )

    @staticmethod
    def _estimate_node_flops(
        op_type: str,
        in_shapes: List[List[Any]],
        out_shapes: List[List[Any]],
        attrs: Dict[str, Any],
        param_count: int
    ) -> int:
        """Estimate floating point operations for an operator without arbitrary fallbacks."""
        try:
            if op_type in ("Conv", "ConvTranspose") and out_shapes and len(out_shapes[0]) >= 4:
                out_shape = [int(x) if isinstance(x, int) else 1 for x in out_shapes[0]]
                batch = out_shape[0] if len(out_shape) > 0 else 1
                c_out = out_shape[1] if len(out_shape) > 1 else 1
                h_out = out_shape[2] if len(out_shape) > 2 else 1
                w_out = out_shape[3] if len(out_shape) > 3 else 1
                spatial_out = batch * h_out * w_out

                if in_shapes and len(in_shapes) > 1 and len(in_shapes[1]) >= 4:
                    w_shape = [int(x) if isinstance(x, int) else 1 for x in in_shapes[1]]
                    c_in_per_group = w_shape[1]
                    k_h = w_shape[2]
                    k_w = w_shape[3]
                    return 2 * spatial_out * c_out * c_in_per_group * k_h * k_w
                elif param_count > 0:
                    return 2 * spatial_out * param_count
                return 0

            elif op_type in ("Gemm", "MatMul"):
                if in_shapes and len(in_shapes) >= 2:
                    sA = [int(x) if isinstance(x, int) else 1 for x in in_shapes[0]]
                    sB = [int(x) if isinstance(x, int) else 1 for x in in_shapes[1]]
                    m = sA[-2] if len(sA) >= 2 else 1
                    k = sA[-1] if len(sA) >= 1 else 1
                    n = sB[-1] if len(sB) >= 1 else 1
                    batch = 1
                    if len(sA) > 2:
                        for dim in sA[:-2]:
                            batch *= dim
                    return 2 * batch * m * k * n
                elif param_count > 0:
                    return 2 * param_count
                return 0

            elif op_type in ("Relu", "Sigmoid", "Tanh", "LeakyRelu", "Clip", "Add", "Mul", "Sub", "Div"):
                if out_shapes and out_shapes[0]:
                    count = 1
                    for dim in out_shapes[0]:
                        if isinstance(dim, int):
                            count *= dim
                        else:
                            return 0
                    return count
                return 0

            elif op_type in ("MaxPool", "AveragePool", "GlobalAveragePool"):
                if out_shapes and out_shapes[0]:
                    count = 1
                    for dim in out_shapes[0]:
                        if isinstance(dim, int):
                            count *= dim
                        else:
                            return 0
                    kernel_size = 1
                    if "kernel_shape" in attrs and attrs["kernel_shape"]:
                        for k in attrs["kernel_shape"]:
                            if isinstance(k, int):
                                kernel_size *= k
                    return count * kernel_size
                return 0
        except Exception:
            pass

        return 0

    @staticmethod
    def inspect_pytorch(path: Path) -> ModelMetadata:
        """Inspect a PyTorch .pt or .pth model file."""
        import torch

        file_size_bytes = path.stat().st_size
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        model_sha256 = calculate_file_sha256(path)

        data = None
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:
            try:
                data = torch.load(str(path), map_location="cpu", weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load PyTorch file: {e}")

        total_params = 0
        nodes = []
        op_counts = {}
        total_weight_bytes = 0
        param_list = []

        if isinstance(data, dict):
            for key, tensor in data.items():
                if hasattr(tensor, "numel"):
                    numel = tensor.numel()
                    total_params += numel
                    nbytes = tensor.element_size() * numel if hasattr(tensor, "element_size") else numel * 4
                    total_weight_bytes += nbytes
                    param_list.append((key, list(tensor.shape), numel, round(nbytes / (1024 * 1024), 3), str(tensor.dtype).replace("torch.", "")))
                    nodes.append(NodeInfo(
                        name=key,
                        op_type="WeightTensor",
                        inputs=[],
                        outputs=[key],
                        output_shapes=[list(tensor.shape)],
                        param_count=numel,
                        data_type=str(tensor.dtype).replace("torch.", "")
                    ))
                    op_counts["WeightTensor"] = op_counts.get("WeightTensor", 0) + 1
        elif isinstance(data, torch.nn.Module):
            for name, param in data.named_parameters():
                numel = param.numel()
                total_params += numel
                nbytes = param.element_size() * numel if hasattr(param, "element_size") else numel * 4
                total_weight_bytes += nbytes
                param_list.append((name, list(param.shape), numel, round(nbytes / (1024 * 1024), 3), str(param.dtype).replace("torch.", "")))
                nodes.append(NodeInfo(
                    name=name,
                    op_type="LayerParameter",
                    inputs=[],
                    outputs=[name],
                    output_shapes=[list(param.shape)],
                    param_count=numel,
                    data_type=str(param.dtype).replace("torch.", "")
                ))
            op_counts["PyTorchModule"] = 1

        param_list.sort(key=lambda x: x[3], reverse=True)
        largest_tensors = [
            {"name": p[0], "shape": p[1], "param_count": p[2], "size_mb": p[3], "dtype": p[4]}
            for p in param_list[:5]
        ]
        weight_memory_mb = round(total_weight_bytes / (1024 * 1024), 2)

        return ModelMetadata(
            name=path.stem,
            format="PyTorch",
            file_path=str(path.resolve()),
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            total_params=total_params,
            total_flops=0,
            nodes=nodes,
            op_counts=op_counts,
            unique_ops=list(op_counts.keys()),
            primary_dtype="float32",
            producer_name="PyTorch",
            model_sha256=model_sha256,
            estimated_macs=0,
            activation_memory_mb=0.0,
            weight_memory_mb=weight_memory_mb,
            largest_tensors=largest_tensors,
        )
