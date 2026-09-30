"""
Model Library Registry for SnapForge.
Maintains catalog of supported vision, speech, and multimodal architectures
with known Snapdragon compatibility profiles and optimization presets.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ModelCatalogItem:
    id: str
    name: str
    category: str  # "Object Detection", "Image Classification", "Segmentation", "Speech", "Language"
    format: str
    description: str
    default_input_shape: str
    typical_precision: str
    snapdragon_target: str
    expected_npu_compatibility: str
    tags: List[str] = field(default_factory=list)
    optimization_preset: str = "Balanced"
    notes: str = ""


SAMPLE_CATALOG: List[ModelCatalogItem] = [
    ModelCatalogItem(
        id="yolo_v8n_detector",
        name="YOLOv8-Nano Object Detector",
        category="Object Detection",
        format="ONNX",
        description="Lightweight real-time object detector (3.2M params) tailored for edge devices.",
        default_input_shape="[1, 3, 640, 640]",
        typical_precision="INT8 / FP16",
        snapdragon_target="Snapdragon NPU + CPU Hybrid",
        expected_npu_compatibility="94.2% (Backbone/Neck on NPU, NMS on CPU)",
        tags=["Vision", "Detection", "Real-time", "Robotics", "Security"],
        optimization_preset="Battery Efficiency",
        notes="NonMaxSuppression is executed on CPU fallback. Conv and C2f layers accelerate natively on Hexagon HTP."
    ),
    ModelCatalogItem(
        id="mobilenet_v2_classifier",
        name="MobileNetV2 Vision Classifier",
        category="Image Classification",
        format="ONNX",
        description="Depthwise-separable convolutional vision model for mobile and edge platforms.",
        default_input_shape="[1, 3, 224, 224]",
        typical_precision="INT8",
        snapdragon_target="Snapdragon NPU",
        expected_npu_compatibility="100.0%",
        tags=["Vision", "Classification", "ImageNet", "Mobile"],
        optimization_preset="Maximum Performance",
        notes="100% compatible with Hexagon Tensor Processor. High memory savings with static INT8 quantization."
    ),
    ModelCatalogItem(
        id="resnet18_classifier",
        name="ResNet-18 Residual Network",
        category="Image Classification",
        format="ONNX",
        description="Standard 18-layer residual network with skip connections and batch normalization.",
        default_input_shape="[1, 3, 224, 224]",
        typical_precision="FP16 / INT8",
        snapdragon_target="Snapdragon NPU",
        expected_npu_compatibility="100.0%",
        tags=["Vision", "Classification", "Baseline"],
        optimization_preset="Balanced",
        notes="Fusing Conv + BatchNormalization eliminates 17 redundant nodes and reduces memory bandwidth."
    ),
    ModelCatalogItem(
        id="tiny_bert_embedding",
        name="TinyBERT Text Embedding",
        category="Language",
        format="ONNX",
        description="Compact 4-layer transformer for sentence embeddings and classification.",
        default_input_shape="[1, 128]",
        typical_precision="FP16",
        snapdragon_target="Snapdragon NPU / Adreno GPU",
        expected_npu_compatibility="88.5%",
        tags=["NLP", "Transformer", "Embeddings"],
        optimization_preset="Balanced",
        notes="Attention QKV matrix multiplications accelerate via Hexagon Tensor Processor."
    ),
    ModelCatalogItem(
        id="whisper_tiny_encoder",
        name="Whisper-Tiny Audio Encoder",
        category="Speech",
        format="ONNX",
        description="Speech-to-text audio feature extractor encoder.",
        default_input_shape="[1, 80, 3000]",
        typical_precision="FP16",
        snapdragon_target="Snapdragon NPU",
        expected_npu_compatibility="92.0%",
        tags=["Audio", "Speech", "ASR"],
        optimization_preset="Maximum Performance",
        notes="1D convolutions and multi-head attention map efficiently to Hexagon DSP."
    ),
]


class ModelRegistry:
    """Provides access to curated catalog and local models."""

    @staticmethod
    def get_catalog() -> List[ModelCatalogItem]:
        return SAMPLE_CATALOG

    @staticmethod
    def get_by_id(model_id: str) -> Optional[ModelCatalogItem]:
        for item in SAMPLE_CATALOG:
            if item.id == model_id:
                return item
        return None
