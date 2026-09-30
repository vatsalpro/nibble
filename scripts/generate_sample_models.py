import os
import sys

# Configure UTF-8 for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx


class ResNetTiny(nn.Module):
    """Tiny ResNet with Conv + BatchNorm layers for fusion demonstration."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        res = out
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class YOLODemoNet(nn.Module):
    """Vision backbone with detection features."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.bbox_pred = nn.Conv2d(64, 85, 1)  # 80 classes + 4 bbox + 1 obj

    def forward(self, x):
        feat = self.stem(x)
        feat = self.head(feat)
        out = self.bbox_pred(feat)
        return out


def main():
    models_dir = Path("E:/snapdragon/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Generating genuine sample ONNX models for SnapForge...")

    # 1. ResNet Tiny (ideal for Conv+BN fusion and INT8 quantization)
    resnet = ResNetTiny()
    dummy_input_resnet = torch.randn(1, 3, 224, 224)
    resnet_path = models_dir / "resnet_classifier.onnx"
    torch.onnx.export(
        resnet,
        dummy_input_resnet,
        str(resnet_path),
        input_names=["input"],
        output_names=["probabilities"],
        training=torch.onnx.TrainingMode.PRESERVE,
        opset_version=18,
        dynamo=False
    )
    print(f"✓ Generated {resnet_path} ({round(resnet_path.stat().st_size / 1024, 1)} KB)")

    # 2. YOLO Demo Net (ideal for detection pipeline analysis)
    yolo = YOLODemoNet().eval()
    dummy_input_yolo = torch.randn(1, 3, 224, 224)
    yolo_path = models_dir / "yolov8_demo.onnx"
    torch.onnx.export(
        yolo,
        dummy_input_yolo,
        str(yolo_path),
        input_names=["images"],
        output_names=["detections"],
        opset_version=18,
        dynamo=False
    )
    # 3. YOLO Detection with NonMaxSuppression (for Snapdragon NPU bottleneck & Hybrid execution demo)
    try:
        yolo_model = onnx.load(str(yolo_path))
        # Add NMS inputs and node to showcase Snapdragon compatibility analyzer detecting NPU blocker
        max_boxes_init = onnx.numpy_helper.from_array(np.array([100], dtype=np.int64), name="max_output_boxes")
        iou_thresh_init = onnx.numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="iou_threshold")
        score_thresh_init = onnx.numpy_helper.from_array(np.array([0.25], dtype=np.float32), name="score_threshold")
        
        boxes_shape = onnx.helper.make_tensor_value_info("nms_boxes", onnx.TensorProto.FLOAT, [1, 100, 4])
        scores_shape = onnx.helper.make_tensor_value_info("nms_scores", onnx.TensorProto.FLOAT, [1, 80, 100])
        
        nms_node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=["nms_boxes", "nms_scores", "max_output_boxes", "iou_threshold", "score_threshold"],
            outputs=["selected_detections"],
            name="postprocess_nms"
        )
        
        yolo_model.graph.initializer.extend([max_boxes_init, iou_thresh_init, score_thresh_init])
        yolo_model.graph.input.extend([boxes_shape, scores_shape])
        yolo_model.graph.node.append(nms_node)
        yolo_model.graph.output.append(onnx.helper.make_tensor_value_info("selected_detections", onnx.TensorProto.INT64, [None, 3]))
        
        nms_model_path = models_dir / "yolov8_with_nms.onnx"
        onnx.save(yolo_model, str(nms_model_path))
        print(f"✓ Generated {nms_model_path} with NonMaxSuppression node ({round(nms_model_path.stat().st_size / 1024, 1)} KB)")
    except Exception as e:
        print(f"Warning: could not append NMS: {e}")

    print("All sample models generated successfully.")


if __name__ == "__main__":
    main()
