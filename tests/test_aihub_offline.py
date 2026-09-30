"""
Offline Unit Tests for Qualcomm AI Hub Integration in Nibble.
Requires NO network access and NO API token.
Validates:
- Pre-upload ONNX model validation
- Qualcomm AI Hub result normalization & null safety
- NPU execution target verification (NPU_CONFIRMED, QNN_CONFIRMED, CPU_EXECUTION, etc.)
- Artifact generation (raw_result.json, normalized_result.json, summary.md)
- Unconfigured client behavior & graceful degradation
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nibble.qualcomm.aihub_client import (
    QualcommAIHubClient,
    AIHubNotConfiguredError,
    AIHubModelValidationError
)
from nibble.qualcomm.aihub_validator import AIHubValidator
from nibble.qualcomm.aihub_results import AIHubResultManager
from nibble.qualcomm.aihub_models import (
    AIHubDevice,
    AIHubModel,
    AIHubJob,
    AIHubProfileResult,
)


class TestAIHubOffline(unittest.TestCase):
    """Offline unit test suite for Qualcomm AI Hub components."""

    def setUp(self):
        self.test_model_path = Path("models/test_cnn.onnx")

    def test_model_validation_valid_onnx(self):
        """Verify pre-upload validation succeeds on a valid ONNX model."""
        if not self.test_model_path.exists():
            self.skipTest(f"Test model {self.test_model_path} not found.")

        meta, proto = QualcommAIHubClient.validate_onnx_model(self.test_model_path)
        self.assertIsInstance(meta, AIHubModel)
        self.assertEqual(meta.name, "test_cnn.onnx")
        self.assertEqual(len(meta.sha256), 64)
        self.assertGreater(meta.size_bytes, 0)
        self.assertGreater(meta.size_mb, 0.0)
        self.assertTrue(len(meta.input_names) > 0)
        self.assertTrue(len(meta.output_names) > 0)

    def test_model_validation_invalid_file(self):
        """Verify pre-upload validation fails on non-existent or non-ONNX files."""
        with self.assertRaises(FileNotFoundError):
            QualcommAIHubClient.validate_onnx_model("models/non_existent.onnx")

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an onnx model")
            bad_path = f.name

        try:
            with self.assertRaises(AIHubModelValidationError):
                QualcommAIHubClient.validate_onnx_model(bad_path)
        finally:
            if os.path.exists(bad_path):
                os.remove(bad_path)

    def test_validator_npu_confirmed(self):
        """Verify NPU_CONFIRMED classification when compute units or layers show HTP/NPU."""
        # Case 1: compute_unit_summary
        raw_profile = {
            "compute_unit_summary": {
                "npu": {"time_percentage": 98.5},
                "cpu": {"time_percentage": 1.5}
            },
            "layers": [
                {"name": "conv1", "compute_unit": "NPU"},
                {"name": "relu1", "compute_unit": "NPU"},
                {"name": "output", "compute_unit": "CPU"}
            ]
        }
        target, notes, npu_l, cpu_l, tot_l = AIHubValidator.verify_execution_target(raw_profile)
        self.assertEqual(target, "NPU_CONFIRMED")
        self.assertEqual(npu_l, 2)
        self.assertEqual(cpu_l, 1)
        self.assertEqual(tot_l, 3)

        # Case 2: HTP compute unit string
        raw_profile_htp = {
            "compute_unit": "htp",
            "layers": [{"name": "dense", "accelerator": "HTP"}]
        }
        target, notes, npu_l, cpu_l, tot_l = AIHubValidator.verify_execution_target(raw_profile_htp)
        self.assertEqual(target, "NPU_CONFIRMED")

    def test_validator_qnn_confirmed(self):
        """Verify QNN_CONFIRMED classification when execution summary indicates QNN."""
        raw_profile = {
            "execution_summary": {
                "runtime": "QNN Execution Provider",
                "accelerator": "Qualcomm QNN"
            },
            "layers": [
                {"name": "op1", "compute_unit": "UNKNOWN"}
            ]
        }
        target, notes, npu_l, cpu_l, tot_l = AIHubValidator.verify_execution_target(raw_profile)
        self.assertEqual(target, "QNN_CONFIRMED")

    def test_validator_cpu_execution(self):
        """Verify CPU_EXECUTION when all layers run on CPU."""
        raw_profile = {
            "compute_unit_summary": {
                "cpu": {"time_percentage": 100.0}
            },
            "layers": [
                {"name": "conv1", "compute_unit": "CPU"},
                {"name": "relu1", "compute_unit": "CPU"}
            ]
        }
        target, notes, npu_l, cpu_l, tot_l = AIHubValidator.verify_execution_target(raw_profile)
        self.assertEqual(target, "CPU_EXECUTION")
        self.assertEqual(npu_l, 0)
        self.assertEqual(cpu_l, 2)

    def test_validator_gpu_execution(self):
        """Verify GPU_EXECUTION when compute units report GPU/Adreno."""
        raw_profile = {
            "compute_unit_summary": {
                "gpu": {"time_percentage": 100.0}
            },
            "layers": [
                {"name": "conv1", "compute_unit": "GPU"}
            ]
        }
        target, notes, npu_l, cpu_l, tot_l = AIHubValidator.verify_execution_target(raw_profile)
        self.assertEqual(target, "GPU_EXECUTION")

    def test_validator_unknown_when_empty(self):
        """Verify UNKNOWN classification when no telemetry or layers are present."""
        raw_profile = {"status": "SUCCESS"}
        target, notes, npu_l, cpu_l, tot_l = AIHubValidator.verify_execution_target(raw_profile)
        self.assertEqual(target, "UNKNOWN")

    def test_normalization_and_artifact_export(self):
        """Verify result normalization schema and file exports."""
        raw_result = {
            "status": "SUCCESS",
            "execution_summary": {
                "inference_time": 4.5,
                "time_unit": "ms"
            },
            "memory": {
                "peak_memory_bytes": 10485760  # 10 MB
            },
            "compute_unit_summary": {
                "npu": {"time_percentage": 95.0}
            },
            "layers": [
                {"name": "conv", "compute_unit": "NPU"},
                {"name": "pool", "compute_unit": "NPU"}
            ]
        }

        normalized = AIHubResultManager.normalize_profile_result(
            raw_result=raw_result,
            job_id="test_job_12345",
            device_name="Snapdragon X Elite CRD",
            model_name="test_model.onnx",
            model_sha256="abc1234567890abcdef"
        )

        self.assertEqual(normalized["source"], "qualcomm_ai_hub")
        self.assertEqual(normalized["device"], "Snapdragon X Elite CRD")
        self.assertEqual(normalized["execution_target"], "NPU_CONFIRMED")
        self.assertEqual(normalized["latency_ms"], 4.5)
        self.assertAlmostEqual(normalized["memory_mb"], 10.0, places=1)
        self.assertEqual(normalized["throughput_fps"], round(1000.0 / 4.5, 1))
        self.assertTrue(normalized["is_actual_hardware_measurement"])
        self.assertEqual(normalized["layer_breakdown"]["npu_layers"], 2)

        # Test artifact export
        with tempfile.TemporaryDirectory() as tmp_dir:
            job_dir = AIHubResultManager.save_job_artifacts(
                job_id="test_job_12345",
                raw_result=raw_result,
                normalized_result=normalized,
                reports_dir=tmp_dir
            )
            self.assertTrue((job_dir / "raw_result.json").exists())
            self.assertTrue((job_dir / "normalized_result.json").exists())
            self.assertTrue((job_dir / "summary.md").exists())

            # Verify saved JSON contents
            with open(job_dir / "normalized_result.json", "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data["job_id"] if "job_id" in saved_data else saved_data["device"], "Snapdragon X Elite CRD")

    def test_unconfigured_client_status(self):
        """Verify unconfigured client reports NOT CONFIGURED and never invents devices."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "is_file", return_value=False):
                client = QualcommAIHubClient(api_token="")
                self.assertFalse(client.is_configured())
                st = client.get_status()
                self.assertEqual(st["status"], "NOT CONFIGURED")
                self.assertEqual(client.get_masked_token(), "[Not Configured]")

                with self.assertRaises(AIHubNotConfiguredError):
                    client.list_devices()


if __name__ == "__main__":
    unittest.main()
