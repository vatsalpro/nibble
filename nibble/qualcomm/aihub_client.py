"""
Qualcomm AI Hub Client for Nibble.
Interfaces with official Qualcomm AI Hub Python SDK (qai-hub) for cloud model validation,
hardware compilation, and physical device profiling on Snapdragon hardware.
Strictly adheres to policy: Never fabricates measurements, never infers NPU execution without proof,
and gracefully degrades when unconfigured or offline.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import onnx

from nibble.utils.hashing import calculate_file_sha256
from nibble.qualcomm.aihub_models import (
    AIHubDevice,
    AIHubModel,
    AIHubJob,
    AIHubProfileResult,
    AIHubValidationResult,
)
from nibble.qualcomm.aihub_validator import AIHubValidator
from nibble.qualcomm.aihub_results import AIHubResultManager


class AIHubNotConfiguredError(RuntimeError):
    """Raised when an AI Hub operation is requested without API credentials."""
    pass


class AIHubModelValidationError(ValueError):
    """Raised when an ONNX model fails integrity checks before upload."""
    pass


class AIHubExecutionError(RuntimeError):
    """Raised when an AI Hub remote job fails or encounters an unrecoverable API error."""
    pass


class QualcommAIHubClient:
    """Official Qualcomm AI Hub SDK wrapper and job orchestrator."""

    def __init__(self, api_token: Optional[str] = None):
        self._explicit_token = api_token
        self._hub = None
        self._client = None
        self._sdk_available = False
        self._init_sdk()

    def _init_sdk(self):
        """Safely import and initialize qai_hub SDK."""
        try:
            import qai_hub
            self._hub = qai_hub
            self._sdk_available = True

            token = self.get_api_token()
            if token:
                try:
                    cfg = qai_hub.ClientConfig(api_token=token)
                    self._client = qai_hub.Client(config=cfg)
                except Exception:
                    try:
                        self._client = qai_hub.Client()
                    except Exception:
                        pass
            else:
                try:
                    self._client = qai_hub.Client()
                except Exception:
                    pass
        except ImportError:
            self._sdk_available = False

    def is_sdk_installed(self) -> bool:
        """Returns True if qai-hub package is installed in current Python environment."""
        return self._sdk_available

    def get_api_token(self) -> str:
        """Retrieve token from parameter or QAI_HUB_API_TOKEN environment variable."""
        if self._explicit_token:
            return self._explicit_token.strip()
        env_token = os.environ.get("QAI_HUB_API_TOKEN", "").strip()
        if env_token:
            return env_token
        return ""

    @staticmethod
    def _get_ini_path() -> Optional[Path]:
        """Safely locate client.ini even in environments without standard home variables."""
        try:
            return Path.home() / ".qai_hub" / "client.ini"
        except Exception:
            return None

    def get_masked_token(self) -> str:
        """Return masked token for secure UI/CLI status display."""
        token = self.get_api_token()
        if not token:
            ini_path = self._get_ini_path()
            if ini_path and ini_path.exists():
                return "[Configured in ~/.qai_hub/client.ini]"
            return "[Not Configured]"
        if len(token) <= 8:
            return "****"
        return f"{token[:4]}****{token[-4:]}"

    def is_configured(self) -> bool:
        """Verify whether credentials exist in env var or ~/.qai_hub/client.ini."""
        if self.get_api_token():
            return True
        ini_path = self._get_ini_path()
        return bool(ini_path and ini_path.is_file())

    def get_status(self) -> Dict[str, Any]:
        """Check connection and authentication status with Qualcomm AI Hub."""
        if not self.is_sdk_installed():
            return {
                "sdk_installed": False,
                "configured": False,
                "status": "SDK NOT INSTALLED",
                "message": "Qualcomm AI Hub SDK (qai-hub) is not installed. Run 'pip install qai-hub' to enable cloud validation.",
                "action": "Local AMD functionality remains fully active."
            }

        if not self.is_configured():
            return {
                "sdk_installed": True,
                "configured": False,
                "status": "NOT CONFIGURED",
                "token_display": self.get_masked_token(),
                "message": "Qualcomm AI Hub credentials are not configured.",
                "action": "Set the QAI_HUB_API_TOKEN environment variable or run 'qai-hub configure' with your API token."
            }

        # Validate connectivity if credentials present
        try:
            client = self._client or self._hub.Client()
            # Test ping / get_devices
            devs = client.get_devices()
            return {
                "sdk_installed": True,
                "configured": True,
                "status": "AUTHENTICATED & ONLINE",
                "token_display": self.get_masked_token(),
                "device_count": len(devs),
                "message": f"Connected to Qualcomm AI Hub ({len(devs)} cloud devices available)."
            }
        except Exception as e:
            return {
                "sdk_installed": True,
                "configured": True,
                "status": "CONNECTION FAILED",
                "token_display": self.get_masked_token(),
                "message": f"Could not contact Qualcomm AI Hub: {e}",
                "action": "Check network connection or verify API token validity."
            }

    @staticmethod
    def validate_onnx_model(model_path: str | Path) -> Tuple[AIHubModel, onnx.ModelProto]:
        """Validate ONNX model integrity before uploading to Qualcomm AI Hub."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if path.suffix.lower() != ".onnx":
            raise AIHubModelValidationError(f"Expected .onnx file, got '{path.suffix}'.")

        sha256 = calculate_file_sha256(path)
        size_bytes = path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 3)

        try:
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
        except Exception as e:
            raise AIHubModelValidationError(f"ONNX checker validation failed for {path.name}: {e}")

        in_names = [inp.name for inp in model.graph.input]
        out_names = [out.name for out in model.graph.output]

        meta = AIHubModel(
            model_id="",
            name=path.name,
            sha256=sha256,
            size_bytes=size_bytes,
            size_mb=size_mb,
            input_names=in_names,
            output_names=out_names,
        )
        return meta, model

    def list_devices(self, name_filter: str = "") -> List[AIHubDevice]:
        """Retrieve actual available devices from Qualcomm AI Hub."""
        if not self.is_sdk_installed():
            raise AIHubNotConfiguredError("qai-hub package is not installed.")

        if not self.is_configured():
            raise AIHubNotConfiguredError(
                "Qualcomm AI Hub credentials are not configured. Set QAI_HUB_API_TOKEN or configure ~/.qai_hub/client.ini."
            )

        client = self._client or self._hub.Client()
        try:
            hub_devices = client.get_devices(name=name_filter)
        except Exception as e:
            raise AIHubExecutionError(f"Failed to query Qualcomm AI Hub devices: {e}")

        result: List[AIHubDevice] = []
        for d in hub_devices:
            dev_name = getattr(d, "name", str(d))
            dev_os = getattr(d, "os", "Qualcomm Platform")
            attrs = list(getattr(d, "attributes", []))
            chip = "Snapdragon"
            for a in attrs:
                if a.startswith("chipset:"):
                    chip = a.replace("chipset:", "")
                    break

            result.append(AIHubDevice(
                name=dev_name,
                os=dev_os,
                chipset=chip,
                npu_tops=45.0 if "x-elite" in chip.lower() or "x-plus" in chip.lower() else None,
                is_available=True,
                attributes=attrs
            ))
        return result

    def upload_model(self, model_path: str | Path, model_name: Optional[str] = None) -> AIHubModel:
        """Validate and upload an ONNX model to Qualcomm AI Hub."""
        meta, _ = self.validate_onnx_model(model_path)
        name = model_name or meta.name

        if not self.is_configured():
            raise AIHubNotConfiguredError("Qualcomm AI Hub credentials are not configured.")

        client = self._client or self._hub.Client()
        try:
            remote_model = client.upload_model(str(model_path), name=name)
            meta.model_id = remote_model.model_id
            meta.uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return meta
        except Exception as e:
            raise AIHubExecutionError(f"Failed to upload model '{name}' to Qualcomm AI Hub: {e}")

    def submit_profile(
        self,
        model: str | Path | AIHubModel,
        device_name: str,
        options: str = "--compute_unit npu"
    ) -> AIHubJob:
        """Submit a remote profiling job on a physical Qualcomm AI Hub device."""
        if not self.is_configured():
            raise AIHubNotConfiguredError("Qualcomm AI Hub credentials are not configured.")

        client = self._client or self._hub.Client()

        # Handle local model path or uploaded model ID
        if isinstance(model, AIHubModel):
            remote_model = client.get_model(model.model_id)
            model_id_str = model.model_id
        elif isinstance(model, (str, Path)) and Path(model).exists():
            uploaded = self.upload_model(model)
            remote_model = client.get_model(uploaded.model_id)
            model_id_str = uploaded.model_id
        else:
            model_id_str = str(model)
            remote_model = client.get_model(model_id_str)

        target_dev = self._hub.Device(device_name)
        try:
            job = client.submit_profile_job(
                model=remote_model,
                device=target_dev,
                options=options
            )
            return AIHubJob(
                job_id=job.job_id,
                job_type="profile",
                model_id=model_id_str,
                device_name=device_name,
                status="QUEUED",
                url=getattr(job, "url", ""),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        except Exception as e:
            raise AIHubExecutionError(f"Failed to submit profile job to Qualcomm AI Hub: {e}")

    def poll_job(
        self,
        job_id: str,
        timeout_sec: int = 300,
        interval_sec: int = 5
    ) -> AIHubJob:
        """Poll job status until completion, failure, or timeout."""
        client = self._client or self._hub.Client()
        job = client.get_job(job_id)

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            st = job.get_status()
            state_str = str(getattr(st, "state", st)).upper()

            if "SUCCESS" in state_str or "COMPLETED" in state_str:
                return AIHubJob(
                    job_id=job_id,
                    job_type=getattr(job, "job_type", "profile"),
                    model_id="",
                    device_name="",
                    status="COMPLETED",
                    url=getattr(job, "url", "")
                )
            elif "FAILED" in state_str:
                err_msg = getattr(st, "message", "Remote job execution failed.")
                return AIHubJob(
                    job_id=job_id,
                    job_type=getattr(job, "job_type", "profile"),
                    model_id="",
                    device_name="",
                    status="FAILED",
                    url=getattr(job, "url", ""),
                    error_message=err_msg
                )
            elif "CANCEL" in state_str:
                return AIHubJob(
                    job_id=job_id,
                    job_type=getattr(job, "job_type", "profile"),
                    model_id="",
                    device_name="",
                    status="CANCELLED",
                    url=getattr(job, "url", "")
                )

            time.sleep(interval_sec)

        raise TimeoutError(f"Qualcomm AI Hub job {job_id} timed out after {timeout_sec} seconds.")

    def get_profile_result(
        self,
        job_id: str,
        device_name: str = "Snapdragon Device",
        model_name: str = "Model",
        model_sha256: str = "",
        save_artifacts: bool = True
    ) -> AIHubProfileResult:
        """Download raw profile telemetry, verify actual NPU execution, and format result."""
        client = self._client or self._hub.Client()
        job = client.get_job(job_id)

        try:
            raw_profile = job.download_profile()
            if not isinstance(raw_profile, dict):
                raw_profile = {"raw": str(raw_profile)}
        except Exception as e:
            raise AIHubExecutionError(f"Failed to download profile results for job {job_id}: {e}")

        normalized = AIHubResultManager.normalize_profile_result(
            raw_result=raw_profile,
            job_id=job_id,
            device_name=device_name,
            model_name=model_name,
            model_sha256=model_sha256
        )

        if save_artifacts:
            AIHubResultManager.save_job_artifacts(
                job_id=job_id,
                raw_result=raw_profile,
                normalized_result=normalized
            )

        return AIHubProfileResult(
            job_id=job_id,
            device_name=device_name,
            runtime=normalized["runtime"],
            execution_target=normalized["execution_target"],
            verification_notes=normalized["verification"],
            median_latency_ms=normalized["latency_ms"],
            peak_memory_mb=normalized["memory_mb"],
            throughput_fps=normalized["throughput_fps"],
            npu_layers_count=normalized["layer_breakdown"]["npu_layers"],
            cpu_layers_count=normalized["layer_breakdown"]["cpu_layers"],
            total_layers_count=normalized["layer_breakdown"]["total_layers"],
            raw_data=raw_profile
        )
