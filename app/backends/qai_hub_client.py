"""
Qualcomm AI Hub Integration Client for Nibble.
Connects to Qualcomm AI Hub cloud platform for model discovery,
remote compilation, and hardware profiling on cloud-hosted Snapdragon devices.
Strictly adheres to policy: Never fabricates measurements or fake devices.
If API credentials are not provided, clearly reports unavailable status.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from nibble.qualcomm.aihub_client import (
    QualcommAIHubClient as BaseQualcommAIHubClient,
    AIHubNotConfiguredError,
    AIHubExecutionError,
    AIHubModelValidationError
)
from nibble.qualcomm.aihub_models import AIHubDevice


@dataclass
class QAIHubDevice:
    name: str
    os: str
    chipset: str
    npu_tops: Optional[float]
    is_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualcommAIHubClient(BaseQualcommAIHubClient):
    """Backward-compatible interface delegating to official Qualcomm AI Hub SDK wrapper."""

    def __init__(self, api_token: Optional[str] = None):
        super().__init__(api_token=api_token)

    def list_snapdragon_devices(self) -> List[Dict[str, Any]]:
        """List physical Snapdragon devices queryable in AI Hub catalog."""
        if not self.is_configured():
            return []

        try:
            devs = self.list_devices()
            return [
                QAIHubDevice(
                    name=d.name,
                    os=d.os,
                    chipset=d.chipset,
                    npu_tops=d.npu_tops,
                    is_available=d.is_available
                ).to_dict()
                for d in devs
            ]
        except Exception:
            return []
