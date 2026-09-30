"""
Qualcomm Integration Package for Nibble.
Provides Qualcomm AI Hub cloud-device validation, physical device discovery,
and execution target verification.
"""

from nibble.qualcomm.aihub_models import (
    AIHubDevice,
    AIHubModel,
    AIHubJob,
    AIHubProfileResult,
    AIHubValidationResult,
)
from nibble.qualcomm.aihub_validator import AIHubValidator
from nibble.qualcomm.aihub_results import AIHubResultManager
from nibble.qualcomm.aihub_client import (
    QualcommAIHubClient,
    AIHubNotConfiguredError,
    AIHubModelValidationError,
    AIHubExecutionError,
)

__all__ = [
    "AIHubDevice",
    "AIHubModel",
    "AIHubJob",
    "AIHubProfileResult",
    "AIHubValidationResult",
    "AIHubValidator",
    "AIHubResultManager",
    "QualcommAIHubClient",
    "AIHubNotConfiguredError",
    "AIHubModelValidationError",
    "AIHubExecutionError",
]
