"""
Optional Live Cloud Tests for Qualcomm AI Hub Integration in Nibble.
Only runs if QAI_HUB_API_TOKEN is provided in environment or configured in ~/.qai_hub/client.ini.
Otherwise automatically skips with an informative message.
"""

import os
import unittest
from pathlib import Path
from nibble.qualcomm.aihub_client import QualcommAIHubClient


class TestAIHubLive(unittest.TestCase):
    """Live cloud test suite requiring valid Qualcomm AI Hub credentials."""

    @classmethod
    def setUpClass(cls):
        cls.client = QualcommAIHubClient()
        if not cls.client.is_configured():
            raise unittest.SkipTest(
                "Qualcomm AI Hub credentials not configured. "
                "Set QAI_HUB_API_TOKEN or configure ~/.qai_hub/client.ini to run live cloud validation."
            )

    def test_live_connection_and_device_enumeration(self):
        """Test live connectivity and query physical Snapdragon devices."""
        status = self.client.get_status()
        self.assertTrue(status.get("configured", False))

        # Query physical devices
        devices = self.client.list_devices()
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0, "Expected at least one device in Qualcomm AI Hub catalog.")

        # Check for Snapdragon device family
        snapdragon_devices = [d for d in devices if "snapdragon" in d.name.lower() or "snapdragon" in d.chipset.lower()]
        self.assertGreater(len(snapdragon_devices), 0, "Expected Snapdragon devices in catalog.")

        first_dev = snapdragon_devices[0]
        self.assertTrue(len(first_dev.name) > 0)
        self.assertTrue(len(first_dev.os) > 0)


if __name__ == "__main__":
    unittest.main()
