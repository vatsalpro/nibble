"""Unit tests for HardwareManager."""
import unittest
from app.core.hardware_manager import HardwareManager, HardwareProfile


class TestHardwareManager(unittest.TestCase):
    def test_get_hardware_profile(self):
        profile = HardwareManager.get_hardware_profile()
        self.assertIsInstance(profile, HardwareProfile)
        self.assertTrue(len(profile.cpu_arch) > 0)
        self.assertTrue(profile.total_ram_gb > 0)
        self.assertIn(profile.npu_status, ["Available", "Unavailable", "Hardware Detected (Runtime Missing)", "Not detected"])

    def test_to_dict(self):
        profile = HardwareManager.get_hardware_profile()
        d = profile.to_dict()
        self.assertIn("cpu_model", d)
        self.assertIn("cpu_vendor", d)
        self.assertIn("cpu_arch", d)
        self.assertIn("npu_present", d)
        self.assertIn("qnn_status", d)
        self.assertIn("total_ram_gb", d)


if __name__ == "__main__":
    unittest.main()
