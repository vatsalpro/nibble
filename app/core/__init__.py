"""Core package initialization."""
from app.core.hardware_manager import HardwareManager, HardwareProfile
from app.core.project_manager import ProjectManager
from app.core.model_manager import ModelManager
from app.core.optimization_manager import OptimizationManager
from app.core.benchmark_manager import BenchmarkManager
from app.core.backend_selector import BackendSelector

__all__ = [
    "HardwareManager", "HardwareProfile",
    "ProjectManager", "ModelManager", "OptimizationManager",
    "BenchmarkManager", "BackendSelector"
]
