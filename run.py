"""
SnapForge — Snapdragon AI Optimization Studio.
Top-level entry point supporting both GUI and CLI invocations:
- Run 'python run.py' to launch the PySide6 Desktop Application
- Run 'python run.py hardware' to view system hardware telemetry
- Run 'python run.py analyze <model.onnx>' to inspect model compatibility
- Run 'python run.py optimize <model.onnx>' to optimize model
- Run 'python run.py benchmark <model.onnx>' to run empirical benchmarks
- Run 'python run.py report <project_id>' to generate reports
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("gui", "--gui"):
        # Run CLI parser
        from app.cli import main as cli_main
        cli_main()
    else:
        # Launch PySide6 GUI
        from app.main import launch_gui
        sys.exit(launch_gui())


if __name__ == "__main__":
    main()
