"""
Main application entry point for nibble.
Launches the 'n' -> 'ibble' cursive boot splash animation, followed by
the nibble desktop GUI application in Light Mode on the active interactive display.
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from app.ui.main_window import MainWindow
from app.ui.splash import NibbleSplashScreen
from app.database.database import Database


def ensure_default_desktop():
    """
    On Windows, ensure the current thread is attached to the interactive
    'Default' desktop (WinSta0\\Default) so that the PySide6 GUI window is
    physically rendered on the user's active monitor, even if launched from a
    virtual desktop or command runner environment.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # 0x01FF = DESKTOP_ALL
            h_default = user32.OpenDesktopW("Default", 0, False, 0x01FF)
            if h_default:
                user32.SetThreadDesktop(h_default)
        except Exception:
            pass


def launch_gui():
    """Initializes SQLite database and launches the nibble desktop application with cursive boot screen."""
    ensure_default_desktop()

    # Ensure database is initialized
    Database.get_instance()

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    app.setApplicationName("nibble")
    app.setOrganizationName("nibble AI")
    app.setStyle("Fusion")

    # Set Application Base Typography
    from PySide6.QtGui import QFont
    base_font = QFont("Segoe UI Variable Text", 10)
    base_font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(base_font)

    # Set Application Icon
    icon_path = Path("E:/snapdragon/assets/nibble_icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Instantiate MainWindow (defaults to Light Mode)
    window = MainWindow()

    # Launch 'n' -> 'ibble' boot splash animation
    splash = NibbleSplashScreen(is_dark=window.is_dark_theme)

    def on_splash_finished():
        window.showNormal()
        window.raise_()
        window.activateWindow()

        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = int(window.winId())
                user32 = ctypes.windll.user32
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.ShowWindow(hwnd, 5)  # SW_SHOW
                user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

    splash.finished.connect(on_splash_finished)
    splash.show()

    if sys.platform == "win32":
        try:
            import ctypes
            hwnd_s = int(splash.winId())
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd_s, 5)
            user32.SetForegroundWindow(hwnd_s)
        except Exception:
            pass

    return app.exec()


def main():
    sys.exit(launch_gui())


if __name__ == "__main__":
    main()
