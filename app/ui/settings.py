"""
Settings View for SnapForge.
Modern Windows 11 & macOS System Settings fusion interface.
Features:
- Visual Theme Switcher (Dark Theme & Light Mode) with instant preview.
- Motion & Animation Controls (Ambient Background, Shake-on-hover, Smooth Transitions).
- Qualcomm AI Hub Cloud Integration settings & live token validation.
- Empirical Benchmarking thresholds & profiling configuration.
- Local Storage & Database maintenance.
"""

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QSpinBox, QMessageBox, QScrollArea,
    QCheckBox, QButtonGroup
)
from PySide6.QtCore import Qt, Signal
from app.backends.qai_hub_client import QualcommAIHubClient
from app.ui.animations import HoverPopFilter, StaggeredEntrance


class SettingsView(QWidget):
    # Signals for global application appearance
    theme_changed = Signal(bool)          # True: Dark, False: Light
    ambient_anim_toggled = Signal(bool)   # Toggle ambient background
    transitions_toggled = Signal(bool)    # Toggle page transitions

    def __init__(self, parent=None, is_dark: bool = True):
        super().__init__(parent)
        self.is_dark = is_dark
        self.hub_client = QualcommAIHubClient()
        self.card_widgets = []
        self.init_ui()

    def set_theme_mode(self, is_dark: bool):
        self.is_dark = is_dark
        self.btn_dark_theme.setChecked(is_dark)
        self.btn_light_theme.setChecked(not is_dark)
        self._update_appearance_card_styles()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(20)

        # Header Title
        title_box = QVBoxLayout()
        lbl_head = QLabel("Preferences & System Settings")
        lbl_head.setObjectName("HeaderTitle")
        lbl_sub = QLabel("Configure visual appearance, motion dynamics, Qualcomm AI Hub credentials, and benchmark thresholds.")
        lbl_sub.setObjectName("HeaderSubtitle")
        title_box.addWidget(lbl_head)
        title_box.addWidget(lbl_sub)
        layout.addLayout(title_box)

        # 1. Appearance & Theme (Windows & Mac Settings Fusion)
        card_theme = self._create_card("Appearance & Display Theme", "Select your preferred visual style. Supports instant dynamic switching.")
        card_theme_lay = QVBoxLayout(card_theme)
        card_theme_lay.setContentsMargins(18, 18, 18, 18)
        card_theme_lay.setSpacing(14)

        theme_cards_row = QHBoxLayout()
        theme_cards_row.setSpacing(14)

        # Dark Theme Option Card
        self.btn_dark_theme = QPushButton("Dark Theme\nQualcomm Engineering Studio")
        self.btn_dark_theme.setCheckable(True)
        self.btn_dark_theme.setChecked(self.is_dark)
        self.btn_dark_theme.setMinimumHeight(64)
        self.btn_dark_theme.clicked.connect(lambda: self._select_theme(True))
        HoverPopFilter.install(self.btn_dark_theme, pop_px=3)
        theme_cards_row.addWidget(self.btn_dark_theme, stretch=1)

        # Light Mode Option Card
        self.btn_light_theme = QPushButton("Light Mode\nCrisp macOS / Fluent Studio")
        self.btn_light_theme.setCheckable(True)
        self.btn_light_theme.setChecked(not self.is_dark)
        self.btn_light_theme.setMinimumHeight(64)
        self.btn_light_theme.clicked.connect(lambda: self._select_theme(False))
        HoverPopFilter.install(self.btn_light_theme, pop_px=3)
        theme_cards_row.addWidget(self.btn_light_theme, stretch=1)

        card_theme_lay.addLayout(theme_cards_row)
        layout.addWidget(card_theme)
        self.card_widgets.append(card_theme)

        # 2. Motion & Dynamic Micro-Interactions
        card_motion = self._create_card("Motion, Transitions & Micro-Interactions", "Smooth transitions, ambient dynamic aura, and snappy feedback.")
        m_lay = QVBoxLayout(card_motion)
        m_lay.setContentsMargins(18, 18, 18, 18)
        m_lay.setSpacing(12)

        self.chk_ambient = QCheckBox("Enable Ambient Background Animation (Living Fluent/Mica Aura)")
        self.chk_ambient.setChecked(True)
        self.chk_ambient.toggled.connect(lambda chk: self.ambient_anim_toggled.emit(chk))
        m_lay.addWidget(self.chk_ambient)

        self.chk_transitions = QCheckBox("Smooth Snappy Page Transitions (Fade & Slide Curve)")
        self.chk_transitions.setChecked(True)
        self.chk_transitions.toggled.connect(lambda chk: self.transitions_toggled.emit(chk))
        m_lay.addWidget(self.chk_transitions)

        self.chk_shake = QCheckBox("Cursor Icon Shake & Jiggle Micro-Interaction on Hover")
        self.chk_shake.setChecked(True)
        m_lay.addWidget(self.chk_shake)

        layout.addWidget(card_motion)
        self.card_widgets.append(card_motion)

        # 3. Qualcomm AI Hub Cloud Integration (Section 18 & 36)
        card_hub = self._create_card("Qualcomm AI Hub Cloud Integration", "Connect to Qualcomm AI Hub for remote cloud compilation and reference device testing.")
        h_lay = QVBoxLayout(card_hub)
        h_lay.setContentsMargins(18, 18, 18, 18)
        h_lay.setSpacing(12)

        lbl_h_desc = QLabel(
            "Enables remote cloud compilation, device profiling on Snapdragon X Elite reference hardware, "
            "and Qualcomm-optimized model discovery. Never hardcodes credentials or responses."
        )
        lbl_h_desc.setStyleSheet("color: #8B949E; font-size: 12px;")
        lbl_h_desc.setWordWrap(True)
        h_lay.addWidget(lbl_h_desc)

        token_row = QHBoxLayout()
        token_row.addWidget(QLabel("API Token:"))
        self.txt_token = QLineEdit()
        self.txt_token.setEchoMode(QLineEdit.Password)
        self.txt_token.setPlaceholderText("Enter QAI_HUB_API_TOKEN or configure via environment variable...")
        existing_token = os.environ.get("QAI_HUB_API_TOKEN", "")
        if existing_token:
            self.txt_token.setText(existing_token)
        token_row.addWidget(self.txt_token, stretch=1)

        btn_test = QPushButton("Test Connection")
        btn_test.clicked.connect(self._test_qai_hub)
        HoverPopFilter.install(btn_test, pop_px=3)
        token_row.addWidget(btn_test)
        h_lay.addLayout(token_row)

        self.lbl_hub_status = QLabel("Status: " + self.hub_client.get_status()["message"])
        self.lbl_hub_status.setStyleSheet("color: #8B949E; font-size: 12px; font-weight: 500;")
        h_lay.addWidget(self.lbl_hub_status)

        layout.addWidget(card_hub)
        self.card_widgets.append(card_hub)

        # 4. Empirical Benchmarking & Profiling Parameters
        card_bench = self._create_card("Benchmarking & Profiling Parameters", "Configure execution iterations and warm-up cycles for empirical timing.")
        b_lay = QVBoxLayout(card_bench)
        b_lay.setContentsMargins(18, 18, 18, 18)
        b_lay.setSpacing(12)

        row_spins = QHBoxLayout()
        row_spins.addWidget(QLabel("Default Warmup Runs:"))
        self.spin_warmup = QSpinBox()
        self.spin_warmup.setValue(10)
        self.spin_warmup.setRange(1, 100)
        row_spins.addWidget(self.spin_warmup)

        row_spins.addSpacing(24)
        row_spins.addWidget(QLabel("Default Measured Runs:"))
        self.spin_measured = QSpinBox()
        self.spin_measured.setValue(100)
        self.spin_measured.setRange(5, 2000)
        row_spins.addWidget(self.spin_measured)
        row_spins.addStretch()

        b_lay.addLayout(row_spins)
        layout.addWidget(card_bench)
        self.card_widgets.append(card_bench)

        # 5. Local Storage & Database Management
        card_storage = self._create_card("Local Storage & Database Maintenance", "Manage local SQLite database, optimization artifacts, and generated reports.")
        s_lay = QVBoxLayout(card_storage)
        s_lay.setContentsMargins(18, 18, 18, 18)
        s_lay.setSpacing(12)

        db_path = Path("E:/snapdragon/snapforge.db")
        lbl_db = QLabel(f"SQLite Database: {db_path.resolve()} ({db_path.stat().st_size // 1024 if db_path.exists() else 0} KB)")
        lbl_db.setStyleSheet("color: #8B949E; font-size: 12px;")
        s_lay.addWidget(lbl_db)

        reports_path = Path("E:/snapdragon/reports")
        report_count = len(list(reports_path.glob("*.pdf"))) if reports_path.exists() else 0
        lbl_rep = QLabel(f"Generated PDF Reports: {report_count} reports in {reports_path.resolve()}")
        lbl_rep.setStyleSheet("color: #8B949E; font-size: 12px;")
        s_lay.addWidget(lbl_rep)

        btn_open_folder = QPushButton("Open Reports Folder")
        btn_open_folder.clicked.connect(lambda: os.startfile(str(reports_path.resolve())) if reports_path.exists() else None)
        HoverPopFilter.install(btn_open_folder, pop_px=3)
        s_lay.addWidget(btn_open_folder)

        layout.addWidget(card_storage)
        self.card_widgets.append(card_storage)

        layout.addStretch()
        scroll.setWidget(container)
        root_layout.addWidget(scroll)

        self._update_appearance_card_styles()

    def showEvent(self, event):
        super().showEvent(event)

    def _create_card(self, title: str, subtitle: str) -> QFrame:
        card = QFrame()
        card.setObjectName("CardFrame")
        return card

    def _select_theme(self, is_dark: bool):
        self.is_dark = is_dark
        self.btn_dark_theme.setChecked(is_dark)
        self.btn_light_theme.setChecked(not is_dark)
        self._update_appearance_card_styles()
        self.theme_changed.emit(is_dark)

    def _update_appearance_card_styles(self):
        if self.is_dark:
            self.btn_dark_theme.setStyleSheet("background-color: #1F2937; border: 2px solid #38BDF8; font-weight: bold; border-radius: 10px; color: #38BDF8; text-align: center;")
            self.btn_light_theme.setStyleSheet("background-color: #151B26; border: 1px solid #222D3D; border-radius: 10px; color: #94A3B8; text-align: center;")
        else:
            self.btn_dark_theme.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; color: #64748B; text-align: center;")
            self.btn_light_theme.setStyleSheet("background-color: #E0F2FE; border: 2px solid #0284C7; font-weight: bold; border-radius: 10px; color: #0284C7; text-align: center;")

    def _test_qai_hub(self):
        token = self.txt_token.text().strip()
        client = QualcommAIHubClient(api_token=token)
        status_info = client.get_status()

        if status_info.get("configured") and "ONLINE" in status_info.get("status", "").upper():
            if token:
                try:
                    qai_dir = Path.home() / ".qai_hub"
                    qai_dir.mkdir(parents=True, exist_ok=True)
                    with open(qai_dir / "client.ini", "w", encoding="utf-8") as f:
                        f.write(f"[api]\napi_token = {token}\n")
                    os.environ["QAI_HUB_API_TOKEN"] = token
                except Exception:
                    pass

            dev_cnt = status_info.get("device_count", 0)
            self.lbl_hub_status.setText(f"Status: Authenticated & Online ({dev_cnt} cloud devices available)")
            self.lbl_hub_status.setStyleSheet("color: #2EA043; font-weight: bold;")
            QMessageBox.information(
                self,
                "Qualcomm AI Hub",
                f"Successfully connected to Qualcomm AI Hub!\n{dev_cnt} Snapdragon cloud devices available for physical profiling."
            )
        else:
            self.lbl_hub_status.setText(f"Status: {status_info.get('status')} — {status_info.get('message')}")
            self.lbl_hub_status.setStyleSheet("color: #F85149; font-weight: bold;")
            QMessageBox.warning(self, "Qualcomm AI Hub", status_info.get("message", "Integration unavailable."))
