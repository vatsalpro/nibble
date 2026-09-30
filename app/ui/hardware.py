"""
Hardware Profiler View for nibble.
Displays deep, genuine system diagnostics on Windows:
CPU, GPU, Qualcomm Hexagon NPU, Memory, Battery, and Qualcomm Toolchain status.
Full support for dynamic Light Mode & Dark Theme with high contrast typography.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from app.core.hardware_manager import HardwareManager, HardwareProfile
from app.profiling.profiler import SystemProfiler
from app.ui.animations import HoverPopFilter, StaggeredEntrance, ScrollFadeTrigger


class HardwareView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.profiler = SystemProfiler()
        self.animated_cards = []
        self.init_ui()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_hardware()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(28, 24, 28, 28)
        main_layout.setSpacing(18)

        # Header
        top_bar = QHBoxLayout()
        head_box = QVBoxLayout()
        lbl_head = QLabel("System Hardware & Qualcomm NPU Profiler")
        lbl_head.setObjectName("HeaderTitle")
        lbl_sub = QLabel("Genuine hardware detection, telemetry, and runtime capability discovery.")
        lbl_sub.setObjectName("HeaderSubtitle")
        head_box.addWidget(lbl_head)
        head_box.addWidget(lbl_sub)
        top_bar.addLayout(head_box)
        top_bar.addStretch()

        btn_refresh = QPushButton("Refresh Telemetry")
        btn_refresh.setObjectName("PrimaryButton")
        btn_refresh.clicked.connect(self.refresh_hardware)
        HoverPopFilter.install(btn_refresh, pop_px=3)
        top_bar.addWidget(btn_refresh)
        main_layout.addLayout(top_bar)

        # Hardware Environment Badges (Requirement 6)
        hw_initial = HardwareManager.get_hardware_profile()
        badge_bar = QHBoxLayout()
        badge_bar.setSpacing(10)
        badge_dev = QLabel(f"🖥️  {hw_initial.hardware_badge_text}")
        badge_dev.setStyleSheet("""
            background: rgba(14, 165, 233, 0.12);
            color: #0284C7;
            border: 1px solid rgba(14, 165, 233, 0.35);
            border-radius: 8px;
            padding: 5px 12px;
            font-weight: 600;
            font-size: 11px;
        """)
        badge_target = QLabel(f"⚡  {hw_initial.target_badge_text}")
        badge_target.setStyleSheet("""
            background: rgba(16, 185, 129, 0.12);
            color: #059669;
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 8px;
            padding: 5px 12px;
            font-weight: 600;
            font-size: 11px;
        """)
        badge_bar.addWidget(badge_dev)
        badge_bar.addWidget(badge_target)
        badge_bar.addStretch()
        main_layout.addLayout(badge_bar)

        # Live Resource Meters Card
        meter_card = QFrame()
        meter_card.setObjectName("CardFrame")
        m_layout = QHBoxLayout(meter_card)
        m_layout.setContentsMargins(18, 14, 18, 14)
        m_layout.setSpacing(24)

        # CPU Meter
        self.lbl_cpu_val = QLabel("CPU: — %")
        self.lbl_cpu_val.setStyleSheet("font-weight: 600; font-size: 13px;")
        m_layout.addWidget(self.lbl_cpu_val)

        # RAM Meter
        self.lbl_ram_val = QLabel("RAM: — GB")
        self.lbl_ram_val.setStyleSheet("font-weight: 600; font-size: 13px;")
        m_layout.addWidget(self.lbl_ram_val)

        # Battery Meter
        self.lbl_batt_val = QLabel("Battery: —")
        self.lbl_batt_val.setStyleSheet("font-weight: 600; font-size: 13px;")
        m_layout.addWidget(self.lbl_batt_val)

        # Qualcomm Status
        self.lbl_qnn_val = QLabel("QNN EP: —")
        self.lbl_qnn_val.setStyleSheet("font-weight: 600; font-size: 13px;")
        m_layout.addWidget(self.lbl_qnn_val)

        m_layout.addStretch()
        main_layout.addWidget(meter_card)
        self.animated_cards.append(meter_card)

        # Detailed Hardware Specs Table
        lbl_tbl = QLabel("Detected Hardware Subsystems & Drivers")
        lbl_tbl.setObjectName("CardTitle")
        main_layout.addWidget(lbl_tbl)

        self.table_hw = QTableWidget()
        self.table_hw.setColumnCount(3)
        self.table_hw.setHorizontalHeaderLabels(["Subsystem", "Detected Specification", "Snapdragon Deployment Assessment"])
        self.table_hw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_hw.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_hw.verticalHeader().setVisible(False)
        self.table_hw.verticalHeader().setDefaultSectionSize(46)
        self.table_hw.setShowGrid(False)
        self.table_hw.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_hw.setFocusPolicy(Qt.NoFocus)
        self.table_hw.setAlternatingRowColors(True)
        self.table_hw.setMinimumHeight(420)
        main_layout.addWidget(self.table_hw, stretch=1)
        self.animated_cards.append(self.table_hw)

        scroll.setWidget(container)
        root_layout.addWidget(scroll)

        # Attach scroll-down fade-in trigger!
        ScrollFadeTrigger.attach(scroll, self.animated_cards)

        self.refresh_hardware()

    def refresh_hardware(self):
        hw: HardwareProfile = HardwareManager.get_hardware_profile(force_refresh=True)
        snap = self.profiler.sample_now()

        # Update Meters
        self.lbl_cpu_val.setText(f"CPU Load: {snap['cpu_util_pct']}% ({snap['cpu_freq_mhz']} MHz)")
        self.lbl_ram_val.setText(f"RAM: {snap['ram_used_gb']} GB / {snap['ram_total_gb']} GB ({snap['ram_pct']}%)")
        batt_str = f"{hw.battery_percent}% (Plugged)" if hw.battery_charging else f"{hw.battery_percent}% (Battery)"
        self.lbl_batt_val.setText(f"Battery: {batt_str if hw.has_battery else 'AC Line'}")
        
        qnn_text = "Active" if hw.ort_qnn_ep_available else "Integration Point"
        qnn_color = "#16A34A" if hw.ort_qnn_ep_available else "#D97706"
        self.lbl_qnn_val.setText(f"Qualcomm QNN EP: {qnn_text}")
        self.lbl_qnn_val.setStyleSheet(f"color: {qnn_color}; font-weight: bold;")

        # Populate Table
        gpu_name = hw.gpu_devices[0] if hw.gpu_devices else "None"
        specs = [
            ("OEM Manufacturer", hw.oem_manufacturer, "Qualcomm Partner OEM" if "hp" in hw.oem_manufacturer.lower() else "Generic PC OEM"),
            ("Computer Model", hw.system_model, "Host Machine"),
            ("Processor Vendor", hw.cpu_vendor, "AMD (Development Machine)" if hw.cpu_vendor == "AMD" else hw.cpu_vendor),
            ("Processor Name", hw.cpu_model, "Snapdragon Native" if hw.is_snapdragon else "Host Development Environment (Non-Snapdragon)"),
            ("Processor Architecture", hw.cpu_arch, "ARM64 (Snapdragon Target)" if hw.is_arm64 else "AMD64 / x64"),
            ("Physical & Logical Cores", f"{hw.cpu_cores_physical} Physical / {hw.cpu_cores_logical} Logical Cores", "Ready for multi-threaded inference"),
            ("Graphics Adapter (GPU)", gpu_name, "Qualcomm Adreno" if hw.has_adreno_gpu else "Host Display Adapter"),
            ("DirectML GPU Acceleration", "Active" if hw.ort_dml_ep_available else "Unavailable", "Available" if hw.ort_dml_ep_available else "Requires onnxruntime-directml"),
            ("Qualcomm Hexagon NPU", hw.npu_name, hw.npu_status),
            ("Qualcomm QNN Provider", hw.qnn_status, "Active" if hw.ort_qnn_ep_available else "Not active (Deployment Target)"),
            ("NPU Integration Details", hw.npu_details, "Ready on Snapdragon" if hw.is_snapdragon else "Hardware Integration Point (AMD Dev Machine)"),
            ("Total System RAM", f"{hw.total_ram_gb} GB Physical Memory", "Optimal for large vision & LLM weights"),
            ("Battery Subsystem", batt_str if hw.has_battery else "AC Desktop Power", "Monitored for energy-aware optimization"),
            ("Operating System", f"{hw.os_name} {hw.os_release} (Build {hw.os_build})", "Windows 11 PC"),
            ("ONNX Runtime Providers", ", ".join(hw.available_ort_providers), "Runtime Providers Active"),
            ("Qualcomm QNN SDK", hw.qnn_sdk_path or "Not Detected in Standard Paths", "Required for offline DLC compilation"),
            ("Qualcomm AI Hub CLI", "Available in PATH" if hw.qai_hub_cli_available else "Not Configured", "Optional Cloud Profiling Tool"),
        ]

        self.table_hw.setRowCount(len(specs))
        for r, (sub, spec, assess) in enumerate(specs):
            self.table_hw.setItem(r, 0, QTableWidgetItem(sub))
            self.table_hw.setItem(r, 1, QTableWidgetItem(str(spec)))
            it_assess = QTableWidgetItem(assess)
            
            # High-contrast, theme-neutral accessible colors
            if "Active" in assess or "Native" in assess or "Available" in assess or "Supported" in assess:
                it_assess.setForeground(QBrush(QColor("#16A34A")))
            elif "Integration Point" in assess or "Missing" in assess or "Not" in assess:
                it_assess.setForeground(QBrush(QColor("#D97706")))
            self.table_hw.setItem(r, 2, it_assess)
