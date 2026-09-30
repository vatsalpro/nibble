"""
Deployment & Target Selection View for SnapForge.
Automatic Backend Selection, Hybrid NPU+CPU Partitioning visualization,
and Qualcomm QNN runtime deployment preparation.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QRadioButton, QButtonGroup, QTextEdit,
    QProgressBar, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from app.core.hardware_manager import HardwareManager, HardwareProfile
from app.models.model_inspector import ModelMetadata
from app.models.compatibility import SnapdragonCompatibilityEngine, CompatibilityAnalysisResult
from app.core.backend_selector import BackendSelector
from app.models.graph_analyzer import GraphAnalyzer, GraphStructure
from app.ui.animations import HoverPopFilter


class DeployView(QWidget):
    navigate_to = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hw_profile: HardwareProfile = HardwareManager.get_hardware_profile()
        self.metadata: ModelMetadata | None = None
        self.compat: CompatibilityAnalysisResult | None = None
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(28, 24, 28, 28)
        main_layout.setSpacing(16)

        # Header
        top_bar = QHBoxLayout()
        head_box = QVBoxLayout()
        lbl_head = QLabel("Deployment & Execution Target Selection")
        lbl_head.setObjectName("HeaderTitle")
        lbl_sub = QLabel("Select or automatically determine the optimal execution backend for Snapdragon deployment.")
        lbl_sub.setObjectName("HeaderSubtitle")
        head_box.addWidget(lbl_head)
        head_box.addWidget(lbl_sub)
        top_bar.addLayout(head_box)
        top_bar.addStretch()

        btn_bench = QPushButton("Proceed to Benchmark →")
        btn_bench.setObjectName("SuccessButton")
        btn_bench.clicked.connect(lambda: self.navigate_to.emit("benchmark"))
        HoverPopFilter.install(btn_bench, pop_px=3)
        top_bar.addWidget(btn_bench)
        main_layout.addLayout(top_bar)

        # Automatic Recommendation Card (Section 23)
        rec_card = QFrame()
        rec_card.setObjectName("CardFrame")
        r_layout = QVBoxLayout(rec_card)
        r_layout.setSpacing(8)

        lbl_r_title = QLabel("Automatic Backend Recommendation Engine")
        lbl_r_title.setObjectName("CardTitle")
        r_layout.addWidget(lbl_r_title)

        self.txt_reasoning = QTextEdit()
        self.txt_reasoning.setReadOnly(True)
        self.txt_reasoning.setMaximumHeight(85)
        self.txt_reasoning.setStyleSheet("font-size: 13px; font-weight: 500;")
        self.txt_reasoning.setText("Recommendation will be generated based on model compatibility and hardware capabilities.")
        r_layout.addWidget(self.txt_reasoning)

        main_layout.addWidget(rec_card)

        # Execution Target Selection Options (Section 20 & 21)
        targets_card = QFrame()
        targets_card.setObjectName("CardFrame")
        t_layout = QVBoxLayout(targets_card)
        t_layout.setSpacing(12)

        lbl_t_head = QLabel("Available Target Backends")
        lbl_t_head.setObjectName("CardTitle")
        t_layout.addWidget(lbl_t_head)

        self.btn_group = QButtonGroup(self)

        # Option 1: Snapdragon NPU
        npu_desc = "Qualcomm Hexagon Tensor Processor (HTP) • Peak TOPS/Watt efficiency • INT8 / FP16"
        if not self.hw_profile.npu_present:
            npu_desc += " [Unavailable on current host - Integration Point]"
        self.rad_npu = QRadioButton("Snapdragon Hexagon NPU (HTP)")
        self.rad_npu.setChecked(True)
        self.btn_group.addButton(self.rad_npu, 1)
        t_layout.addWidget(self.rad_npu)
        t_layout.addWidget(self._create_target_desc(npu_desc))

        # Option 2: Hybrid NPU + CPU
        self.rad_hybrid = QRadioButton("Hybrid Execution (Hexagon NPU + CPU Fallback)")
        self.btn_group.addButton(self.rad_hybrid, 2)
        t_layout.addWidget(self.rad_hybrid)
        t_layout.addWidget(self._create_target_desc("Accelerate supported vision/backbone layers on NPU; route dynamic ops and NMS to CPU host."))

        # Option 3: Snapdragon GPU (Adreno / DirectML)
        self.rad_gpu = QRadioButton("Qualcomm Adreno GPU (DirectML)")
        self.btn_group.addButton(self.rad_gpu, 3)
        t_layout.addWidget(self.rad_gpu)
        t_layout.addWidget(self._create_target_desc("Full float and FP16 support via Direct3D 12 DirectML execution provider."))

        # Option 4: CPU
        self.rad_cpu = QRadioButton("Host CPU (ARM64 / x86_64 NEON/AVX)")
        self.btn_group.addButton(self.rad_cpu, 4)
        t_layout.addWidget(self.rad_cpu)
        t_layout.addWidget(self._create_target_desc("Universal ONNX Runtime CPUExecutionProvider with multi-core parallelism."))

        main_layout.addWidget(targets_card)

        # Hybrid Partition Visualization Card (Section 21)
        partition_card = QFrame()
        partition_card.setObjectName("CardFrame")
        p_layout = QVBoxLayout(partition_card)
        p_layout.setSpacing(8)

        lbl_p_title = QLabel("Hybrid Workload Partitioning Breakdown")
        lbl_p_title.setObjectName("CardTitle")
        p_layout.addWidget(lbl_p_title)

        p_row = QHBoxLayout()
        self.lbl_npu_pct = QLabel("NPU Workload: 100.0%")
        self.lbl_npu_pct.setStyleSheet("color: #2EA043; font-weight: bold; font-size: 14px;")
        self.lbl_cpu_pct = QLabel("CPU Fallback: 0.0%")
        self.lbl_cpu_pct.setStyleSheet("color: #F85149; font-weight: bold; font-size: 14px;")
        p_row.addWidget(self.lbl_npu_pct)
        p_row.addWidget(self.lbl_cpu_pct)
        p_row.addStretch()
        p_layout.addLayout(p_row)

        self.partition_bar = QProgressBar()
        self.partition_bar.setRange(0, 100)
        self.partition_bar.setValue(100)
        self.partition_bar.setStyleSheet("""
            QProgressBar { background-color: #DC2626; border-radius: 6px; text-align: center; }
            QProgressBar::chunk { background-color: #16A34A; }
        """)
        p_layout.addWidget(self.partition_bar)

        self.lbl_partition_details = QLabel("100% of computational graph executes on Snapdragon NPU. Zero host transitions.")
        self.lbl_partition_details.setStyleSheet("color: #64748B; font-size: 12px;")
        p_layout.addWidget(self.lbl_partition_details)

        main_layout.addWidget(partition_card)

        scroll.setWidget(container)
        root_layout.addWidget(scroll)

    def _create_target_desc(self, text: str) -> QLabel:
        lbl = QLabel(f"    {text}")
        lbl.setStyleSheet("color: #64748B; font-size: 11.5px; margin-bottom: 6px;")
        return lbl

    def set_model(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.compat = SnapdragonCompatibilityEngine.analyze(metadata)
        hw = HardwareManager.get_hardware_profile()

        # Run Backend Selector
        selection = BackendSelector.select_best_backend(self.metadata, self.compat, hw)
        self.txt_reasoning.setText(f"RECOMMENDED TARGET: {selection['selected_backend']}\n\n{selection['reasoning']}")

        # Auto select radio button
        target_name = selection["selected_backend"].lower()
        if "hybrid" in target_name:
            self.rad_hybrid.setChecked(True)
        elif "npu" in target_name and hw.npu_present:
            self.rad_npu.setChecked(True)
        elif "gpu" in target_name:
            self.rad_gpu.setChecked(True)
        else:
            self.rad_cpu.setChecked(True)

        # Update Partition Visualization
        graph_struct = GraphAnalyzer.analyze_graph(self.metadata, self.compat)
        npu_pct = graph_struct.npu_compute_pct
        cpu_pct = graph_struct.cpu_fallback_compute_pct

        self.lbl_npu_pct.setText(f"NPU Workload: {npu_pct}%")
        self.lbl_cpu_pct.setText(f"CPU Fallback: {cpu_pct}%")
        self.partition_bar.setValue(int(npu_pct))

        if cpu_pct > 0:
            self.lbl_partition_details.setText(
                f"Graph partition boundary: {npu_pct}% compute on Hexagon HTP, {cpu_pct}% compute on CPU "
                f"({graph_struct.boundary_crossings} boundary transitions). Fallback blocker: {self.compat.primary_bottleneck or 'dynamic nodes'}."
            )
        else:
            self.lbl_partition_details.setText("100% of computational graph executes natively on Snapdragon NPU without host roundtrips.")
