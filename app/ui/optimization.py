"""
Optimization Studio View for SnapForge.
AI Optimization Planner, Objective Selector, Quantization, Operator Fusion,
and non-blocking execution with real-time transformation auditing.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QCheckBox, QProgressBar, QTextEdit,
    QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from app.models.model_inspector import ModelMetadata
from app.models.compatibility import CompatibilityAnalysisResult
from app.optimization.optimization_planner import (
    OptimizationPlanner, OptimizationPlan, OptimizationObjective
)
from app.core.optimization_manager import OptimizationManager
from app.core.hardware_manager import HardwareManager
from app.ui.animations import HoverPopFilter


class OptimizationWorker(QThread):
    """Runs model optimization in a background thread."""
    progress = Signal(str)
    finished_success = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, model_path: str, project_id: int, model_id: int, backend: str, strategy: str, precision: str, fusion: bool, graph_opt: bool):
        super().__init__()
        self.model_path = model_path
        self.project_id = project_id
        self.model_id = model_id
        self.backend = backend
        self.strategy = strategy
        self.precision = precision
        self.fusion = fusion
        self.graph_opt = graph_opt

    def run(self):
        try:
            self.progress.emit("Initiating Snapdragon optimization pipeline...")
            res = OptimizationManager.run_optimization(
                input_model_path=self.model_path,
                project_id=self.project_id,
                model_id=self.model_id,
                target_backend=self.backend,
                strategy=self.strategy,
                precision=self.precision,
                enable_fusion=self.fusion,
                enable_graph_opt=self.graph_opt
            )
            self.finished_success.emit(res)
        except Exception as e:
            self.finished_error.emit(str(e))


class OptimizationView(QWidget):
    navigate_to = Signal(str)
    optimization_completed = Signal(str, dict)  # (optimized_model_path, opt_result)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metadata: ModelMetadata | None = None
        self.project_id: int = 1
        self.model_id: int = 1
        self.current_plan: OptimizationPlan | None = None
        self.worker: OptimizationWorker | None = None
        self.last_optimized_path: str | None = None
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        container.setMinimumWidth(1160)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(28, 24, 28, 28)
        main_layout.setSpacing(16)

        # Header
        top_bar = QHBoxLayout()
        head_box = QVBoxLayout()
        lbl_head = QLabel("AI Optimization Studio")
        lbl_head.setObjectName("HeaderTitle")
        lbl_sub = QLabel("Formulates and applies verified precision quantization, operator fusion, and graph transformations.")
        lbl_sub.setObjectName("HeaderSubtitle")
        head_box.addWidget(lbl_head)
        head_box.addWidget(lbl_sub)
        top_bar.addLayout(head_box)
        top_bar.addStretch()

        self.btn_benchmark = QPushButton("Proceed to Benchmark →")
        self.btn_benchmark.setObjectName("SuccessButton")
        self.btn_benchmark.setEnabled(False)
        self.btn_benchmark.clicked.connect(lambda: self.navigate_to.emit("benchmark"))
        HoverPopFilter.install(self.btn_benchmark, pop_px=3)
        top_bar.addWidget(self.btn_benchmark)
        main_layout.addLayout(top_bar)

        # Objective & Strategy Configuration Card
        config_card = QFrame()
        config_card.setObjectName("CardFrame")
        c_layout = QVBoxLayout(config_card)
        c_layout.setSpacing(12)

        lbl_c_title = QLabel("Optimization Objectives & Hardware Presets")
        lbl_c_title.setObjectName("CardTitle")
        c_layout.addWidget(lbl_c_title)

        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("User Objective:"))
        self.combo_objective = QComboBox()
        self.combo_objective.addItems([
            OptimizationObjective.BALANCED,
            OptimizationObjective.MAX_PERFORMANCE,
            OptimizationObjective.MAX_BATTERY,
            OptimizationObjective.MAX_ACCURACY
        ])
        self.combo_objective.currentIndexChanged.connect(self._recalculate_plan)
        row_cfg.addWidget(self.combo_objective)

        row_cfg.addWidget(QLabel("Target Precision:"))
        self.combo_precision = QComboBox()
        self.combo_precision.addItems(["INT8", "FP16", "FP32"])
        row_cfg.addWidget(self.combo_precision)

        row_cfg.addWidget(QLabel("Target Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["Snapdragon NPU", "Hybrid (NPU + CPU)", "Snapdragon GPU", "CPU"])
        row_cfg.addWidget(self.combo_backend)

        c_layout.addLayout(row_cfg)

        # Toggles Row
        row_toggles = QHBoxLayout()
        self.chk_fusion = QCheckBox("Enable Conv + BatchNormalization Folding")
        self.chk_fusion.setChecked(True)
        row_toggles.addWidget(self.chk_fusion)

        self.chk_graph_opt = QCheckBox("Enable Level 1 Constant Folding & Dead Node Removal")
        self.chk_graph_opt.setChecked(True)
        row_toggles.addWidget(self.chk_graph_opt)

        self.chk_calib = QCheckBox("Use Synthetic Calibration Dataset (Static QDQ)")
        self.chk_calib.setChecked(True)
        row_toggles.addWidget(self.chk_calib)

        row_toggles.addStretch()
        c_layout.addLayout(row_toggles)
        main_layout.addWidget(config_card)

        # Middle: AI Optimization Plan Display
        plan_card = QFrame()
        plan_card.setObjectName("CardFrame")
        p_layout = QVBoxLayout(plan_card)
        p_layout.setSpacing(10)

        lbl_plan_title = QLabel("Synthesized AI Optimization Plan")
        lbl_plan_title.setObjectName("CardTitle")
        p_layout.addWidget(lbl_plan_title)

        self.txt_plan = QTextEdit()
        self.txt_plan.setReadOnly(True)
        self.txt_plan.setMaximumHeight(160)
        self.txt_plan.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        p_layout.addWidget(self.txt_plan)

        # Action Button & Progress
        act_row = QHBoxLayout()
        self.btn_execute = QPushButton("Apply Optimizations Now")
        self.btn_execute.setObjectName("PrimaryButton")
        self.btn_execute.clicked.connect(self._execute_optimization)
        HoverPopFilter.install(self.btn_execute, pop_px=3)
        act_row.addWidget(self.btn_execute)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        act_row.addWidget(self.progress_bar, stretch=1)

        p_layout.addLayout(act_row)
        main_layout.addWidget(plan_card)

        # Bottom: Transformation Log
        log_card = QFrame()
        log_card.setObjectName("CardFrame")
        l_layout = QVBoxLayout(log_card)
        l_layout.setSpacing(6)

        lbl_log = QLabel("Transformation & Optimization Audit Log")
        lbl_log.setObjectName("CardTitle")
        l_layout.addWidget(lbl_log)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(180)
        self.txt_log.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        self.txt_log.setPlaceholderText("Audit log will stream here when optimization executes...")
        l_layout.addWidget(self.txt_log)

        main_layout.addWidget(log_card)

        scroll.setWidget(container)
        root_layout.addWidget(scroll)

    def set_model(self, metadata: ModelMetadata, project_id: int):
        self.metadata = metadata
        self.project_id = project_id
        self._recalculate_plan()

    def _recalculate_plan(self):
        if not self.metadata:
            return

        from app.models.compatibility import SnapdragonCompatibilityEngine
        compat = SnapdragonCompatibilityEngine.analyze(self.metadata)
        hw = HardwareManager.get_hardware_profile()
        obj = self.combo_objective.currentText()

        self.current_plan = OptimizationPlanner.create_plan(self.metadata, compat, objective=obj, hardware=hw)

        # Render Plan Text
        lines = [
            f"=== SNAPDRAGON AI OPTIMIZATION PLAN (Objective: {obj}) ===",
            f"Summary: {self.current_plan.summary}",
            f"Target Backend: {self.current_plan.target_backend} | Target Precision: {self.current_plan.target_precision}",
            f"Hardware Context: {self.current_plan.hardware_notes}\n",
            "PLANNED STEPS:"
        ]
        for step in self.current_plan.steps:
            lines.append(f"  [{step.step_number}] {step.name}")
            lines.append(f"      Action: {step.description}")
            lines.append(f"      Rationale: {step.rationale}")
            lines.append(f"      Risk: {step.accuracy_risk} | Expected Benefit: {step.expected_benefit}\n")

        self.txt_plan.setText("\n".join(lines))
        self.combo_precision.setCurrentText(self.current_plan.target_precision)

    def _execute_optimization(self):
        if not self.metadata:
            # Fallback to sample model if not loaded yet
            p = Path("E:/snapdragon/models/resnet_classifier.onnx")
            if p.exists():
                from app.models.model_inspector import ModelInspector
                self.metadata = ModelInspector.inspect(p)
                self.project_id = 1
                self.model_id = 1
            else:
                QMessageBox.warning(self, "No Model", "Please import an AI model first.")
                return

        self.btn_execute.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # indeterminate pulse
        self.txt_log.clear()

        backend = self.combo_backend.currentText()
        strategy = self.combo_objective.currentText()
        precision = self.combo_precision.currentText()
        fusion = self.chk_fusion.isChecked()
        graph_opt = self.chk_graph_opt.isChecked()

        self.worker = OptimizationWorker(
            model_path=self.metadata.file_path,
            project_id=self.project_id,
            model_id=self.model_id,
            backend=backend,
            strategy=strategy,
            precision=precision,
            fusion=fusion,
            graph_opt=graph_opt
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished_success.connect(self._on_worker_success)
        self.worker.finished_error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, msg: str):
        self.txt_log.append(f"• {msg}")

    def _on_worker_success(self, res: dict):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.btn_execute.setEnabled(True)
        self.btn_benchmark.setEnabled(True)

        self.last_optimized_path = res["optimized_model_path"]
        self.txt_log.setText(res["log_text"])
        self.optimization_completed.emit(self.last_optimized_path, res)

        QMessageBox.information(
            self,
            "Optimization Complete",
            f"Successfully optimized model!\n\n"
            f"Original Size: {res['original_size_mb']} MB\n"
            f"Optimized Size: {res['optimized_size_mb']} MB ({res['size_reduction_pct']}% reduction)\n\n"
            f"Model saved to:\n{res['optimized_model_path']}"
        )

    def _on_worker_error(self, err_msg: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.btn_execute.setEnabled(True)
        self.txt_log.append(f"[ERROR] {err_msg}")
        QMessageBox.critical(self, "Optimization Failed", f"Optimization encountered an error:\n{err_msg}")
