"""
Model Import View for SnapForge.
Supports importing ONNX and PyTorch models, file browsing, sample model loading,
and instant extraction of model metadata (parameters, input/output tensors, operator count).
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QFileDialog, QLineEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from app.models.model_inspector import ModelInspector, ModelMetadata
from app.core.project_manager import ProjectManager
from app.core.model_manager import ModelManager
from app.ui.animations import HoverPopFilter


class ModelImportView(QWidget):
    # Emitted when a model is successfully imported and inspected
    model_imported = Signal(object, int)  # (ModelMetadata, project_id)
    navigate_to = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_path: Path | None = None
        self.current_metadata: ModelMetadata | None = None
        self.current_project_id: int | None = None
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        container.setMinimumWidth(1160)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(18)

        # Header
        lbl_head = QLabel("Model Import & Ingestion")
        lbl_head.setObjectName("HeaderTitle")
        lbl_sub = QLabel("Select an ONNX or PyTorch model file to inspect architecture and computational graph.")
        lbl_sub.setObjectName("HeaderSubtitle")
        layout.addWidget(lbl_head)
        layout.addWidget(lbl_sub)

        # Import Controls Card
        import_card = QFrame()
        import_card.setObjectName("CardFrame")
        c_layout = QVBoxLayout(import_card)
        c_layout.setSpacing(14)

        # Row 1: File selection
        row1 = QHBoxLayout()
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Select model path (.onnx, .pt, .pth)...")
        self.txt_path.setReadOnly(True)
        row1.addWidget(self.txt_path)

        btn_browse = QPushButton("Browse File...")
        btn_browse.clicked.connect(self._browse_file)
        HoverPopFilter.install(btn_browse, pop_px=3)
        row1.addWidget(btn_browse)

        # Quick sample picker
        self.combo_sample = QComboBox()
        self.combo_sample.addItem("Or Select Sample Model...")
        self.combo_sample.addItem("ResNet-18 Vision Classifier (models/resnet_classifier.onnx)")
        self.combo_sample.addItem("YOLOv8 Detection Head with NMS (models/yolov8_with_nms.onnx)")
        self.combo_sample.currentIndexChanged.connect(self._on_sample_selected)
        row1.addWidget(self.combo_sample)

        c_layout.addLayout(row1)

        # Row 2: Project Metadata
        row2 = QHBoxLayout()
        lbl_pname = QLabel("Project Name:")
        self.txt_project_name = QLineEdit("Snapdragon Optimization Project")
        row2.addWidget(lbl_pname)
        row2.addWidget(self.txt_project_name, stretch=1)

        self.btn_inspect = QPushButton("Inspect & Ingest Model")
        self.btn_inspect.setObjectName("PrimaryButton")
        self.btn_inspect.clicked.connect(self._run_import)
        HoverPopFilter.install(self.btn_inspect, pop_px=3)
        row2.addWidget(self.btn_inspect)

        c_layout.addLayout(row2)
        layout.addWidget(import_card)

        # Metadata Table (Section 9: format, size, params, shapes, ops)
        lbl_meta = QLabel("Model Inspection Metadata")
        lbl_meta.setObjectName("CardTitle")
        layout.addWidget(lbl_meta)

        self.table_meta = QTableWidget()
        self.table_meta.setColumnCount(2)
        self.table_meta.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.table_meta.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_meta.verticalHeader().setVisible(False)
        self.table_meta.verticalHeader().setDefaultSectionSize(44)
        self.table_meta.setShowGrid(False)
        self.table_meta.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_meta.setFocusPolicy(Qt.NoFocus)
        self.table_meta.setAlternatingRowColors(True)
        self.table_meta.setMinimumHeight(400)
        layout.addWidget(self.table_meta, stretch=1)

        # Bottom Bar: Proceed to Analysis
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        self.btn_analyze = QPushButton("Proceed to Snapdragon Analysis →")
        self.btn_analyze.setObjectName("SuccessButton")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._on_proceed_analysis)
        HoverPopFilter.install(self.btn_analyze, pop_px=3)
        bottom_bar.addWidget(self.btn_analyze)

        layout.addLayout(bottom_bar)

        scroll.setWidget(container)
        root_layout.addWidget(scroll)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select AI Model",
            "E:/snapdragon/models",
            "Model Files (*.onnx *.pt *.pth);;ONNX Models (*.onnx);;PyTorch Models (*.pt *.pth)"
        )
        if file_path:
            self.selected_path = Path(file_path)
            self.txt_path.setText(str(self.selected_path))
            self.txt_project_name.setText(f"Project - {self.selected_path.stem}")

    def _on_sample_selected(self, index: int):
        if index == 1:
            p = Path("E:/snapdragon/models/resnet_classifier.onnx")
        elif index == 2:
            p = Path("E:/snapdragon/models/yolov8_with_nms.onnx")
        else:
            return

        if p.exists():
            self.selected_path = p
            self.txt_path.setText(str(p))
            self.txt_project_name.setText(f"Project - {p.stem}")

    def _run_import(self):
        if not self.selected_path or not self.selected_path.exists():
            QMessageBox.warning(self, "Invalid Model", "Please select a valid model file first.")
            return

        try:
            # Create Project in DB
            proj_name = self.txt_project_name.text().strip() or f"Project - {self.selected_path.stem}"
            proj = ProjectManager.create_project(name=proj_name, model_path=str(self.selected_path))
            self.current_project_id = proj.id

            # Import & Inspect
            res = ModelManager.import_model(self.selected_path, project_id=proj.id)
            self.current_metadata = res["metadata"]

            # Populate Table
            meta = self.current_metadata
            attributes = [
                ("Model Name", meta.name),
                ("Format", meta.format),
                ("File Size", f"{meta.file_size_mb} MB ({meta.file_size_bytes:,} bytes)"),
                ("Total Parameters", f"{meta.total_params:,}"),
                ("Estimated FLOPs", f"{meta.total_flops:,}"),
                ("Total Operators / Nodes", f"{len(meta.nodes)} nodes ({len(meta.unique_ops)} unique ops)"),
                ("Primary Data Type", meta.primary_dtype),
                ("Input Tensors", ", ".join([f"{i.name}: {i.shape} ({i.dtype})" for i in meta.inputs])),
                ("Output Tensors", ", ".join([f"{o.name}: {o.shape} ({o.dtype})" for o in meta.outputs])),
                ("ONNX Opset Version", f"Opset {meta.opset_version}"),
                ("Producer / Framework", meta.producer_name),
            ]

            self.table_meta.setRowCount(len(attributes))
            for r, (attr, val) in enumerate(attributes):
                self.table_meta.setItem(r, 0, QTableWidgetItem(attr))
                self.table_meta.setItem(r, 1, QTableWidgetItem(str(val)))

            self.btn_analyze.setEnabled(True)
            self.model_imported.emit(self.current_metadata, self.current_project_id)
            QMessageBox.information(self, "Model Ingested", f"Successfully imported '{meta.name}' with {len(meta.nodes)} operators.")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to inspect model:\n{e}")

    def _on_proceed_analysis(self):
        self.navigate_to.emit("analyze")
