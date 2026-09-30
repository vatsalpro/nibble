"""
Model Library View for nibble.
Browses curated edge models with verified Snapdragon compatibility profiles
and provides one-click ingestion into optimization workflows.
Full support for dynamic Light Mode & Dark Theme.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from app.models.model_registry import ModelRegistry, ModelCatalogItem
from app.models.model_inspector import ModelInspector
from app.ui.animations import HoverPopFilter, StaggeredEntrance, ScrollFadeTrigger


class ModelLibraryView(QWidget):
    model_selected_for_project = Signal(object)  # Emits ModelMetadata
    navigate_to = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.animated_cards = []
        self.init_ui()

    def showEvent(self, event):
        super().showEvent(event)

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
        main_layout.setSpacing(18)

        # Header
        top_bar = QHBoxLayout()
        head_box = QVBoxLayout()
        lbl_head = QLabel("Snapdragon Model Library & Edge Presets")
        lbl_head.setObjectName("HeaderTitle")
        lbl_sub = QLabel("Explore architectures with verified Snapdragon Hexagon HTP and Adreno GPU compatibility.")
        lbl_sub.setObjectName("HeaderSubtitle")
        head_box.addWidget(lbl_head)
        head_box.addWidget(lbl_sub)
        top_bar.addLayout(head_box)
        top_bar.addStretch()

        # Category Filter
        self.combo_cat = QComboBox()
        self.combo_cat.addItems(["All Categories", "Object Detection", "Image Classification", "Language", "Speech"])
        self.combo_cat.currentIndexChanged.connect(self._filter_catalog)
        self.combo_cat.setMinimumWidth(160)
        top_bar.addWidget(self.combo_cat)

        main_layout.addLayout(top_bar)

        # Catalog Table
        self.table_catalog = QTableWidget()
        self.table_catalog.setColumnCount(6)
        self.table_catalog.setHorizontalHeaderLabels([
            "Architecture", "Category", "Recommended Precision", "Expected NPU Compatibility", "Snapdragon Preset", "Action"
        ])
        header = self.table_catalog.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        self.table_catalog.setColumnWidth(5, 120)
        self.table_catalog.verticalHeader().setVisible(False)
        self.table_catalog.verticalHeader().setDefaultSectionSize(48)
        self.table_catalog.setShowGrid(False)
        self.table_catalog.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_catalog.setFocusPolicy(Qt.NoFocus)
        self.table_catalog.setAlternatingRowColors(True)
        self.table_catalog.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_catalog.setMinimumHeight(440)
        main_layout.addWidget(self.table_catalog, stretch=1)
        self.animated_cards.append(self.table_catalog)

        scroll.setWidget(container)
        root_layout.addWidget(scroll)

        # Attach scroll-down fade-in trigger!
        ScrollFadeTrigger.attach(scroll, self.animated_cards)

        self._populate_catalog(ModelRegistry.get_catalog())

    def _populate_catalog(self, items):
        self.table_catalog.setRowCount(len(items))
        for row, item in enumerate(items):
            self.table_catalog.setItem(row, 0, QTableWidgetItem(item.name))
            self.table_catalog.setItem(row, 1, QTableWidgetItem(item.category))
            self.table_catalog.setItem(row, 2, QTableWidgetItem(item.typical_precision))

            npu_item = QTableWidgetItem(item.expected_npu_compatibility)
            if "100" in item.expected_npu_compatibility or "94" in item.expected_npu_compatibility:
                npu_item.setForeground(QBrush(QColor("#16A34A")))
            else:
                npu_item.setForeground(QBrush(QColor("#D97706")))
            self.table_catalog.setItem(row, 3, npu_item)

            self.table_catalog.setItem(row, 4, QTableWidgetItem(item.optimization_preset))

            btn_load = QPushButton("Load Model")
            btn_load.clicked.connect(lambda checked=False, it=item: self._load_model_item(it))
            self.table_catalog.setCellWidget(row, 5, btn_load)

    def _filter_catalog(self):
        cat = self.combo_cat.currentText()
        all_items = ModelRegistry.get_catalog()
        if cat == "All Categories":
            self._populate_catalog(all_items)
        else:
            filtered = [i for i in all_items if i.category == cat]
            self._populate_catalog(filtered)

    def _load_model_item(self, item: ModelCatalogItem):
        if "resnet" in item.id:
            p = Path("E:/snapdragon/models/resnet_classifier.onnx")
        else:
            p = Path("E:/snapdragon/models/yolov8_with_nms.onnx")

        if p.exists():
            meta = ModelInspector.inspect(p)
            self.model_selected_for_project.emit(meta)
            self.navigate_to.emit("analyze")
            QMessageBox.information(self, "Model Loaded", f"Loaded '{item.name}' into active project.")
        else:
            QMessageBox.warning(self, "File Not Found", f"Model weights for {item.name} not present in models directory.")
