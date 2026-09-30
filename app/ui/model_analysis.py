"""
Model Analysis & Snapdragon Compatibility View for SnapForge.
Visualizes computational graphs, color-codes operators by NPU/GPU/CPU compatibility:
- Green: NPU Supported
- Yellow: Potential Fallback
- Red: NPU Unsupported
Provides operator search, parameter/FLOPs inspection, and bottleneck diagnostics.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QSplitter, QTextEdit, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from app.models.model_inspector import ModelMetadata, NodeInfo
from app.models.compatibility import (
    SnapdragonCompatibilityEngine, CompatibilityAnalysisResult, SupportLevel
)
from app.models.graph_analyzer import GraphAnalyzer, GraphStructure


class ModelAnalysisView(QWidget):
    navigate_to = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metadata: ModelMetadata | None = None
        self.compat: CompatibilityAnalysisResult | None = None
        self.graph_struct: GraphStructure | None = None
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
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Header
        top_bar = QHBoxLayout()
        head_box = QVBoxLayout()
        lbl_head = QLabel("Snapdragon Compatibility & Graph Analysis")
        lbl_head.setObjectName("HeaderTitle")
        lbl_sub = QLabel("Evaluates computational graphs against Qualcomm Hexagon HTP and Adreno GPU architectures.")
        lbl_sub.setObjectName("HeaderSubtitle")
        head_box.addWidget(lbl_head)
        head_box.addWidget(lbl_sub)
        top_bar.addLayout(head_box)
        top_bar.addStretch()

        btn_opt = QPushButton("Proceed to Optimization →")
        btn_opt.setObjectName("SuccessButton")
        btn_opt.clicked.connect(lambda: self.navigate_to.emit("optimize"))
        top_bar.addWidget(btn_opt)
        main_layout.addLayout(top_bar)

        # Compatibility Overview Score Cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)

        self.card_npu_score = self._create_score_card("Snapdragon NPU Score", "— %", "Compute-Weighted FLOPs", "#16A34A")
        self.card_gpu_score = self._create_score_card("Qualcomm Adreno GPU", "— %", "DirectML / Float Precision", "#0284C7")
        self.card_cpu_score = self._create_score_card("Host CPU Fallback", "100.0%", "Universal Execution", "#64748B")
        self.card_bottleneck = self._create_score_card("Primary NPU Bottleneck", "None", "Blockers requiring fallback", "#DC2626")

        cards_row.addWidget(self.card_npu_score)
        cards_row.addWidget(self.card_gpu_score)
        cards_row.addWidget(self.card_cpu_score)
        cards_row.addWidget(self.card_bottleneck)
        main_layout.addLayout(cards_row)

        # Splitter: Left side = Operator table & Graph view, Right side = Inspector details
        splitter = QSplitter(Qt.Horizontal)

        # Left Container
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Search Bar
        search_row = QHBoxLayout()
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Filter operators by type or name (e.g. Conv, Gemm, NMS)...")
        self.txt_filter.textChanged.connect(self._filter_table)
        search_row.addWidget(self.txt_filter)
        left_layout.addLayout(search_row)

        # Legend Bar
        legend_row = QHBoxLayout()
        legend_row.addWidget(QLabel("Compatibility Legend: "))
        legend_row.addWidget(self._create_legend_dot("● NPU Supported", "#16A34A"))
        legend_row.addWidget(self._create_legend_dot("● Partial / Constrained", "#D97706"))
        legend_row.addWidget(self._create_legend_dot("● NPU Unsupported", "#DC2626"))
        legend_row.addStretch()
        left_layout.addLayout(legend_row)

        # Operator Nodes Table
        self.table_nodes = QTableWidget()
        self.table_nodes.setColumnCount(6)
        self.table_nodes.setHorizontalHeaderLabels([
            "Op Type", "Node Identifier", "NPU Support", "GPU Support", "Compute (FLOPs)", "Data Type"
        ])
        header = self.table_nodes.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table_nodes.verticalHeader().setVisible(False)
        self.table_nodes.verticalHeader().setDefaultSectionSize(44)
        self.table_nodes.setShowGrid(False)
        self.table_nodes.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_nodes.setFocusPolicy(Qt.NoFocus)
        self.table_nodes.setAlternatingRowColors(True)
        self.table_nodes.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table_nodes.itemSelectionChanged.connect(self._on_node_selected)
        left_layout.addWidget(self.table_nodes)

        splitter.addWidget(left_widget)

        # Right Container: Detailed Operator Inspector Card
        right_card = QFrame()
        right_card.setObjectName("CardFrame")
        right_layout = QVBoxLayout(right_card)
        right_layout.setSpacing(12)

        lbl_insp_title = QLabel("Selected Operator Inspector")
        lbl_insp_title.setObjectName("CardTitle")
        right_layout.addWidget(lbl_insp_title)

        self.txt_inspector = QTextEdit()
        self.txt_inspector.setReadOnly(True)
        self.txt_inspector.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        self.txt_inspector.setPlaceholderText("Select an operator from the table on the left to inspect its input/output shapes, tensor types, compute requirements, and Qualcomm Hexagon execution constraints.")
        right_layout.addWidget(self.txt_inspector)

        lbl_recs_title = QLabel("Actionable Snapdragon Strategy")
        lbl_recs_title.setObjectName("CardTitle")
        right_layout.addWidget(lbl_recs_title)

        self.txt_recs = QTextEdit()
        self.txt_recs.setReadOnly(True)
        self.txt_recs.setMaximumHeight(140)
        self.txt_recs.setStyleSheet("font-size: 12px;")
        right_layout.addWidget(self.txt_recs)

        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter, stretch=1)
        scroll.setWidget(container)
        root_layout.addWidget(scroll)

    def _create_score_card(self, title: str, value: str, subtext: str, color: str) -> QFrame:
        card = QFrame()
        card.setObjectName("CardFrame")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(4)

        lbl_t = QLabel(title.upper())
        lbl_t.setObjectName("MetricLabel")
        lay.addWidget(lbl_t)

        lbl_v = QLabel(value)
        lbl_v.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {color};")
        card.value_label = lbl_v
        lay.addWidget(lbl_v)

        lbl_s = QLabel(subtext)
        lbl_s.setStyleSheet("color: #8B949E; font-size: 11px;")
        card.sub_label = lbl_s
        lay.addWidget(lbl_s)

        return card

    def _create_legend_dot(self, text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold; margin-right: 12px;")
        return lbl

    def set_model(self, metadata: ModelMetadata):
        """Analyze and populate compatibility metrics."""
        self.metadata = metadata
        self.compat = SnapdragonCompatibilityEngine.analyze(metadata)
        self.graph_struct = GraphAnalyzer.analyze_graph(metadata, self.compat)

        # Update Score Cards
        self.card_npu_score.value_label.setText(f"{self.compat.weighted_npu_score}%")
        self.card_npu_score.sub_label.setText(f"Op count: {self.compat.npu_score}% ({self.compat.supported_nodes_count}/{self.compat.total_nodes})")

        self.card_gpu_score.value_label.setText(f"{self.compat.gpu_score}%")
        self.card_cpu_score.value_label.setText(f"{self.compat.cpu_score}%")

        if self.compat.primary_bottleneck:
            self.card_bottleneck.value_label.setText(self.compat.primary_bottleneck[:20] + "...")
            self.card_bottleneck.sub_label.setText(self.compat.primary_bottleneck)
        else:
            self.card_bottleneck.value_label.setText("Zero Blockers")
            self.card_bottleneck.value_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2EA043;")
            self.card_bottleneck.sub_label.setText("100% NPU Native")

        # Update Recommendations
        rec_lines = [f"• {r}" for r in self.compat.recommendations]
        self.txt_recs.setText("\n".join(rec_lines))

        # Populate Table
        self._populate_table()

    def _populate_table(self):
        if not self.compat:
            return

        self.table_nodes.setRowCount(len(self.compat.node_evaluations))

        for row, node in enumerate(self.compat.node_evaluations):
            item_op = QTableWidgetItem(node.op_type)
            item_id = QTableWidgetItem(node.node_name)
            item_npu = QTableWidgetItem(node.npu_status)
            item_gpu = QTableWidgetItem(node.gpu_status)
            item_flops = QTableWidgetItem(f"{node.estimated_flops:,}")
            item_dtype = QTableWidgetItem(node.data_type)

            # Color coding per Section 12 (Green, Yellow, Red)
            if node.npu_status == SupportLevel.SUPPORTED:
                item_npu.setForeground(QColor("#16A34A"))
                item_op.setForeground(QColor("#16A34A"))
            elif node.npu_status == SupportLevel.PARTIAL:
                item_npu.setForeground(QColor("#D97706"))
                item_op.setForeground(QColor("#D97706"))
            else:
                item_npu.setForeground(QColor("#DC2626"))
                item_op.setForeground(QColor("#DC2626"))

            self.table_nodes.setItem(row, 0, item_op)
            self.table_nodes.setItem(row, 1, item_id)
            self.table_nodes.setItem(row, 2, item_npu)
            self.table_nodes.setItem(row, 3, item_gpu)
            self.table_nodes.setItem(row, 4, item_flops)
            self.table_nodes.setItem(row, 5, item_dtype)

    def _filter_table(self, query: str):
        query = query.lower().strip()
        for r in range(self.table_nodes.rowCount()):
            op_text = self.table_nodes.item(r, 0).text().lower()
            name_text = self.table_nodes.item(r, 1).text().lower()
            match = (query in op_text or query in name_text)
            self.table_nodes.setRowHidden(r, not match)

    def _on_node_selected(self):
        selected_rows = self.table_nodes.selectionModel().selectedRows()
        if not selected_rows or not self.compat or not self.metadata:
            return

        row = selected_rows[0].row()
        node_name = self.table_nodes.item(row, 1).text()

        # Find matching NodeInfo & NodeCompatibility
        node_info = next((n for n in self.metadata.nodes if n.name == node_name), None)
        node_comp = next((c for c in self.compat.node_evaluations if c.node_name == node_name), None)

        if not node_info or not node_comp:
            return

        detail_text = (
            f"=== OPERATOR INSPECTION: {node_info.op_type} ===\n\n"
            f"Node Identifier:   {node_info.name}\n"
            f"Primary Data Type: {node_info.data_type}\n"
            f"Parameters:        {node_info.param_count:,}\n"
            f"Estimated Compute: {node_info.estimated_flops:,} FLOPs\n\n"
            f"--- TENSOR SHAPES ---\n"
            f"Inputs ({len(node_info.inputs)}):  {node_info.inputs}\n"
            f"Input Shapes:        {node_info.input_shapes}\n"
            f"Outputs ({len(node_info.outputs)}): {node_info.outputs}\n"
            f"Output Shapes:       {node_info.output_shapes}\n\n"
            f"--- HARDWARE COMPATIBILITY ---\n"
            f"Snapdragon NPU:    [{node_comp.npu_status}]\n"
            f"Qualcomm HTP Note: {node_comp.npu_notes or 'Accelerated natively'}\n\n"
            f"Qualcomm GPU:      [{node_comp.gpu_status}]\n"
            f"DirectML Note:     {node_comp.gpu_notes or 'Supported'}\n\n"
            f"Host CPU Fallback: [{node_comp.cpu_status}]\n"
        )
        self.txt_inspector.setText(detail_text)
