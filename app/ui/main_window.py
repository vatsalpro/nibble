"""
Main Window for nibble — Snapdragon AI Optimization Studio.
Assembles sidebar navigation with background app icon hover animation,
unified segmented telemetry header (zero overlaps), buttery smooth transitions,
and enlarged readable typography. Defaults to Light Mode.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QStatusBar, QApplication
)
from PySide6.QtCore import Qt, QSize, QVariantAnimation, QEasingCurve, QRectF, QPointF
from PySide6.QtGui import QIcon, QPixmap, QResizeEvent, QPainter, QColor, QPen, QFont

from app.ui.theme import DARK_STYLESHEET, LIGHT_STYLESHEET, get_stylesheet
from app.ui.animations import (
    HoverPopFilter, AmbientCanvasWidget, AnimatedStackedWidget, StaggeredEntrance
)
from app.ui.icons import VectorIconRenderer
from app.ui.dashboard import DashboardView
from app.ui.model_import import ModelImportView
from app.ui.model_analysis import ModelAnalysisView
from app.ui.optimization import OptimizationView
from app.ui.deploy import DeployView
from app.ui.benchmark import BenchmarkView
from app.ui.hardware import HardwareView
from app.ui.model_library import ModelLibraryView
from app.ui.reports import ReportsView
from app.ui.settings import SettingsView

from app.core.hardware_manager import HardwareManager, HardwareProfile
from app.models.model_inspector import ModelInspector, ModelMetadata


class NavPillButton(QPushButton):
    """
    Sidebar Navigation Button with crisp vector icon, active indicator pill,
    and a subtle animated app icon watermark that smoothly scales and fades
    in the background when the user hovers over it.
    """
    def __init__(self, label: str, tag: str, parent=None):
        super().__init__(parent)
        self.setObjectName("NavButton")
        self.setCheckable(True)
        self.label_text = label
        self.tag = tag
        self.setMinimumHeight(44)
        self.setCursor(Qt.PointingHandCursor)

        self.hover_progress = 0.0
        self._anim = None

        icon_path = Path("E:/snapdragon/assets/nibble_icon.png")
        self.app_icon = QPixmap(str(icon_path)) if icon_path.exists() else None

    def enterEvent(self, event):
        super().enterEvent(event)
        self._animate_hover(1.0)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._animate_hover(0.0)

    def _animate_hover(self, target: float):
        if self._anim:
            self._anim.stop()
        anim = QVariantAnimation(self)
        anim.setDuration(260)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(self.hover_progress)
        anim.setEndValue(target)
        anim.valueChanged.connect(self._on_hover_step)
        self._anim = anim
        anim.start()

    def _on_hover_step(self, val: float):
        self.hover_progress = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        if not painter.begin(self):
            return

        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            w = self.width()
            h = self.height()
            rect = QRectF(4, 2, w - 8, h - 4)

            # Determine theme state
            main_win = self.window()
            is_dark = getattr(main_win, "is_dark_theme", False)

            # 1. Base pill surface
            if self.isChecked():
                bg_color = QColor(224, 242, 254) if not is_dark else QColor(3, 105, 161, 60)
                border_color = QColor(186, 230, 253) if not is_dark else QColor(56, 189, 248, 80)
                painter.setPen(QPen(border_color, 1.2))
                painter.setBrush(bg_color)
                painter.drawRoundedRect(rect, 13, 13)

                # Vibrant active indicator pill on the left
                ind_color = QColor("#0284C7") if not is_dark else QColor("#38BDF8")
                painter.setPen(Qt.NoPen)
                painter.setBrush(ind_color)
                painter.drawRoundedRect(QRectF(8, (h - 22) / 2, 3.5, 22), 1.75, 1.75)
            elif self.hover_progress > 0.01:
                # Smooth hover pill
                alpha = int(self.hover_progress * (40 if is_dark else 70))
                hover_bg = QColor(255, 255, 255, alpha) if is_dark else QColor(226, 232, 240, alpha)
                painter.setPen(Qt.NoPen)
                painter.setBrush(hover_bg)
                painter.drawRoundedRect(rect, 13, 13)

            # 2. ANIMATED APP ICON WATERMARK IN BACKGROUND ON HOVER
            if self.app_icon and self.hover_progress > 0.01:
                scale_px = int(36 + 8 * self.hover_progress)
                scaled_icon = self.app_icon.scaled(scale_px, scale_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                icon_x = w - scale_px - 14
                icon_y = (h - scale_px) / 2.0

                painter.save()
                watermark_alpha = self.hover_progress * (0.28 if not is_dark else 0.22)
                painter.setOpacity(watermark_alpha)
                painter.drawPixmap(QPointF(icon_x, icon_y), scaled_icon)
                painter.restore()

            # 3. Vector Icon (Left-aligned)
            if self.isChecked():
                icon_color = QColor("#0284C7") if not is_dark else QColor("#38BDF8")
            elif self.hover_progress > 0.3:
                icon_color = QColor("#0F172A") if not is_dark else QColor("#F0F6FC")
            else:
                icon_color = QColor("#64748B") if not is_dark else QColor("#8B949E")

            icon_box = QRectF(20, (h - 20) / 2, 20, 20)
            VectorIconRenderer.draw_icon(painter, self.tag, icon_box, icon_color)

            # 4. Label Typography
            font = QFont("Segoe UI Variable Text", 11)
            if self.isChecked():
                font.setWeight(QFont.DemiBold)
                text_color = QColor("#0284C7") if not is_dark else QColor("#38BDF8")
            elif self.hover_progress > 0.3:
                font.setWeight(QFont.Medium)
                text_color = QColor("#0F172A") if not is_dark else QColor("#F8FAFC")
            else:
                font.setWeight(QFont.Normal)
                text_color = QColor("#334155") if not is_dark else QColor("#94A3B8")

            painter.setFont(font)
            painter.setPen(text_color)
            text_rect = QRectF(50, 0, w - 54, h)
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.label_text)

        finally:
            painter.end()


class ThemeTogglePill(QPushButton):
    """
    Crisp vector-rendered toggle pill (zero emoji, subpixel antialiased).
    Renders Sun/Moon vector icons with a sleek rounded surface and smooth hover transition.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setFixedWidth(132)
        self.setCursor(Qt.PointingHandCursor)
        self.hover_progress = 0.0
        self._anim = None
        self.setMouseTracking(True)
        self.setObjectName("ThemeTogglePill")

    def enterEvent(self, event):
        super().enterEvent(event)
        self._animate_hover(1.0)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._animate_hover(0.0)

    def _animate_hover(self, target: float):
        if self._anim:
            self._anim.stop()
        anim = QVariantAnimation(self)
        anim.setDuration(220)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(self.hover_progress)
        anim.setEndValue(target)
        anim.valueChanged.connect(self._on_hover_step)
        self._anim = anim
        anim.start()

    def _on_hover_step(self, val: float):
        self.hover_progress = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        if not painter.begin(self):
            return

        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            w = self.width()
            h = self.height()
            rect = QRectF(1.5, 1.5, w - 3, h - 3)

            main_win = self.window()
            is_dark = getattr(main_win, "is_dark_theme", False)

            # Capsule background and border
            if is_dark:
                alpha = int(self.hover_progress * 40)
                bg = QColor(26, 34, 48, 230 + alpha)
                border = QColor(56, 189, 248, int(70 + 100 * self.hover_progress))
                icon_col = QColor("#38BDF8")
                text_col = QColor("#F0F6FC")
            else:
                alpha = int(self.hover_progress * 30)
                bg = QColor(241, 245, 249, 220 + alpha)
                border = QColor(203, 213, 225, int(160 + 80 * self.hover_progress))
                icon_col = QColor("#0284C7")
                text_col = QColor("#0F172A")

            painter.setPen(QPen(border, 1.3))
            painter.setBrush(bg)
            painter.drawRoundedRect(rect, h / 2, h / 2)

            # Draw vector icon (Sun or Moon)
            tag = "moon" if is_dark else "sun"
            icon_rect = QRectF(14, (h - 18) / 2, 18, 18)
            VectorIconRenderer.draw_icon(painter, tag, icon_rect, icon_col)

            # Text
            painter.setFont(QFont("Segoe UI Variable Text", 9, QFont.DemiBold))
            painter.setPen(text_col)
            mode_text = "Dark Theme" if is_dark else "Light Mode"
            text_rect = QRectF(40, 0, w - 46, h)
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, mode_text)

        finally:
            painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("nibble — Snapdragon AI Optimization Studio")
        self.resize(1360, 880)
        self.setMinimumSize(1140, 740)

        # Set App Window Icon
        icon_path = Path("E:/snapdragon/assets/nibble_icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Default as Light Mode
        self.is_dark_theme: bool = False
        self.setStyleSheet(get_stylesheet(self.is_dark_theme))

        self.hw_profile: HardwareProfile = HardwareManager.get_hardware_profile()
        self.current_metadata: ModelMetadata | None = None
        self.current_project_id: int = 1

        self.init_ui()
        self._load_initial_model_if_present()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if hasattr(self, "ambient_canvas") and self.centralWidget():
            self.ambient_canvas.setGeometry(self.centralWidget().rect())
            self.ambient_canvas.lower()

    def init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        # Ambient Living Background Canvas (slow, serene drifting)
        self.ambient_canvas = AmbientCanvasWidget(central_widget, is_dark=self.is_dark_theme)
        self.ambient_canvas.lower()

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Header Bar (Segmented & Spaced, zero overlaps!)
        root_layout.addWidget(self._create_header_bar())

        # 2. Main Content Split: Sidebar + Animated Stacked Widget
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Left Sidebar (macOS / Windows 11 Settings aesthetic)
        self.sidebar_frame = self._create_sidebar()
        content_layout.addWidget(self.sidebar_frame)

        # Animated Stacked Pages (Buttery smooth transitions)
        self.stacked_widget = AnimatedStackedWidget(transition_ms=240)
        self._init_pages()
        content_layout.addWidget(self.stacked_widget, stretch=1)

        root_layout.addLayout(content_layout)

        # 3. Status Bar
        status_bar = QStatusBar()
        status_bar.setStyleSheet("border-top: 1px solid rgba(128,128,128,0.18); font-size: 12px; padding: 4px 16px;")
        self.lbl_status_model = QLabel("Active Model: None")
        self.lbl_status_target = QLabel("Target: Snapdragon NPU / Host CPU")
        self.lbl_status_mode = QLabel("Mode: Offline-First Local Engine")
        status_bar.addWidget(self.lbl_status_model, stretch=1)
        status_bar.addWidget(self.lbl_status_target, stretch=1)
        status_bar.addPermanentWidget(self.lbl_status_mode)
        self.setStatusBar(status_bar)

        # Select Dashboard by default
        self._switch_page(0, "dashboard")

    def _create_header_bar(self) -> QFrame:
        header = QFrame()
        header.setObjectName("HeaderBar")
        header.setMinimumHeight(86)
        header.setMaximumHeight(96)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(28, 14, 28, 14)
        layout.setSpacing(18)

        # 1. Left Group: Luxury Squircle App Icon + nibble wordmark + subtitle
        brand_box = QHBoxLayout()
        brand_box.setSpacing(12)
        
        icon_path = Path("E:/snapdragon/assets/nibble_icon.png")
        lbl_icon = QLabel()
        if icon_path.exists():
            pix = QPixmap(str(icon_path)).scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_icon.setPixmap(pix)
        HoverPopFilter.install(lbl_icon, pop_px=3)
        brand_box.addWidget(lbl_icon)

        lbl_brand = QLabel("nibble")
        lbl_brand.setStyleSheet("font-size: 24px; font-weight: 800; letter-spacing: -0.5px; color: #0284C7;")
        HoverPopFilter.install(lbl_brand, pop_px=3)
        brand_box.addWidget(lbl_brand)

        lbl_sub = QLabel("• Snapdragon AI Studio")
        lbl_sub.setObjectName("HeaderSubtitle")
        brand_box.addWidget(lbl_sub)
        layout.addLayout(brand_box)

        layout.addStretch()

        # 2. Center Group: Unified Segmented Telemetry Capsule (Super spacious, zero overlap!)
        telemetry_frame = QFrame()
        telemetry_frame.setObjectName("HeaderTelemetryCapsule")
        telemetry_frame.setStyleSheet("""
            QFrame#HeaderTelemetryCapsule {
                background-color: rgba(128,128,128,0.08);
                border: 1px solid rgba(128,128,128,0.20);
                border-radius: 20px;
                padding: 4px 16px;
            }
        """)
        t_lay = QHBoxLayout(telemetry_frame)
        t_lay.setContentsMargins(14, 4, 14, 4)
        t_lay.setSpacing(16)

        hw = self.hw_profile
        t_lay.addWidget(self._create_telemetry_item("CPU", hw.cpu_arch, "#16A34A"))
        t_lay.addWidget(self._create_v_divider())
        t_lay.addWidget(self._create_telemetry_item("GPU", "Adreno" if hw.has_adreno_gpu else "Host GPU", "#16A34A"))
        t_lay.addWidget(self._create_v_divider())
        npu_color = "#16A34A" if hw.npu_status == "Available" else "#D97706"
        t_lay.addWidget(self._create_telemetry_item("NPU", hw.npu_status, npu_color))
        t_lay.addWidget(self._create_v_divider())
        t_lay.addWidget(self._create_telemetry_item("RAM", f"{hw.available_ram_gb} GB Free", "#0284C7"))
        
        HoverPopFilter.install(telemetry_frame, pop_px=3)
        layout.addWidget(telemetry_frame)
        layout.addStretch()

        # 3. Right Group: Offline Engine Pill & Theme Switcher (Generous spacing, zero overlaps!)
        right_box = QHBoxLayout()
        right_box.setSpacing(14)

        lbl_conn = QLabel("● OFFLINE ENGINE")
        lbl_conn.setStyleSheet("""
            background-color: rgba(22, 163, 74, 0.12);
            border: 1px solid rgba(22, 163, 74, 0.35);
            border-radius: 16px;
            padding: 7px 14px;
            font-size: 11.5px;
            font-weight: 700;
            color: #16A34A;
        """)
        HoverPopFilter.install(lbl_conn, pop_px=3)
        right_box.addWidget(lbl_conn)

        self.btn_theme_toggle = ThemeTogglePill()
        self.btn_theme_toggle.setToolTip("Toggle between Light Mode and Dark Theme")
        self.btn_theme_toggle.clicked.connect(self.toggle_theme)
        HoverPopFilter.install(self.btn_theme_toggle, pop_px=3)
        right_box.addWidget(self.btn_theme_toggle)

        layout.addLayout(right_box)
        return header

    def _create_telemetry_item(self, label: str, val: str, color: str) -> QWidget:
        item = QWidget()
        lay = QHBoxLayout(item)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)
        lbl_k = QLabel(f"{label}:")
        lbl_k.setObjectName("MetricLabel")
        lbl_k.setStyleSheet("background: transparent; border: none; font-size: 12px; font-weight: 600; color: #64748B;")
        lbl_v = QLabel(val)
        lbl_v.setStyleSheet(f"background: transparent; border: none; color: {color}; font-size: 12px; font-weight: bold;")
        lay.addWidget(lbl_k)
        lay.addWidget(lbl_v)
        return item

    def _create_v_divider(self) -> QFrame:
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setFixedWidth(1)
        div.setStyleSheet("background-color: rgba(128,128,128,0.25); border: none; margin: 4px 0px;")
        return div

    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("SidebarFrame")
        sidebar.setFixedWidth(230)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 16, 10, 16)
        layout.setSpacing(4)

        self.nav_buttons = []
        nav_items = [
            ("Dashboard", "dashboard"),
            ("Model Import", "import"),
            ("Model Analysis", "analyze"),
            ("Optimization", "optimize"),
            ("Deploy & Target", "deploy"),
            ("Benchmarking", "benchmark"),
            ("Hardware Profiler", "hardware"),
            ("Model Library", "library"),
            ("Reports", "reports"),
            ("Settings", "settings"),
        ]

        for idx, (label, tag) in enumerate(nav_items):
            btn = NavPillButton(label, tag)
            btn.clicked.connect(lambda checked=False, i=idx, t=tag: self._switch_page(i, t))
            HoverPopFilter.install(btn, pop_px=2)
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch()

        # Version stamp (macOS / Windows style)
        lbl_ver = QLabel("nibble v1.4.0\nQualcomm AI Ready • HP PC")
        lbl_ver.setStyleSheet("color: #64748B; font-size: 11px; padding-left: 14px; line-height: 16px;")
        layout.addWidget(lbl_ver)

        return sidebar

    def _init_pages(self):
        # 0: Dashboard
        self.page_dashboard = DashboardView()
        self.page_dashboard.navigate_to.connect(self._navigate_by_name)
        self.stacked_widget.addWidget(self.page_dashboard)

        # 1: Model Import
        self.page_import = ModelImportView()
        self.page_import.model_imported.connect(self._on_model_imported)
        self.page_import.navigate_to.connect(self._navigate_by_name)
        self.stacked_widget.addWidget(self.page_import)

        # 2: Model Analysis
        self.page_analysis = ModelAnalysisView()
        self.page_analysis.navigate_to.connect(self._navigate_by_name)
        self.stacked_widget.addWidget(self.page_analysis)

        # 3: Optimization
        self.page_optimization = OptimizationView()
        self.page_optimization.navigate_to.connect(self._navigate_by_name)
        self.page_optimization.optimization_completed.connect(self._on_optimization_completed)
        self.stacked_widget.addWidget(self.page_optimization)

        # 4: Deploy & Target
        self.page_deploy = DeployView()
        self.page_deploy.navigate_to.connect(self._navigate_by_name)
        self.stacked_widget.addWidget(self.page_deploy)

        # 5: Benchmark
        self.page_benchmark = BenchmarkView()
        self.page_benchmark.navigate_to.connect(self._navigate_by_name)
        self.page_benchmark.benchmark_completed.connect(self._on_benchmark_completed)
        self.stacked_widget.addWidget(self.page_benchmark)

        # 6: Hardware Profiler
        self.page_hardware = HardwareView()
        self.stacked_widget.addWidget(self.page_hardware)

        # 7: Model Library
        self.page_library = ModelLibraryView()
        self.page_library.model_selected_for_project.connect(self._on_model_loaded_from_library)
        self.page_library.navigate_to.connect(self._navigate_by_name)
        self.stacked_widget.addWidget(self.page_library)

        # 8: Reports
        self.page_reports = ReportsView()
        self.stacked_widget.addWidget(self.page_reports)

        # 9: Settings (Windows & macOS Fusion)
        self.page_settings = SettingsView(is_dark=self.is_dark_theme)
        self.page_settings.theme_changed.connect(self.apply_theme)
        self.page_settings.ambient_anim_toggled.connect(self.ambient_canvas.set_animation_enabled)
        self.page_settings.transitions_toggled.connect(lambda en: setattr(self.stacked_widget, "transitions_enabled", en))
        self.stacked_widget.addWidget(self.page_settings)

    def _switch_page(self, index: int, tag: str):
        self.stacked_widget.slide_to_index(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def _navigate_by_name(self, name: str):
        mapping = {
            "dashboard": 0,
            "import": 1,
            "analyze": 2,
            "optimize": 3,
            "deploy": 4,
            "benchmark": 5,
            "hardware": 6,
            "library": 7,
            "reports": 8,
            "settings": 9
        }
        idx = mapping.get(name.lower(), 0)
        self._switch_page(idx, name)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "ambient_canvas") and self.ambient_canvas and self.centralWidget():
            self.ambient_canvas.resize(self.centralWidget().size())

    def toggle_theme(self):
        """Toggles between Dark Theme and Light Mode."""
        self.apply_theme(not self.is_dark_theme)

    def apply_theme(self, is_dark: bool):
        """Applies the selected theme across the entire application."""
        self.is_dark_theme = is_dark
        self.setStyleSheet(get_stylesheet(is_dark))
        if hasattr(self, "ambient_canvas"):
            self.ambient_canvas.set_dark_mode(is_dark)
        if hasattr(self, "btn_theme_toggle"):
            self.btn_theme_toggle.update()
        if hasattr(self, "page_settings"):
            self.page_settings.set_theme_mode(is_dark)

    def _on_model_imported(self, metadata: ModelMetadata, project_id: int):
        self.current_metadata = metadata
        self.current_project_id = project_id
        self.lbl_status_model.setText(f"Active Model: {metadata.name} ({metadata.file_size_mb} MB, {metadata.total_params:,} params)")

        # Propagate to child views
        self.page_analysis.set_model(metadata)
        self.page_optimization.set_model(metadata, project_id)
        self.page_deploy.set_model(metadata)
        self.page_benchmark.set_models(metadata.file_path, metadata.file_path)

    def _on_model_loaded_from_library(self, metadata: ModelMetadata):
        self._on_model_imported(metadata, 1)

    def _on_optimization_completed(self, opt_path: str, res: dict):
        if self.current_metadata:
            self.page_benchmark.set_models(self.current_metadata.file_path, opt_path)
        self.page_dashboard.refresh_data()

    def _on_benchmark_completed(self, comp: dict):
        self.page_dashboard.refresh_data()
        self.page_reports.refresh_reports()

    def _load_initial_model_if_present(self):
        """Auto-load default sample model on first startup for immediate interactive demo."""
        sample_path = Path("E:/snapdragon/models/resnet_classifier.onnx")
        if sample_path.exists():
            try:
                meta = ModelInspector.inspect(sample_path)
                self._on_model_imported(meta, 1)
            except Exception:
                pass
