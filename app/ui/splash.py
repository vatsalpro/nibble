"""
nibble Boot Screen & Splash Animation.
Features:
- Mathematical baseline alignment for letter 'n' and cursive 'ibble' wordmark.
- Buttery smooth, slow, deliberate pacing (no rush, calm and luxurious).
- Slow, hypnotic circular buffering ring (replaces progress bar).
- Smooth cross-fade into MainWindow.
"""

import math
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsOpacityEffect
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal, QRectF, QPointF,
    QVariantAnimation
)
from PySide6.QtGui import (
    QPainter, QColor, QFont, QFontMetrics, QRadialGradient, QPaintEvent,
    QPen, QBrush
)


class NibbleWordmarkCanvas(QWidget):
    """
    Renders 'n' and cursive 'ibble' with 100% mathematically exact baseline alignment
    and animated cursive fade-in.
    """
    def __init__(self, parent=None, is_dark: bool = False):
        super().__init__(parent)
        self.is_dark = is_dark
        self.setMinimumHeight(120)
        self.n_opacity = 0.0
        self.ibble_opacity = 0.0
        self.buffer_angle = 0.0

        # Fonts: modern geometric 'n' + flowing script 'ibble'
        self.font_n = QFont("Segoe UI Variable Display", 62, QFont.Bold)
        self.font_n.setStyleStrategy(QFont.PreferAntialias)

        # Cursive script font with fallbacks
        self.font_ibble = QFont("Segoe Script", 52, QFont.Normal)
        self.font_ibble.setStyleStrategy(QFont.PreferAntialias)

        # Timer for slow, smooth buffering ring (~30 FPS, slow gentle spin)
        self.ring_timer = QTimer(self)
        self.ring_timer.setInterval(33)
        self.ring_timer.timeout.connect(self._on_ring_tick)
        self.ring_timer.start()

    def _on_ring_tick(self):
        # 1 full rotation every ~2.5 seconds (slow, hypnotic, buttery)
        self.buffer_angle = (self.buffer_angle + 2.4) % 360.0
        self.update()

    def set_n_opacity(self, val: float):
        self.n_opacity = val
        self.update()

    def set_ibble_opacity(self, val: float):
        self.ibble_opacity = val
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        w = self.width()
        h = self.height()

        fm_n = QFontMetrics(self.font_n)
        fm_ibble = QFontMetrics(self.font_ibble)

        width_n = fm_n.horizontalAdvance("n")
        width_ibble = fm_ibble.horizontalAdvance("ibble")
        total_width = width_n + width_ibble

        # Center horizontally
        start_x = (w - total_width) / 2.0
        # Common mathematical baseline for both fonts
        baseline_y = 75.0

        # Colors
        color_n = QColor(2, 132, 199, int(255 * self.n_opacity))      # Vibrant Qualcomm Blue
        color_ibble = QColor(56, 189, 248, int(255 * self.ibble_opacity)) # Soft Cyan Script

        # 1. Draw 'n' at baseline
        if self.n_opacity > 0.01:
            painter.setFont(self.font_n)
            painter.setPen(color_n)
            painter.drawText(QPointF(start_x, baseline_y), "n")

        # 2. Draw 'ibble' at the EXACT same baseline right next to 'n'
        if self.ibble_opacity > 0.01:
            painter.setFont(self.font_ibble)
            painter.setPen(color_ibble)
            painter.drawText(QPointF(start_x + width_n, baseline_y), "ibble")

        # 3. Slow, elegant buffering ring below text
        ring_cx = w / 2.0
        ring_cy = 108.0
        ring_radius = 12.0
        ring_rect = QRectF(ring_cx - ring_radius, ring_cy - ring_radius, ring_radius * 2, ring_radius * 2)

        # Subtle background track
        pen_track = QPen(QColor(2, 132, 199, 35), 2.5)
        pen_track.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_track)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(ring_rect)

        # Rotating glowing arc
        pen_arc = QPen(QColor(2, 132, 199, 220), 2.5)
        pen_arc.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_arc)
        # Draw 80-degree sweep arc
        start_angle_16ths = int(self.buffer_angle * 16)
        span_angle_16ths = int(85 * 16)
        painter.drawArc(ring_rect, start_angle_16ths, span_angle_16ths)


class NibbleSplashScreen(QWidget):
    finished = Signal()

    def __init__(self, is_dark: bool = False):
        super().__init__()
        self.is_dark = is_dark
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(600, 360)

        # Center on primary display
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.move((geo.width() - self.width()) // 2, (geo.height() - self.height()) // 2)

        self._fade_anim = None
        self.init_ui()
        self._start_buttery_sequence()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 48, 40, 36)
        layout.setSpacing(10)

        layout.addStretch()

        # Canvas with baseline-aligned 'n' and cursive 'ibble' + buffering ring
        self.wordmark_canvas = NibbleWordmarkCanvas(self, is_dark=self.is_dark)
        layout.addWidget(self.wordmark_canvas)

        # Subtitle
        self.lbl_sub = QLabel("Snapdragon AI Optimization Studio")
        self.lbl_sub.setAlignment(Qt.AlignCenter)
        self.lbl_sub.setStyleSheet("color: #64748B; font-size: 14px; font-weight: 500; letter-spacing: 0.5px;")
        layout.addWidget(self.lbl_sub)

        # Quiet live telemetry status
        self.lbl_status = QLabel("Calibrating Snapdragon AI runtime...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 500;")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

    def _start_buttery_sequence(self):
        # 1. Entrance fade-in: slow and smooth (550ms)
        self.effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.effect)
        self.effect.setOpacity(0.0)

        fade_in = QPropertyAnimation(self.effect, b"opacity", self)
        fade_in.setDuration(550)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.OutCubic)
        fade_in.start()

        # 2. Letter 'n' smoothly fades in (0 to 600ms)
        self.anim_n = QVariantAnimation(self)
        self.anim_n.setDuration(600)
        self.anim_n.setStartValue(0.0)
        self.anim_n.setEndValue(1.0)
        self.anim_n.setEasingCurve(QEasingCurve.OutCubic)
        self.anim_n.valueChanged.connect(self.wordmark_canvas.set_n_opacity)
        self.anim_n.start()

        # 3. Cursive 'ibble' starts fading in gently at 650ms, taking 850ms (buttery cursive flow)
        QTimer.singleShot(650, self._start_ibble_anim)

        # 4. Telemetry status updates (calm, spaced out)
        QTimer.singleShot(1100, lambda: self.lbl_status.setText("Binding Qualcomm Hexagon NPU & DirectML providers..."))
        QTimer.singleShot(2100, lambda: self.lbl_status.setText("Optimizing graph pipelines for Snapdragon X Elite..."))
        QTimer.singleShot(3000, lambda: self.lbl_status.setText("Ready"))

        # 5. Slow, graceful fade-out into MainWindow at 3500ms
        QTimer.singleShot(3500, self._fade_out_and_finish)

    def _start_ibble_anim(self):
        self.anim_ibble = QVariantAnimation(self)
        self.anim_ibble.setDuration(850)
        self.anim_ibble.setStartValue(0.0)
        self.anim_ibble.setEndValue(1.0)
        self.anim_ibble.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim_ibble.valueChanged.connect(self.wordmark_canvas.set_ibble_opacity)
        self.anim_ibble.start()

    def _fade_out_and_finish(self):
        if self._fade_anim:
            return

        # Luxurious 480ms cross-fade exit
        anim = QPropertyAnimation(self.effect, b"opacity", self)
        anim.setDuration(480)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InOutCubic)

        def on_done():
            self.wordmark_canvas.ring_timer.stop()
            self.close()
            self.finished.emit()

        anim.finished.connect(on_done)
        self._fade_anim = anim
        anim.start()

    def mousePressEvent(self, event):
        self._fade_out_and_finish()

    def keyPressEvent(self, event):
        self._fade_out_and_finish()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(0, 0, self.width(), self.height())
        
        # Adaptive glass card
        if self.is_dark:
            bg_color = QColor(11, 15, 23, 250)
            border_color = QColor(255, 255, 255, 22)
            glow_c1 = QColor(2, 132, 199, 24)
        else:
            bg_color = QColor(255, 255, 255, 250)
            border_color = QColor(0, 0, 0, 18)
            glow_c1 = QColor(2, 132, 199, 16)

        painter.setPen(border_color)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(rect, 24, 24)

        # Gentle radial background glow
        grad = QRadialGradient(self.width() / 2, self.height() / 2, self.width() * 0.45)
        grad.setColorAt(0.0, glow_c1)
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), grad)
