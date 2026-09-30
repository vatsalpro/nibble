"""
High-DPI Vector Icon Rendering for nibble.
Renders crisp, antialiased geometric vector icons directly using QPainterPath.
Eliminates all Unicode glyph/font mismatches, emoji rendering bugs, and tofumarks.
"""

from PySide6.QtGui import QPainter, QPainterPath, QColor, QPen, QBrush
from PySide6.QtCore import Qt, QRectF, QPointF


class VectorIconRenderer:
    @staticmethod
    def draw_icon(painter: QPainter, tag: str, rect: QRectF, color: QColor):
        """Draws a crisp, DPI-independent vector icon centered inside rect."""
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()
        cx = x + w / 2.0
        cy = y + h / 2.0

        pen = QPen(color, 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        tag = tag.lower()

        if tag == "dashboard":
            # 2x2 rounded grid tiles
            tile_s = min(w, h) * 0.38
            gap = min(w, h) * 0.12
            r = 2.5
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(cx - tile_s - gap/2, cy - tile_s - gap/2, tile_s, tile_s), r, r)
            painter.drawRoundedRect(QRectF(cx + gap/2, cy - tile_s - gap/2, tile_s, tile_s), r, r)
            painter.drawRoundedRect(QRectF(cx - tile_s - gap/2, cy + gap/2, tile_s, tile_s), r, r)
            painter.drawRoundedRect(QRectF(cx + gap/2, cy + gap/2, tile_s, tile_s), r, r)

        elif tag == "import":
            # Upward arrow from base tray
            s = min(w, h) * 0.45
            path = QPainterPath()
            # Arrow
            path.moveTo(cx, cy + s * 0.6)
            path.lineTo(cx, cy - s * 0.7)
            path.moveTo(cx - s * 0.45, cy - s * 0.25)
            path.lineTo(cx, cy - s * 0.7)
            path.lineTo(cx + s * 0.45, cy - s * 0.25)
            # Base tray
            path.moveTo(cx - s * 0.8, cy + s * 0.2)
            path.lineTo(cx - s * 0.8, cy + s * 0.8)
            path.lineTo(cx + s * 0.8, cy + s * 0.8)
            path.lineTo(cx + s * 0.8, cy + s * 0.2)
            painter.drawPath(path)

        elif tag in ("analyze", "analysis"):
            # Neural graph with 3 interconnected nodes
            r_node = min(w, h) * 0.14
            p1 = QPointF(cx - w * 0.28, cy - h * 0.22)
            p2 = QPointF(cx - w * 0.28, cy + h * 0.22)
            p3 = QPointF(cx + w * 0.28, cy)

            # Connecting lines
            pen_line = QPen(color, 1.4)
            painter.setPen(pen_line)
            painter.drawLine(p1, p3)
            painter.drawLine(p2, p3)
            painter.drawLine(p1, p2)

            # Filled node circles
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(p1, r_node, r_node)
            painter.drawEllipse(p2, r_node, r_node)
            painter.drawEllipse(p3, r_node * 1.2, r_node * 1.2)

        elif tag in ("optimize", "optimization"):
            # Lightning bolt
            s = min(w, h) * 0.45
            path = QPainterPath()
            path.moveTo(cx + s * 0.2, cy - s)
            path.lineTo(cx - s * 0.5, cy + s * 0.05)
            path.lineTo(cx + s * 0.05, cy + s * 0.05)
            path.lineTo(cx - s * 0.2, cy + s)
            path.lineTo(cx + s * 0.6, cy - s * 0.05)
            path.lineTo(cx + s * 0.05, cy - s * 0.05)
            path.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawPath(path)

        elif tag in ("deploy", "target"):
            # Target reticle with center dot
            r_outer = min(w, h) * 0.40
            painter.drawEllipse(QPointF(cx, cy), r_outer, r_outer)
            # Center target dot
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cx, cy), 3.0, 3.0)
            # Crosshair ticks
            painter.setPen(pen)
            painter.drawLine(QPointF(cx, cy - r_outer - 2), QPointF(cx, cy - r_outer + 3))
            painter.drawLine(QPointF(cx, cy + r_outer - 3), QPointF(cx, cy + r_outer + 2))
            painter.drawLine(QPointF(cx - r_outer - 2, cy), QPointF(cx - r_outer + 3, cy))
            painter.drawLine(QPointF(cx + r_outer - 3, cy), QPointF(cx + r_outer + 2, cy))

        elif tag in ("benchmark", "benchmarking"):
            # Stopwatch gauge with needle
            r_gauge = min(w, h) * 0.38
            painter.drawEllipse(QPointF(cx, cy + 1), r_gauge, r_gauge)
            # Top button
            painter.drawLine(QPointF(cx - 3, cy - r_gauge - 2), QPointF(cx + 3, cy - r_gauge - 2))
            painter.drawLine(QPointF(cx, cy - r_gauge + 1), QPointF(cx, cy - r_gauge - 2))
            # Needle pointing to 2 o'clock
            pen_needle = QPen(color, 2.0)
            painter.setPen(pen_needle)
            painter.drawLine(QPointF(cx, cy + 1), QPointF(cx + r_gauge * 0.55, cy - r_gauge * 0.35))

        elif tag in ("hardware", "cpu"):
            # CPU chip with pins
            chip_s = min(w, h) * 0.44
            r_chip = QRectF(cx - chip_s/2, cy - chip_s/2, chip_s, chip_s)
            painter.drawRoundedRect(r_chip, 3, 3)
            # Inner die
            die_s = chip_s * 0.45
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(cx - die_s/2, cy - die_s/2, die_s, die_s), 1.5, 1.5)
            # Pins extending
            painter.setPen(QPen(color, 1.6))
            pin_len = 3.5
            # Top/Bottom pins
            for offset in (-chip_s * 0.25, chip_s * 0.25):
                painter.drawLine(QPointF(cx + offset, cy - chip_s/2), QPointF(cx + offset, cy - chip_s/2 - pin_len))
                painter.drawLine(QPointF(cx + offset, cy + chip_s/2), QPointF(cx + offset, cy + chip_s/2 + pin_len))
                painter.drawLine(QPointF(cx - chip_s/2, cy + offset), QPointF(cx - chip_s/2 - pin_len, cy + offset))
                painter.drawLine(QPointF(cx + chip_s/2, cy + offset), QPointF(cx + chip_s/2 + pin_len, cy + offset))

        elif tag in ("library", "models"):
            # Stacked model layers / library books
            w_b = min(w, h) * 0.65
            h_b = min(w, h) * 0.18
            r = 2.0
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(cx - w_b * 0.5, cy - h_b * 1.6, w_b, h_b), r, r)
            painter.drawRoundedRect(QRectF(cx - w_b * 0.4, cy - h_b * 0.3, w_b * 0.9, h_b), r, r)
            painter.drawRoundedRect(QRectF(cx - w_b * 0.5, cy + h_b * 1.0, w_b, h_b), r, r)

        elif tag in ("reports", "report"):
            # Document sheet with folded corner & content lines
            doc_w = min(w, h) * 0.56
            doc_h = min(w, h) * 0.72
            path = QPainterPath()
            path.moveTo(cx - doc_w/2, cy - doc_h/2)
            path.lineTo(cx + doc_w/2 - 4, cy - doc_h/2)
            path.lineTo(cx + doc_w/2, cy - doc_h/2 + 4)
            path.lineTo(cx + doc_w/2, cy + doc_h/2)
            path.lineTo(cx - doc_w/2, cy + doc_h/2)
            path.closeSubpath()
            painter.drawPath(path)
            # Internal content lines
            painter.drawLine(QPointF(cx - doc_w/2 + 3.5, cy - doc_h/6), QPointF(cx + doc_w/2 - 3.5, cy - doc_h/6))
            painter.drawLine(QPointF(cx - doc_w/2 + 3.5, cy + doc_h/8), QPointF(cx + doc_w/2 - 3.5, cy + doc_h/8))

        elif tag in ("settings", "preferences"):
            # Modern configuration sliders (macOS settings style)
            track_w = min(w, h) * 0.70
            track_y1 = cy - h * 0.24
            track_y2 = cy
            track_y3 = cy + h * 0.24

            painter.setPen(QPen(color, 1.8))
            painter.drawLine(QPointF(cx - track_w/2, track_y1), QPointF(cx + track_w/2, track_y1))
            painter.drawLine(QPointF(cx - track_w/2, track_y2), QPointF(cx + track_w/2, track_y2))
            painter.drawLine(QPointF(cx - track_w/2, track_y3), QPointF(cx + track_w/2, track_y3))

            # Knobs
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cx - track_w * 0.15, track_y1), 2.8, 2.8)
            painter.drawEllipse(QPointF(cx + track_w * 0.22, track_y2), 2.8, 2.8)
            painter.drawEllipse(QPointF(cx - track_w * 0.26, track_y3), 2.8, 2.8)

        elif tag == "sun":
            # Sun core and radiant rays
            r_sun = min(w, h) * 0.22
            painter.drawEllipse(QPointF(cx, cy), r_sun, r_sun)
            painter.setPen(QPen(color, 1.8))
            ray_dist = r_sun + 3.5
            ray_len = 2.5
            painter.drawLine(QPointF(cx, cy - ray_dist), QPointF(cx, cy - ray_dist - ray_len))
            painter.drawLine(QPointF(cx, cy + ray_dist), QPointF(cx, cy + ray_dist + ray_len))
            painter.drawLine(QPointF(cx - ray_dist, cy), QPointF(cx - ray_dist - ray_len, cy))
            painter.drawLine(QPointF(cx + ray_dist, cy), QPointF(cx + ray_dist + ray_len, cy))

        elif tag == "moon":
            # Crescent moon
            path = QPainterPath()
            path.moveTo(cx + 2, cy - min(w, h) * 0.38)
            path.arcTo(QRectF(cx - min(w, h) * 0.40, cy - min(w, h) * 0.40, min(w, h) * 0.80, min(w, h) * 0.80), 90, 180)
            path.arcTo(QRectF(cx - min(w, h) * 0.26, cy - min(w, h) * 0.36, min(w, h) * 0.65, min(w, h) * 0.72), 270, -180)
            path.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawPath(path)

        painter.restore()
