"""
nibble UI Theme Engine — Windows Fluent & macOS Fusion Design System.
Provides full dual-theme support (Dark Theme & Light Mode), enlarged readable typography,
extra rounded 20px-24px geometry (cards, pills, capsules), and generous vertical headroom.
Defaults to Light Mode.
"""

PALETTE_LIGHT = {
    "bg_main": "#F8FAFC",
    "bg_card": "#FFFFFF",
    "bg_card_hover": "#F1F5F9",
    "bg_sidebar": "#F1F5F9",
    "bg_header": "#FFFFFF",
    "bg_hover": "#E2E8F0",
    "bg_active": "#E0F2FE",
    "border": "#E2E8F0",
    "border_subtle": "#EDF2F7",
    "border_highlight": "rgba(0, 0, 0, 0.06)",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#94A3B8",
    "accent_blue": "#0284C7",
    "accent_blue_hover": "#0369A1",
    "accent_primary": "#2563EB",
    "success_green": "#16A34A",
    "warning_yellow": "#D97706",
    "danger_red": "#DC2626",
    "npu_purple": "#9333EA",
}

PALETTE_DARK = {
    "bg_main": "#0B0F17",
    "bg_card": "#151B26",
    "bg_card_hover": "#1C2433",
    "bg_sidebar": "#0E131D",
    "bg_header": "#131822",
    "bg_hover": "#1F2937",
    "bg_active": "#223047",
    "border": "#222D3D",
    "border_subtle": "#1B2330",
    "border_highlight": "rgba(255, 255, 255, 0.08)",
    "text_primary": "#F0F6FC",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "accent_blue": "#38BDF8",
    "accent_blue_hover": "#0EA5E9",
    "accent_primary": "#2563EB",
    "success_green": "#22C55E",
    "warning_yellow": "#F59E0B",
    "danger_red": "#EF4444",
    "npu_purple": "#A855F7",
}

# Aliases
PALETTE = PALETTE_LIGHT

FONT_STACK = "'Segoe UI Variable Text', 'Segoe UI', -apple-system, BlinkMacSystemFont, 'SF Pro Text', Roboto, sans-serif"


LIGHT_STYLESHEET = f"""
QMainWindow, QWidget#CentralWidget {{
    background-color: #F8FAFC;
    color: #0F172A;
    font-family: {FONT_STACK};
    font-size: 14px;
}}

QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: transparent;
    border: none;
}}

/* Refined, smooth pill scrollbar */
QScrollBar:vertical {{
    border: none;
    background: rgba(0, 0, 0, 0.03);
    width: 9px;
    margin: 6px 2px 6px 2px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: #CBD5E1;
    min-height: 32px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: #0284C7;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* Sidebar Navigation (Clean macOS / Windows 11 Light) */
QFrame#SidebarFrame {{
    background-color: #F1F5F9;
    border-right: 1px solid #E2E8F0;
}}

QPushButton#NavButton {{
    background-color: transparent;
    border: none;
    margin: 2px 6px;
    font-family: {FONT_STACK};
}}

/* Spacious Top Header Bar (Zero vertical crunch!) */
QFrame#HeaderBar {{
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    min-height: 84px;
    padding: 14px 24px;
}}

/* Super-Rounded Clean Cards (20px radius) */
QFrame.CardFrame, QFrame#CardFrame {{
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 22px;
}}

QFrame#MetricCard, QFrame.MetricCard {{
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 18px;
}}

QFrame#MetricCard:hover, QFrame.MetricCard:hover {{
    border-color: #0284C7;
    background-color: #F8FAFC;
}}

/* Typography */
QLabel#HeaderTitle {{
    font-size: 24px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.4px;
}}

QLabel#HeaderSubtitle {{
    font-size: 13px;
    color: #64748B;
}}

QLabel#CardTitle {{
    font-size: 16px;
    font-weight: 600;
    color: #0F172A;
    letter-spacing: -0.2px;
}}

QLabel#CardSubtitle {{
    font-size: 13px;
    color: #64748B;
}}

QLabel#MetricValue {{
    font-size: 26px;
    font-weight: 700;
    color: #0284C7;
}}

QLabel#MetricLabel {{
    font-size: 12px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Rounded Pill Buttons (24px radius) */
QPushButton {{
    background-color: #F1F5F9;
    border: 1px solid #CBD5E1;
    color: #0F172A;
    padding: 10px 24px;
    border-radius: 24px;
    font-weight: 500;
    font-size: 13px;
}}

QPushButton:hover {{
    background-color: #E2E8F0;
    border-color: #94A3B8;
}}

QPushButton:pressed {{
    background-color: #CBD5E1;
}}

QPushButton#PrimaryButton {{
    background-color: #0284C7;
    border: 1px solid #0369A1;
    color: #FFFFFF;
    font-weight: 600;
    border-radius: 24px;
}}

QPushButton#PrimaryButton:hover {{
    background-color: #0369A1;
}}

QPushButton#SuccessButton {{
    background-color: #16A34A;
    border: 1px solid #15803D;
    color: #FFFFFF;
    font-weight: 600;
    border-radius: 24px;
}}

QPushButton#SuccessButton:hover {{
    background-color: #15803D;
}}

QPushButton#ThemeToggleBtn {{
    background-color: #E2E8F0;
    border: 1px solid #CBD5E1;
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
    color: #0F172A;
}}

QPushButton#ThemeToggleBtn:hover {{
    background-color: #CBD5E1;
    border-color: #0284C7;
}}

/* Form Controls with 12px rounded corners */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {{
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 9px 14px;
    color: #0F172A;
    font-size: 13px;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
    border: 1px solid #0284C7;
    background-color: #F8FAFC;
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

/* Rounded Tables (18px radius) */
QTableWidget, QTableView {{
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    gridline-color: transparent;
    color: #0F172A;
    font-size: 13.5px;
    selection-background-color: #E0F2FE;
    selection-color: #0284C7;
    outline: none;
}}

QTableWidget::item {{
    padding: 10px 16px;
    border-bottom: 1px solid #F1F5F9;
    color: #1E293B;
}}

QTableWidget::item:selected {{
    background-color: #E0F2FE;
    color: #0284C7;
}}

QTableWidget QPushButton, QTableView QPushButton {{
    background-color: #F1F5F9;
    border: 1px solid #CBD5E1;
    color: #0F172A;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    min-height: 24px;
    max-height: 28px;
}}

QTableWidget QPushButton:hover, QTableView QPushButton:hover {{
    background-color: #0284C7;
    border-color: #0369A1;
    color: #FFFFFF;
}}

QHeaderView::section {{
    background-color: #F8FAFC;
    color: #64748B;
    border: none;
    border-bottom: 1px solid #E2E8F0;
    padding: 12px 16px;
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Rounded Progress Bar */
QProgressBar {{
    background-color: #E2E8F0;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    height: 12px;
    text-align: center;
    color: #0F172A;
    font-size: 11px;
}}

QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284C7, stop:1 #38BDF8);
    border-radius: 7px;
}}
"""


DARK_STYLESHEET = f"""
QMainWindow, QWidget#CentralWidget {{
    background-color: #0B0F17;
    color: #F0F6FC;
    font-family: {FONT_STACK};
    font-size: 14px;
}}

QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: transparent;
    border: none;
}}

QScrollBar:vertical {{
    border: none;
    background: rgba(255, 255, 255, 0.03);
    width: 9px;
    margin: 6px 2px 6px 2px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: #242E3E;
    min-height: 32px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: #38BDF8;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* Sidebar Navigation */
QFrame#SidebarFrame {{
    background-color: #0E131D;
    border-right: 1px solid #1C2433;
}}

QPushButton#NavButton {{
    background-color: transparent;
    border: none;
    margin: 2px 6px;
    font-family: {FONT_STACK};
}}

/* Spacious Top Header Bar */
QFrame#HeaderBar {{
    background-color: #131822;
    border-bottom: 1px solid #1E2633;
    min-height: 84px;
    padding: 14px 24px;
}}

/* Super-Rounded Glass & Acrylic Cards (20px radius) */
QFrame.CardFrame, QFrame#CardFrame {{
    background-color: #151B26;
    border: 1px solid #222D3D;
    border-radius: 20px;
    padding: 22px;
}}

QFrame#MetricCard, QFrame.MetricCard {{
    background-color: #151B26;
    border: 1px solid #222D3D;
    border-radius: 18px;
    padding: 18px;
}}

QFrame#MetricCard:hover, QFrame.MetricCard:hover {{
    border-color: #38BDF8;
    background-color: #18202D;
}}

/* Typography */
QLabel#HeaderTitle {{
    font-size: 24px;
    font-weight: 700;
    color: #F0F6FC;
    letter-spacing: -0.4px;
}}

QLabel#HeaderSubtitle {{
    font-size: 13px;
    color: #94A3B8;
}}

QLabel#CardTitle {{
    font-size: 16px;
    font-weight: 600;
    color: #F0F6FC;
    letter-spacing: -0.2px;
}}

QLabel#CardSubtitle {{
    font-size: 13px;
    color: #94A3B8;
}}

QLabel#MetricValue {{
    font-size: 26px;
    font-weight: 700;
    color: #38BDF8;
}}

QLabel#MetricLabel {{
    font-size: 12px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Rounded Pill Buttons (24px radius) */
QPushButton {{
    background-color: #1C2433;
    border: 1px solid #2A3649;
    color: #F0F6FC;
    padding: 10px 24px;
    border-radius: 24px;
    font-weight: 500;
    font-size: 13px;
}}

QPushButton:hover {{
    background-color: #242E40;
    border-color: #3B4D66;
    color: #FFFFFF;
}}

QPushButton:pressed {{
    background-color: #151B26;
}}

QPushButton#PrimaryButton {{
    background-color: #2563EB;
    border: 1px solid #3B82F6;
    color: #FFFFFF;
    font-weight: 600;
    border-radius: 24px;
}}

QPushButton#PrimaryButton:hover {{
    background-color: #1D4ED8;
    border-color: #60A5FA;
}}

QPushButton#SuccessButton {{
    background-color: #16A34A;
    border: 1px solid #22C55E;
    color: #FFFFFF;
    font-weight: 600;
    border-radius: 24px;
}}

QPushButton#SuccessButton:hover {{
    background-color: #15803D;
    border-color: #4ADE80;
}}

QPushButton#ThemeToggleBtn {{
    background-color: #1A2230;
    border: 1px solid #2B374A;
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
    color: #F0F6FC;
}}

QPushButton#ThemeToggleBtn:hover {{
    background-color: #253147;
    border-color: #38BDF8;
}}

/* Form Controls with 12px rounded corners */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {{
    background-color: #0E131D;
    border: 1px solid #222D3D;
    border-radius: 12px;
    padding: 9px 14px;
    color: #F0F6FC;
    font-size: 13px;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
    border: 1px solid #38BDF8;
    background-color: #121824;
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

/* Rounded Tables (18px radius) */
QTableWidget, QTableView {{
    background-color: #121722;
    border: 1px solid #222D3D;
    border-radius: 18px;
    gridline-color: transparent;
    color: #F0F6FC;
    font-size: 13.5px;
    selection-background-color: rgba(3, 105, 161, 0.35);
    selection-color: #38BDF8;
    outline: none;
}}

QTableWidget::item {{
    padding: 10px 16px;
    border-bottom: 1px solid #1E2636;
    color: #E2E8F0;
}}

QTableWidget::item:selected {{
    background-color: rgba(3, 105, 161, 0.35);
    color: #38BDF8;
}}

QTableWidget QPushButton, QTableView QPushButton {{
    background-color: #1E293B;
    border: 1px solid #334155;
    color: #F0F6FC;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    min-height: 24px;
    max-height: 28px;
}}

QTableWidget QPushButton:hover, QTableView QPushButton:hover {{
    background-color: #0284C7;
    border-color: #38BDF8;
    color: #FFFFFF;
}}

QHeaderView::section {{
    background-color: #151B26;
    color: #94A3B8;
    border: none;
    border-bottom: 1px solid #222D3D;
    padding: 12px 16px;
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Rounded Progress Bar */
QProgressBar {{
    background-color: #1A2230;
    border: 1px solid #283446;
    border-radius: 8px;
    height: 12px;
    text-align: center;
    color: #F0F6FC;
    font-size: 11px;
}}

QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #16A34A, stop:1 #22C55E);
    border-radius: 7px;
}}
"""


def get_stylesheet(is_dark: bool = False) -> str:
    """Returns the comprehensive stylesheet, defaulting to Light Mode."""
    return DARK_STYLESHEET if is_dark else LIGHT_STYLESHEET


def get_palette(is_dark: bool = False) -> dict:
    """Returns color dictionary tokens, defaulting to Light Mode."""
    return PALETTE_DARK if is_dark else PALETTE_LIGHT
