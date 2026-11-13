"""3dmod look: default colours, toolbar icons (drawn here, not copied from IMOD) and small widgets."""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Callable, Optional

from qtpy.QtCore import QPointF, QRectF, QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStyleFactory,
    QToolBar,
    QToolButton,
    QWidget,
)

# 3dmod's default named colours (preferences.cpp).
BACKGROUND = QColor(64, 64, 96)
FOREGROUND = QColor(255, 255, 128)  # XY box and Z marker in the XYZ window
CURPOINT = QColor(255, 255, 128)  # current image point
BGNPOINT = QColor(0, 255, 0)  # X marker / YZ box in the XYZ window
ENDPOINT = QColor(255, 0, 0)  # Y marker / XZ box in the XYZ window
FRACTION_BOX = QColor(64, 192, 255)
CURSOR_ARM = 4  # 3dmod minimum current image point size
ICON_SIZE = 16
TOOLBUT_SIZE = 20  # 3dmod utilities.cpp
RED = QColor(230, 0, 0)


def apply_app_style(app: QApplication) -> None:
    """Use the Qt style 3dmod uses: "Windows" on Linux (Fusion on macOS). ``CMOD_STYLE`` overrides it."""
    key = os.environ.get("CMOD_STYLE") or ("Fusion" if sys.platform == "darwin" else "Windows")
    if key.lower() in (k.lower() for k in QStyleFactory.keys()):  # noqa: SIM118 - a list, not a dict
        app.setStyle(key)


def zoom_text(zoom: float) -> str:
    """3dmod's zoom box format: 4 decimals with a trailing "00" dropped."""
    text = f"{zoom:.4f}"
    return text[:-2] if text.endswith("00") else text


def draw_plus(p: QPainter, x: float, y: float, color: QColor = CURPOINT, size: int = CURSOR_ARM) -> None:
    p.setPen(QPen(color, 1))
    p.drawLine(QPointF(x - size, y), QPointF(x + size, y))
    p.drawLine(QPointF(x, y - size), QPointF(x, y + size))


_LIVE_WINDOWS: set = set()


def keep_alive(widget) -> None:
    """Hold a Python reference to a top-level window until Qt destroys it.

    Parentless windows are owned by Python. If their wrapper is only reachable through reference cycles (signal
    lambdas), the cycle collector can delete the C++ window at an arbitrary moment, even while it is painting, which
    crashes. With this, windows are only ever deleted by Qt (WA_DeleteOnClose), from the event loop.
    """
    _LIVE_WINDOWS.add(widget)
    widget.destroyed.connect(lambda *_args, w=widget: _LIVE_WINDOWS.discard(w))


# ---------------------------------------------------------------------------------------------- icons
def _pixmap(painter_fn: Callable[[QPainter], None], size: int = ICON_SIZE) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    painter_fn(p)
    p.end()
    return pix


def checker_icon(fine: bool) -> QPixmap:
    """Low-resolution (coarse) or high-resolution (fine) checkerboard, like 3dmod's HQ toggle."""
    step = 2 if fine else 4

    def paint(p: QPainter) -> None:
        for i in range(0, ICON_SIZE, step):
            for j in range(0, ICON_SIZE, step):
                if (i // step + j // step) % 2 == 0:
                    p.fillRect(i, j, step, step, Qt.black)

    return _pixmap(paint)


def lock_icon(locked: bool) -> QPixmap:
    color = QColor(230, 0, 0) if locked else Qt.black

    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(color, 2))
        shackle = QRectF(4, 1, 8, 9) if locked else QRectF(7, 1, 8, 9)
        p.drawArc(shackle, 0, 180 * 16)
        p.drawLine(QPointF(shackle.left(), 5.5), QPointF(shackle.left(), 8))
        if locked:
            p.drawLine(QPointF(shackle.right(), 5.5), QPointF(shackle.right(), 8))
        p.fillRect(QRectF(2, 8, 11, 8), color)
        p.fillRect(QRectF(7, 10, 1.5, 4), QColor(200, 200, 200))

    return _pixmap(paint)


def center_icon(keep: bool) -> QPixmap:
    """Keep-centred (dot in a box) or smart-centring (box in a box), as in 3dmod."""

    def paint(p: QPainter) -> None:
        p.setPen(QPen(Qt.black, 1))
        p.drawRect(0, 0, 15, 15)
        if keep:
            p.fillRect(7, 7, 2, 2, Qt.black)
        else:
            p.drawRect(3, 3, 9, 9)

    return _pixmap(paint)


def triangle_icon(up: bool, size: int = 13) -> QPixmap:
    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(40, 40, 40))
        if up:
            poly = QPolygonF([QPointF(1, 10), QPointF(11, 10), QPointF(6, 1)])
        else:
            poly = QPolygonF([QPointF(1, 2), QPointF(11, 2), QPointF(6, 11)])
        p.drawPolygon(poly)

    return _pixmap(paint, size)


def side_triangle_icon(right: bool, size: int = 13) -> QPixmap:
    """A triangle pointing left or right (previous / next tomogram)."""

    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(40, 40, 40))
        if right:
            poly = QPolygonF([QPointF(2, 1), QPointF(2, 11), QPointF(11, 6)])
        else:
            poly = QPolygonF([QPointF(10, 1), QPointF(10, 11), QPointF(1, 6)])
        p.drawPolygon(poly)

    return _pixmap(paint, size)


def _pen(color, width: float = 1.0) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.FlatCap)
    return pen


def insert_icon(after: bool) -> QPixmap:
    """3dmod's insert-direction toggle: a T (before) or an upside-down T (after) with a dot."""

    def paint(p: QPainter) -> None:
        if after:
            p.fillRect(7, 2, 2, 12, Qt.black)
            p.fillRect(2, 13, 12, 2, Qt.black)
            p.fillRect(6, 1, 4, 3, Qt.black)
        else:
            p.fillRect(7, 2, 2, 12, Qt.black)
            p.fillRect(2, 1, 12, 2, Qt.black)
            p.fillRect(6, 12, 4, 3, Qt.black)

    return _pixmap(paint)


def rubberband_icon(on: bool) -> QPixmap:
    def paint(p: QPainter) -> None:
        pen = _pen(RED if on else Qt.black, 2 if on else 1)
        pen.setDashPattern([2, 2])
        p.setPen(pen)
        p.drawRect(2, 4, 11, 8)

    return _pixmap(paint)


def lasso_icon(on: bool) -> QPixmap:
    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(_pen(RED if on else Qt.black, 1.2))
        p.drawEllipse(QRectF(4, 1.5, 10.5, 6.5))
        path = QPainterPath(QPointF(5, 6.5))
        path.cubicTo(QPointF(2.5, 9), QPointF(5.5, 10.5), QPointF(4, 14.5))
        p.drawPath(path)

    return _pixmap(paint)


def arrow_icon(on: bool) -> QPixmap:
    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(_pen(RED if on else Qt.black, 1.5))
        p.drawLine(QPointF(2, 14), QPointF(14, 2))
        p.drawLine(QPointF(14, 2), QPointF(8, 2))
        p.drawLine(QPointF(14, 2), QPointF(14, 8))

    return _pixmap(paint)


def zscale_icon(on: bool) -> QPixmap:
    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(_pen(RED if on else Qt.black, 1.5))
        top, bottom = (1.5, 14.5) if on else (4.5, 13)
        p.drawPolyline(QPolygonF([QPointF(3, top), QPointF(13, top), QPointF(3, bottom), QPointF(13, bottom)]))

    return _pixmap(paint)


def raise_icon() -> QPixmap:
    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.black)
        for y in (1.0, 7.5):
            p.drawPolygon(QPolygonF([QPointF(2, y + 6), QPointF(14, y + 6), QPointF(8, y)]))

    return _pixmap(paint)


def undo_icon(redo: bool) -> QPixmap:
    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        if redo:
            p.translate(16, 0)
            p.scale(-1, 1)
        p.setPen(_pen(Qt.black, 1.6))
        p.drawArc(QRectF(3, 3.5, 10, 9.5), 200 * 16, -250 * 16)
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.black)
        p.drawPolygon(QPolygonF([QPointF(1, 5.5), QPointF(6.5, 5.5), QPointF(3, 10)]))

    return _pixmap(paint)


def pin_icon(pegged: bool) -> QPixmap:
    """Keep-on-top pin: lying down when off, pushed in (red head) when on."""

    def paint(p: QPainter) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        if pegged:
            p.setPen(_pen(Qt.black, 1))
            p.drawLine(QPointF(8, 9), QPointF(8, 15))
            p.setBrush(RED)
            p.drawEllipse(QRectF(4, 1, 8, 8))
        else:
            p.setPen(_pen(Qt.black, 1))
            p.drawLine(QPointF(1, 8), QPointF(7, 8))
            p.setBrush(QColor(200, 200, 200))
            p.drawRect(QRectF(7, 4, 6, 8))
            p.drawLine(QPointF(13, 3), QPointF(13, 13))

    return _pixmap(paint)


# ---------------------------------------------------------------------------------------------- widgets
def tool_button(
    icon: Optional[QPixmap] = None,
    text: str = "",
    tip: str = "",
    on_click: Optional[Callable] = None,
) -> QToolButton:
    b = QToolButton()
    b.setAutoRaise(True)
    b.setFocusPolicy(Qt.NoFocus)
    if icon is not None:
        b.setIcon(QIcon(icon))
        b.setIconSize(icon.size())
    if text:
        b.setText(text)
    if tip:
        b.setToolTip(tip)
    if on_click is not None:
        b.clicked.connect(lambda _=False: on_click())
    return b


class ToggleButton(QToolButton):
    """A 3dmod toolbar toggle that shows a different icon when on."""

    def __init__(self, off: QPixmap, on: QPixmap, tip: str = "", checked: bool = False):
        super().__init__()
        self._icons = (QIcon(off), QIcon(on))
        self.setAutoRaise(True)
        self.setCheckable(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.setToolTip(tip)
        self.toggled.connect(self._show)
        self.setChecked(checked)
        self._show(checked)

    def _show(self, on: bool) -> None:
        self.setIcon(self._icons[1 if on else 0])


def zoom_controls(on_step: Callable[[bool], None], on_typed: Callable[[float], None]):
    """▲ ▼ buttons and the zoom text box of 3dmod toolbars. Returns (up, down, edit)."""
    up = tool_button(triangle_icon(True), tip="Increase zoom factor", on_click=lambda: on_step(True))
    down = tool_button(triangle_icon(False), tip="Decrease zoom factor", on_click=lambda: on_step(False))
    edit = QLineEdit()
    edit.setAlignment(Qt.AlignRight)
    edit.setFixedWidth(edit.fontMetrics().horizontalAdvance("00.0000") + 10)
    edit.setToolTip("Enter an arbitrary zoom factor")

    def typed() -> None:
        with contextlib.suppress(ValueError):
            on_typed(float(edit.text()))

    edit.editingFinished.connect(typed)
    return up, down, edit


def toolbar_row(*widgets, stretch_after: Optional[QWidget] = None) -> QWidget:
    """A compact horizontal toolbar strip like 3dmod's (no frame, tight spacing)."""
    bar = QWidget()
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(4, 2, 4, 2)
    lay.setSpacing(2)
    for w in widgets:
        if isinstance(w, int):
            lay.addSpacing(w)
        elif isinstance(w, str):
            lay.addWidget(QLabel(w))
        else:
            lay.addWidget(w, 1 if w is stretch_after else 0)
    return bar


# ---------------------------------------------------------------------------------------------- 3dmod toolbars
def make_toolbar(window: QMainWindow, title: str) -> QToolBar:
    """A 3dmod tool bar (utilMakeToolBar): top area, tight spacing, auto-raised 20-pixel buttons."""
    tb = QToolBar(title, window)
    tb.layout().setSpacing(0)
    tb.setAllowedAreas(Qt.TopToolBarArea)
    tb.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
    tb.setContextMenuPolicy(Qt.PreventContextMenu)
    window.addToolBar(tb)
    return tb


def tb_button(icon: QPixmap, tip: str = "", on_click=None) -> QToolButton:
    b = QToolButton()
    b.setAutoRaise(True)
    b.setFocusPolicy(Qt.NoFocus)
    b.setFixedSize(TOOLBUT_SIZE, TOOLBUT_SIZE)
    b.setIcon(QIcon(icon))
    b.setIconSize(icon.size())
    if tip:
        b.setToolTip(tip)
    if on_click is not None:
        b.clicked.connect(lambda _=False: on_click())
    return b


def tb_toggle(off: QPixmap, on: QPixmap, tip: str, checked: bool = False) -> "ToggleButton":
    b = ToggleButton(off, on, tip, checked)
    b.setFixedSize(TOOLBUT_SIZE, TOOLBUT_SIZE)
    return b


def tb_push(text: str, tip: str = "", on_click=None) -> QPushButton:
    """3dmod utilTBPushButton: a framed push button ("I", "Help")."""
    b = QPushButton(text)
    b.setFocusPolicy(Qt.NoFocus)
    b.setFixedWidth(max(b.fontMetrics().horizontalAdvance(text) + 12, TOOLBUT_SIZE + 4))
    if tip:
        b.setToolTip(tip)
    if on_click is not None:
        b.clicked.connect(lambda _=False: on_click())
    return b


def tool_edit(columns: int, tip: str = "") -> QLineEdit:
    """3dmod ToolEdit: a right-aligned line edit sized for ``columns`` digits."""
    e = QLineEdit()
    e.setAlignment(Qt.AlignRight)
    e.setFocusPolicy(Qt.ClickFocus)
    width = e.fontMetrics().horizontalAdvance("8" * columns)
    e.setFixedWidth(((2 * columns + 3) * width) // (2 * columns) + 6)
    if tip:
        e.setToolTip(tip)
    return e


def tb_zoom_tools(toolbar: QToolBar, on_step, on_typed) -> QLineEdit:
    """utilTBZoomTools: up/down arrows and a 6-column zoom box."""
    toolbar.addWidget(
        tb_button(triangle_icon(True), "Increase zoom factor (hot key = (equals))", lambda: on_step(True)),
    )
    toolbar.addWidget(
        tb_button(triangle_icon(False), "Decrease zoom factor (hot key - (minus))", lambda: on_step(False)),
    )
    edit = tool_edit(6, "Enter a zoom factor")

    def typed() -> None:
        with contextlib.suppress(ValueError):
            on_typed(float(edit.text()))

    edit.editingFinished.connect(typed)
    toolbar.addWidget(edit)
    return edit


def section_slider_width(n: int, slider) -> int:
    """3dmod: the section slider is as wide as the number of sections, between 20 and 100 pixels, plus its handle."""
    return min(max(n, 20), 100) + slider.minimumSizeHint().width() + 5
