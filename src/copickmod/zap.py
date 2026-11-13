"""The ZaP window: 3dmod's main XY slice viewer (xzap.cpp, zap_classes.cpp)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtCore import QPointF, QRectF, QSize, Qt
from qtpy.QtGui import QPainter, QPen
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QSlider, QWidget

from copickmod import keys
from copickmod.canvas import (
    ClickDragMixin,
    PlaneView,
    axis_center,
    fit_zoom,
    image_offset,
    pixel_center,
    trans_factor,
    zoom_step,
)
from copickmod.snapshot import quick_snapshot, snapshot_format
from copickmod.state import Session
from copickmod.style import (
    BACKGROUND,
    RED,
    arrow_icon,
    center_icon,
    checker_icon,
    insert_icon,
    keep_alive,
    lasso_icon,
    lock_icon,
    make_toolbar,
    rubberband_icon,
    section_slider_width,
    side_triangle_icon,
    tb_push,
    tb_toggle,
    tb_zoom_tools,
    tool_button,
    tool_edit,
    zoom_text,
)

KEYPAD_PAN = 5  # image pixels per keypad arrow (3dmod)


def available_size(widget=None):
    screen = QApplication.primaryScreen()
    geo = screen.availableGeometry() if screen is not None else None
    if geo is None or geo.width() < 100:
        return 1200, 900
    return geo.width(), geo.height()


def paint_arrow(p: QPainter, tail: QPointF, head: QPointF, scale: float = 1.0) -> None:
    """3dmod's red arrow annotation."""
    d = head - tail
    length = float(np.hypot(d.x(), d.y()))
    if length < 2:
        return
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(RED, 2 * scale))
    p.drawLine(tail, head)
    ux, uy = d.x() / length, d.y() / length
    size = min(12.0 * scale, length / 3)
    for sign in (1, -1):
        p.drawLine(head, QPointF(head.x() - size * (ux + sign * 0.5 * uy), head.y() - size * (uy - sign * 0.5 * ux)))
    p.setRenderHint(QPainter.Antialiasing, False)


class ZapCanvas(ClickDragMixin, QWidget):
    """The image area. Panning follows 3dmod: ``xtrans``/``ytrans`` are integer image-pixel offsets of the image
    centre; an image that fits the window is always centred, a larger one stops 1/6 of a window past its edges."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.view = PlaneView(session, axis=0)
        self.view.point_source = self.shown_point
        _, y, x = session.volume.shape
        aw, ah = available_size()
        self.zoom = fit_zoom(x, y, aw * 0.9, ah * 0.8)
        self.xtrans = 0
        self.ytrans = 0
        self.hq = True
        self.keep_centered = False
        self.locked_z: Optional[int] = None  # lock: keep showing this section
        self.band_mode = False
        self.band: Optional[tuple] = None  # rubber band (row0, col0, row1, col1) in world coordinates
        self.arrow_mode = False
        self.arrow: Optional[tuple] = None  # ((row, col) tail, (row, col) head)
        self._drawing = None  # ("band" | "arrow", row, col) while the button is down
        self._mouse = QPointF(0, 0)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(64, 64)
        session.point_changed.connect(self.update)
        session.center_on_point.connect(self.sync_to_point)
        session.image_changed.connect(self._image_changed)
        for sig in (session.contrast_changed, session.overlays_changed, session.data_arrived):
            sig.connect(self.update)

    # ------------------------------------------------------------------ geometry
    def shown_point(self):
        z, y, x = self.session.point
        return [self.locked_z if self.locked_z is not None else z, y, x]

    def sizeHint(self) -> QSize:
        return self.image_size()

    def image_size(self) -> QSize:
        aw, ah = available_size()
        _, y, x = self.session.volume.shape
        return QSize(int(min(max(x * self.zoom, 100), aw * 0.95)), int(min(max(y * self.zoom, 100), ah * 0.85)))

    def rect_f(self) -> QRectF:
        return QRectF(0, 0, self.width(), self.height())

    @property
    def center(self):
        """World (row = y, col = x) at the canvas centre; also clamps the pan like 3dmod."""
        _, ny, nx = self.session.volume.shape
        if self.keep_centered:
            _, py, px = self.session.point
            self.xtrans = int(nx * 0.5 - (px + 0.5) + 0.5)
            self.ytrans = int(ny * 0.5 - (py + 0.5) + 0.5)
        cx, self.xtrans = axis_center(self.width(), nx, self.zoom, self.xtrans)
        cy, self.ytrans = axis_center(self.height(), ny, self.zoom, self.ytrans)
        return [cy, cx]

    def to_screen(self, row: float, col: float, center=None) -> QPointF:
        return self.view.to_screen(self.rect_f(), self.zoom, center or self.center, row, col)

    def world_at(self, pos):
        return self.view.to_world(self.rect_f(), self.zoom, self.center, pixel_center(pos))

    def band_screen_rect(self) -> Optional[QRectF]:
        if self.band is None:
            return None
        r0, c0, r1, c1 = self.band
        return QRectF(self.to_screen(max(r0, r1), min(c0, c1)), self.to_screen(min(r0, r1), max(c0, c1))).normalized()

    # ------------------------------------------------------------------ painting
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), BACKGROUND)
        center = self.center
        complete = self.view.render(p, self.rect_f(), self.zoom, center, self.hq)
        s = self.session
        s.report_level(self.view.view_id, self.view.level, zap=True)
        self._report_subarea(center)
        p.setRenderHint(QPainter.Antialiasing, False)
        band = self.band_screen_rect()
        if band is not None:
            pen = QPen(RED, 1)
            pen.setDashPattern([4, 3])
            p.setPen(pen)
            p.drawRect(band)
        if self.arrow is not None:
            self._draw_arrow(p, center)
        if not complete:
            p.setPen(Qt.yellow)
            p.drawText(6, self.height() - 6, "loading...")
        p.end()
        win = self.window()
        if hasattr(win, "size_changed"):
            win.size_changed()

    def _draw_arrow(self, p: QPainter, center) -> None:
        (tr, tc), (hr, hc) = self.arrow
        paint_arrow(p, self.to_screen(tr, tc, center), self.to_screen(hr, hc, center))

    def _report_subarea(self, center) -> None:
        """Visible part of the section at the displayed level, for Float/Auto with Subarea on."""
        s = self.session
        vol = s.volume
        level = self.view.level
        top, left = self.view.to_world(self.rect_f(), self.zoom, center, QPointF(0, 0))
        bottom, right = self.view.to_world(self.rect_f(), self.zoom, center, QPointF(self.width(), self.height()))
        rows = (
            int(np.floor(vol.level_index(level, vol, bottom, 1))),
            int(np.ceil(vol.level_index(level, vol, top, 1))),
        )
        cols = (
            int(np.floor(vol.level_index(level, vol, left, 2))),
            int(np.ceil(vol.level_index(level, vol, right, 2))),
        )
        s.set_subarea_roi(level, rows, cols)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        win = self.window()
        if hasattr(win, "size_changed"):
            win.size_changed()

    # ------------------------------------------------------------------ zoom / pan / centring
    def set_zoom(self, zoom: float) -> None:
        self.zoom = float(min(max(zoom, 0.005), 50.0))
        self.update()
        win = self.window()
        if hasattr(win, "zoom_changed"):
            win.zoom_changed()

    def step_zoom(self, up: bool) -> None:
        self.set_zoom(zoom_step(self.zoom, up))

    def _image_changed(self, old) -> None:
        """Another tomogram is shown: keep the same field of view (in angstroms), lock, band and arrow."""
        s = self.session
        ratio = s.volume.voxel_size / old.voxel_size
        if abs(ratio - 1.0) > 1e-6:
            self.zoom = float(min(max(self.zoom * ratio, 0.005), 50.0))
            self.xtrans = int(round(self.xtrans / ratio))
            self.ytrans = int(round(self.ytrans / ratio))
        if self.locked_z is not None:
            z = int(np.floor(s.map_world(old, 0, self.locked_z + 0.5)))
            self.locked_z = min(max(z, 0), s.volume.shape[0] - 1)
        if self.band is not None:
            r0, c0, r1, c1 = self.band
            self.band = (
                s.map_world(old, 1, r0),
                s.map_world(old, 2, c0),
                s.map_world(old, 1, r1),
                s.map_world(old, 2, c1),
            )
        if self.arrow is not None:
            self.arrow = tuple((s.map_world(old, 1, r), s.map_world(old, 2, c)) for r, c in self.arrow)
        self.update()

    def translate(self, dx: int, dy: int) -> None:
        """3dmod ZapFuncs::translate: shift the image by image pixels (x right, y up)."""
        _, ny, nx = self.session.volume.shape
        self.xtrans = int(np.clip(self.xtrans + dx, -nx, nx))
        self.ytrans = int(np.clip(self.ytrans + dy, -ny, ny))
        self.update()

    def center_on(self, row: float, col: float) -> None:
        _, ny, nx = self.session.volume.shape
        self.xtrans = int(nx * 0.5 - col + 0.5)
        self.ytrans = int(ny * 0.5 - row + 0.5)

    def sync_to_point(self) -> None:
        """3dmod ZapFuncs::syncImage: recentre on the current point if it is near an edge (10% of the window,
        50 to 125 pixels) or outside, and recentring would actually move the image."""
        _, ny, nx = self.session.volume.shape
        _, py, px = self.session.point
        pos = self.to_screen(py + 0.5, px + 0.5)
        shift = False
        for wsize, n, p, wpos, trans in (
            (self.width(), nx, px, pos.x(), self.xtrans),
            (self.height(), ny, py, self.height() - pos.y(), self.ytrans),
        ):
            border = min(125, max(50, int(wsize * 0.1)))
            if wpos < border or wpos > wsize - border:
                try_trans = int(n * 0.5 - (p + 0.5) + 0.5)
                if image_offset(wsize, n, self.zoom, try_trans)[:2] != image_offset(wsize, n, self.zoom, trans)[:2]:
                    shift = True
        if shift:
            self.center_on(py + 0.5, px + 0.5)
            self.update()

    def set_section(self, z: int) -> None:
        """Show section ``z`` (0-based): the locked section if locked, otherwise the global current Z."""
        z = int(np.clip(z, 0, self.session.volume.shape[0] - 1))
        if self.locked_z is not None:
            self.locked_z = z
            self.update()
            self.window()._sync()
        else:
            self.session.set_axis(0, z)

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._press(event)
            if self.band_mode or self.arrow_mode:
                row, col = self.world_at(event.pos())
                self._drawing = ("band" if self.band_mode else "arrow", row, col)
        elif event.button() == Qt.MiddleButton:
            self.session.toggle_movie(1, 0)
        elif event.button() == Qt.RightButton:
            self.session.toggle_movie(-1, 0)

    def mouseMoveEvent(self, event) -> None:
        self._mouse = QPointF(event.pos())
        delta = self._move(event)
        if delta is None:
            return
        if self._drawing is not None:
            kind, r0, c0 = self._drawing
            row, col = self.world_at(event.pos())
            if kind == "band":
                self.band = (r0, c0, row, col)
                win = self.window()
                if hasattr(win, "size_changed"):
                    win.size_changed()  # the size box shows the band size while it is drawn
            else:
                self.arrow = ((r0, c0), (row, col))
            self.update()
            return
        fac = trans_factor(self.zoom)
        self.xtrans += int(np.floor(fac * delta[0] + 0.5))
        self.ytrans -= int(np.floor(fac * delta[1] + 0.5))
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        drawing, self._drawing = self._drawing, None
        if self._release_is_click(event):
            row, col = self.world_at(event.pos())
            z = self.locked_z if self.locked_z is not None else None
            self.session.set_point(z=z, y=int(np.floor(row)), x=int(np.floor(col)))
        elif drawing is not None:
            self.update()

    def wheelEvent(self, event) -> None:
        event.ignore()  # 3dmod does nothing with the wheel in movie mode

    # ------------------------------------------------------------------ keys
    def keyPressEvent(self, event) -> None:
        key, mods = event.key(), event.modifiers()
        shift = bool(mods & Qt.ShiftModifier)
        ctrl = bool(mods & Qt.ControlModifier)
        keypad = bool(mods & Qt.KeypadModifier)
        s = self.session
        win = self.window()
        if key == Qt.Key_Minus:
            self.step_zoom(False)
        elif key in (Qt.Key_Equal, Qt.Key_Plus):
            self.step_zoom(True)
        elif key == Qt.Key_R and shift and ctrl:
            self.zoom_to_band()
        elif key == Qt.Key_R and shift:
            if self.band is not None:
                win.fit_to_band()
            else:
                win.fit_to_image()
        elif key == Qt.Key_B and shift:
            win.band_button.toggle()
        elif key == Qt.Key_K and not shift:
            win.center_button.toggle()
        elif key == Qt.Key_Q and not ctrl:
            self._report_distance()
        elif snapshot_format(event):
            quick_snapshot(win, snapshot_format(event))
        elif key == Qt.Key_I and (shift or ctrl):
            self.report_info()
        elif key in (Qt.Key_PageUp, Qt.Key_PageDown) and not keypad and self.locked_z is not None:
            self.set_section(self.locked_z + (1 if key == Qt.Key_PageUp else -1))  # a locked ZaP steps alone
        # Keypad keys pan the image in movie mode (3dmod translate()).
        elif keypad and key in (Qt.Key_Left, Qt.Key_4):
            self.translate(-KEYPAD_PAN, 0)
        elif keypad and key in (Qt.Key_Right, Qt.Key_6):
            self.translate(KEYPAD_PAN, 0)
        elif keypad and key in (Qt.Key_Up, Qt.Key_8):
            self.translate(0, KEYPAD_PAN)
        elif keypad and key in (Qt.Key_Down, Qt.Key_2):
            self.translate(0, -KEYPAD_PAN)
        elif keypad and key in (Qt.Key_PageUp, Qt.Key_9):
            self.translate(KEYPAD_PAN, KEYPAD_PAN)
        elif keypad and key in (Qt.Key_PageDown, Qt.Key_3):
            self.translate(KEYPAD_PAN, -KEYPAD_PAN)
        elif keypad and key in (Qt.Key_Home, Qt.Key_7):
            self.translate(-KEYPAD_PAN, KEYPAD_PAN)
        elif keypad and key in (Qt.Key_End, Qt.Key_1):
            self.translate(-KEYPAD_PAN, -KEYPAD_PAN)
        elif key == Qt.Key_Escape:
            win.close()
        elif not keys.handle_global(s, event, axis=0):
            super().keyPressEvent(event)

    def zoom_to_band(self) -> None:
        """Ctrl+Shift+R: zoom so the rubber band fills the window, and turn the band off."""
        if self.band is None:
            return
        r0, c0, r1, c1 = self.band
        h, w = abs(r1 - r0), abs(c1 - c0)
        if h < 1 or w < 1:
            return
        self.center_on((r0 + r1) / 2, (c0 + c1) / 2)
        self.set_zoom(min(self.width() / w, self.height() / h))
        self.window().band_button.setChecked(False)

    def _report_distance(self) -> None:
        s = self.session
        row, col = self.world_at(self._mouse.toPoint())
        z, y, x = s.point
        dy, dx = row - (y + 0.5), col - (x + 0.5)
        d = float(np.hypot(dx, dy))
        s.message.emit(f"Distance = {d:.1f} pixels = {d * s.volume.voxel_size:.1f} A  (dx {dx:.1f}, dy {dy:.1f})")

    def report_info(self) -> None:
        s = self.session
        lv = self.view.level
        cy, cx = self.center
        s.message.emit(
            f"ZaP window {self.width()} x {self.height()}, zoom {self.zoom:g}, pyramid level {lv} "
            f"(bin {s.volume.factor(lv)[2]:g}), centre x {cx + 1:.0f} y {cy + 1:.0f}",
        )


class ZapWindow(QMainWindow):
    """Top-level ZaP window: 3dmod's tool bar above the image."""

    def __init__(self, session: Session):
        super().__init__(None)
        self.session = session
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"cmod ZaP Window: {session.title}")
        keep_alive(self)
        self.canvas = ZapCanvas(session, self)
        self.setCentralWidget(self.canvas)
        nz = session.volume.shape[0]

        tb = make_toolbar(self, "ZaP Toolbar")
        self.toolbar = tb
        self.zoom_edit = tb_zoom_tools(tb, self.canvas.step_zoom, self._zoom_typed)
        self.size_label = QLabel(" 0000x0000")
        tb.addWidget(self.size_label)
        self.hq_button = tb_toggle(
            checker_icon(False),
            checker_icon(True),
            "Toggle between regular and high-resolution (interpolated) image",
            checked=True,
        )
        self.hq_button.toggled.connect(self._hq_toggled)
        self.lock_button = tb_toggle(
            lock_icon(False),
            lock_icon(True),
            "Lock window at current section unless section is changed in this window",
        )
        self.lock_button.toggled.connect(self._lock_toggled)
        self.center_button = tb_toggle(
            center_icon(False),
            center_icon(True),
            "Toggle between centering when model point nears edge and keeping model point centered (K)",
        )
        self.center_button.toggled.connect(self._center_toggled)
        insert = tb_toggle(insert_icon(True), insert_icon(False), "Insert points before or after current point")
        self.band_button = tb_toggle(
            rubberband_icon(False),
            rubberband_icon(True),
            "Toggle rubber band on or off: drag with the first mouse button (Shift+B)",
        )
        self.band_button.toggled.connect(self._band_toggled)
        lasso = tb_toggle(lasso_icon(False), lasso_icon(True), "Toggle lasso for selecting contours")
        self.arrow_button = tb_toggle(
            arrow_icon(False),
            arrow_icon(True),
            "Toggle arrow: draw an arrow with the first mouse button",
        )
        self.arrow_button.toggled.connect(self._arrow_toggled)
        for b in (insert, lasso):  # model-mode tools: cmod is view-only
            b.setEnabled(False)
            b.setToolTip(b.toolTip() + " (model mode only; cmod is view-only)")
        for b in (
            self.hq_button,
            self.lock_button,
            self.center_button,
            insert,
            self.band_button,
            lasso,
            self.arrow_button,
        ):
            tb.addWidget(b)
        z_label = QLabel("Z")
        z_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tb.addWidget(z_label)
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(1, nz)
        self.z_slider.setFocusPolicy(Qt.NoFocus)
        self.z_slider.setFixedWidth(section_slider_width(nz, self.z_slider))
        self.z_slider.setToolTip("Select or riffle through sections")
        self.z_slider.valueChanged.connect(lambda v: self.canvas.set_section(v - 1))
        tb.addWidget(self.z_slider)
        self.z_edit = tool_edit(4, "Enter section to display")
        self.z_edit.editingFinished.connect(self._section_typed)
        tb.addWidget(self.z_edit)
        tb.addWidget(tb_push("I", "Bring Info Window to top, get Zap window info (hot key Ctrl+I)", self._info))
        tb.addWidget(tb_push("Help", "Open help window", lambda: session.open("help")))

        # The tomograms of this ZaP, on their own row (like 3dmod's time tool bar): shown with more than one.
        self.addToolBarBreak()
        self.image_bar = make_toolbar(self, "Tomogram Toolbar")
        self.image_bar.addWidget(
            tool_button(
                side_triangle_icon(False),
                tip="Previous tomogram (Left, or 1)",
                on_click=lambda: session.step_image(-1),
            ),
        )
        self.image_bar.addWidget(
            tool_button(
                side_triangle_icon(True),
                tip="Next tomogram (Right, or 2)",
                on_click=lambda: session.step_image(1),
            ),
        )
        self.image_combo = QComboBox()
        self.image_combo.setFocusPolicy(Qt.NoFocus)
        self.image_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.image_combo.setMinimumContentsLength(12)
        self.image_combo.setToolTip("Tomogram shown in this window (Left/Right arrows, or 1 and 2, switch)")
        self.image_combo.activated.connect(lambda i: (session.show_image(i), self.canvas.setFocus()))
        self.image_bar.addWidget(self.image_combo)

        session.point_changed.connect(self._sync)
        session.image_changed.connect(self._image_changed)
        session.images_changed.connect(self._fill_images)
        self.setAcceptDrops(True)  # drop MRC files to open them here
        self._fill_images()
        self._sync()
        self.zoom_changed()
        self.canvas.setFocus()

    def _fill_images(self) -> None:
        s = self.session
        combo = self.image_combo
        combo.blockSignals(True)
        combo.clear()
        for i, img in enumerate(s.images):
            combo.addItem(f"{i + 1}: {img.title}")
        combo.setCurrentIndex(s.current)
        combo.blockSignals(False)
        several = len(s.images) > 1
        if several != self.image_bar.isVisibleTo(self):
            grow = self.image_bar.sizeHint().height() if several and self.isVisible() else 0
            self.image_bar.setVisible(several)
            if grow:
                self.resize(self.width(), self.height() + grow)  # keep the size of the image area
        self.setWindowTitle(f"cmod ZaP Window: {s.title}")

    def _image_changed(self, old) -> None:
        nz = self.session.volume.shape[0]
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(1, nz)
        self.z_slider.blockSignals(False)
        self.z_slider.setFixedWidth(section_slider_width(nz, self.z_slider))
        self._fill_images()
        self._sync()
        self.zoom_changed()

    def dragEnterEvent(self, event) -> None:
        from copickmod.mrc import dropped_paths

        if dropped_paths(event):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        from copickmod.mrc import dropped_paths

        paths = dropped_paths(event)
        if paths:
            event.acceptProposedAction()
            self.session.open_files(paths)

    # ------------------------------------------------------------------ toolbar
    def _sync(self) -> None:
        z = self.canvas.shown_point()[0] + 1
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(z)
        self.z_slider.blockSignals(False)
        if not self.z_edit.hasFocus():
            self.z_edit.setText(str(z))

    def zoom_changed(self) -> None:
        self.zoom_edit.setText(zoom_text(self.canvas.zoom))

    def size_changed(self) -> None:
        band = self.canvas.band_screen_rect()
        w, h = (band.width(), band.height()) if band is not None else (self.canvas.width(), self.canvas.height())
        text = f" {int(w)}x{int(h)}"
        if self.size_label.text() != text:
            self.size_label.setText(text)

    def _zoom_typed(self, zoom: float) -> None:
        self.canvas.set_zoom(zoom)
        self.canvas.setFocus()

    def _section_typed(self) -> None:
        try:
            self.canvas.set_section(int(self.z_edit.text()) - 1)
        except ValueError:
            self._sync()
        self.canvas.setFocus()

    def _hq_toggled(self, on: bool) -> None:
        self.canvas.hq = on
        self.canvas.update()

    def _lock_toggled(self, on: bool) -> None:
        self.canvas.locked_z = self.session.point[0] if on else None
        self._sync()
        self.canvas.update()

    def _center_toggled(self, on: bool) -> None:
        self.canvas.keep_centered = on
        self.session.message.emit(f"ZaP: keep current point centered {'ON' if on else 'OFF'}")
        self.canvas.update()

    def _band_toggled(self, on: bool) -> None:
        self.canvas.band_mode = on
        if on:
            self.arrow_button.setChecked(False)
        else:
            self.canvas.band = None
        self.canvas.update()

    def _arrow_toggled(self, on: bool) -> None:
        self.canvas.arrow_mode = on
        if on:
            self.band_button.setChecked(False)
        else:
            self.canvas.arrow = None
        self.canvas.update()

    def _info(self) -> None:
        self.canvas.report_info()
        self.session.open("info")

    # ------------------------------------------------------------------ window
    def _bar_height(self) -> int:
        height = self.toolbar.sizeHint().height()
        if self.image_bar.isVisibleTo(self):
            height += self.image_bar.sizeHint().height()
        return height

    def fit_to_image(self) -> None:
        """Shift+R: resize the window so the whole image fits at the current zoom (the tool bar may be wider)."""
        hint = self.canvas.image_size()
        self.resize(hint.width(), hint.height() + self._bar_height())
        self.canvas.update()

    def fit_to_band(self) -> None:
        """Shift+R with a rubber band: resize the window to the band and turn the band off."""
        band = self.canvas.band_screen_rect()
        r0, c0, r1, c1 = self.canvas.band
        self.canvas.center_on((r0 + r1) / 2, (c0 + c1) / 2)
        self.resize(max(int(band.width()), 64), max(int(band.height()), 64) + self._bar_height())
        self.band_button.setChecked(False)

    def render_image(self, scale: int = 1, overlays: bool = True, cursor: bool = True, band: bool = True):
        """The ZaP image at ``scale`` times the window size (finer data), cropped to the rubber band if asked.

        Returns the image and its angstroms per pixel.
        """
        from copickmod.snapshot import render_until_complete

        c, s = self.canvas, self.session
        width, height, zoom = c.width() * scale, c.height() * scale, c.zoom * scale
        _, ny, nx = s.volume.shape
        center = [axis_center(height, ny, zoom, c.ytrans)[0], axis_center(width, nx, zoom, c.xtrans)[0]]
        rect = QRectF(0, 0, width, height)
        view = PlaneView(s, 0)
        view.point_source = c.shown_point

        def paint(p: QPainter) -> bool:
            p.fillRect(rect, BACKGROUND)
            complete = view.render(p, rect, zoom, center, c.hq, cursor=cursor, overlays=overlays)
            if overlays and c.arrow is not None:
                (tr, tc), (hr, hc) = c.arrow
                to_screen = view.to_screen
                paint_arrow(p, to_screen(rect, zoom, center, tr, tc), to_screen(rect, zoom, center, hr, hc), scale)
            return complete

        try:
            img = render_until_complete(paint, width, height, s)
        finally:
            s.drop_view(view.view_id)
        if band and c.band is not None:
            r0, c0, r1, c1 = c.band
            tl = view.to_screen(rect, zoom, center, max(r0, r1), min(c0, c1))
            br = view.to_screen(rect, zoom, center, min(r0, r1), max(c0, c1))
            img = img.copy(QRectF(tl, br).normalized().toAlignedRect().intersected(img.rect()))
        return img, s.volume.voxel_size / zoom

    def keyPressEvent(self, event) -> None:
        self.canvas.keyPressEvent(event)

    def closeEvent(self, event) -> None:
        from copickmod import settings

        settings.save_window("zap", self)
        self.session.drop_view(self.canvas.view.view_id)
        super().closeEvent(event)
